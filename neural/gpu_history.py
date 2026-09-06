"""Dense GPU repetition history shared by self-play, arena and MCTS.

A drop starts a new repetition era because it permanently adds a stone. Between
drops we only need the sequence of previously visited hashes. Keeping that
sequence on-device avoids the per-ply ``.cpu().tolist()`` round trips and Python
dictionaries that otherwise serialize a batched GPU actor.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DenseHistoryView:
    """Compact live-game view passed repeatedly through one MCTS search."""

    hashes: torch.Tensor
    lengths: torch.Tensor


class DenseHistory:
    """Per-game sequence of hashes since the most recent drop.

    ``capacity`` only needs to cover the caller's ply guard. Old values are not
    cleared on a drop; ``lengths`` is the validity boundary, so resets are O(1).
    """

    def __init__(self, games: int, capacity: int, device):
        if games < 0 or capacity < 1:
            raise ValueError("history dimensions must be positive")
        self.device = torch.device(device)
        self.capacity = capacity
        self.hashes = torch.zeros((games, capacity), dtype=torch.int64, device=self.device)
        self.lengths = torch.zeros((games,), dtype=torch.int64, device=self.device)
        self.slots = torch.arange(capacity, dtype=torch.int64, device=self.device)

    def counts(self, game_ids, query_hashes):
        """Number of prior occurrences of each queried position.

        This full-width variant is used once after the real game move. MCTS uses
        ``search_view`` below so its many simulations compare only the live
        prefix rather than the whole ply guard.
        """
        game_ids = torch.as_tensor(game_ids, dtype=torch.int64, device=self.device)
        query_hashes = torch.as_tensor(query_hashes, dtype=torch.int64, device=self.device)
        lengths = self.lengths[game_ids]
        rows = self.hashes[game_ids]
        valid = self.slots[None, :] < lengths[:, None]
        return ((rows == query_hashes[:, None]) & valid).sum(dim=1)

    def append_or_reset(self, game_ids, hashes, drops):
        """Record the current position for transforms; reset after drops."""
        game_ids = torch.as_tensor(game_ids, dtype=torch.int64, device=self.device)
        hashes = torch.as_tensor(hashes, dtype=torch.int64, device=self.device)
        drops = torch.as_tensor(drops, dtype=torch.bool, device=self.device)
        transform = ~drops
        ids = game_ids[transform]
        positions = self.lengths[ids]
        # Empty advanced indexing is a no-op; no device->host truth check is
        # needed here. The outer game loop guarantees positions < capacity.
        self.hashes[ids, positions] = hashes[transform]
        self.lengths[ids] = positions + 1
        self.lengths[game_ids[drops]] = 0

    def search_view(self, game_ids):
        """Compact descriptor consumed repeatedly by ``history_counts``.

        One scalar synchronization per played ply finds the widest active era;
        this replaces several whole-batch CPU copies and prevents each MCTS
        simulation from comparing against the full 220-ply guard.
        """
        game_ids = torch.as_tensor(game_ids, dtype=torch.int64, device=self.device)
        lengths = self.lengths[game_ids]
        width = int(lengths.max().item()) if lengths.numel() else 0
        return DenseHistoryView(self.hashes[game_ids, :width], lengths)


def history_counts(history, query_hashes):
    """Count query hashes in dense GPU or legacy packed history."""
    if history is None:
        return torch.zeros_like(query_hashes, dtype=torch.int64)
    if isinstance(history, DenseHistoryView):
        slots = torch.arange(history.hashes.shape[1], dtype=torch.int64,
                             device=history.hashes.device)
        return ((history.hashes == query_hashes[:, None])
                & (slots[None, :] < history.lengths[:, None])).sum(dim=1)
    hashes, counts = history
    return ((hashes == query_hashes[:, None]) * counts).sum(dim=1)
