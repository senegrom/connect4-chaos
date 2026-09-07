"""Batched Connect-k environment on the GPU: any mix of board shapes (up to
10x10), connect lengths and rule sets in one batch, every transition a
tensor op. Semantics mirror neural/chaos_game.py exactly (verified by
tests): drops, the column flip (every column's stack reversed), both
rotations with gravity re-fall over the combined occupancy,
terminal rules, and mover-relative perspective (planes swap after every
move). Row 0 is the bottom row.

Outcome codes returned by `step`, for the player who just moved:
  NOT_TERMINAL = 2, WIN = 1, DRAW = 0, LOSS = -1.
"""

from __future__ import annotations

import torch

CANVAS = 10
ACTIONS = 13          # 10 drops, flip, rotate cw, rotate ccw
FLIP, ROT_CW, ROT_CCW = 10, 11, 12
NOT_TERMINAL, WIN, DRAW, LOSS = 2, 1, 0, -1
MAX_CONNECT = 10          # the canvas is 10x10, so no longer line exists
HASH_BITS = 46            # 100 cell keys sum below 2^53; shape and side sit above

_INDEX_CACHE = {}
_GAME_INDEX_CACHE = {}

def _indices(device):
    """Cached 0..9 tensor; these tiny allocations sit on every hot path."""
    key = str(torch.device(device))
    value = _INDEX_CACHE.get(key)
    if value is None:
        value = torch.arange(CANVAS, device=device)
        _INDEX_CACHE[key] = value
    return value


def _game_indices(n, device):
    """Cached row index for a fixed live-batch width."""
    key = (str(torch.device(device)), int(n))
    value = _GAME_INDEX_CACHE.get(key)
    if value is None:
        value = torch.arange(n, device=device)
        _GAME_INDEX_CACHE[key] = value
    return value


def hash_keys(device, generator=None):
    """Random per-cell keys for position_hash(); one set per run, shared by
    every history and every search in it."""
    return torch.randint(0, 1 << HASH_BITS, (2, CANVAS, CANVAS), dtype=torch.int64,
                         device=device, generator=generator)


class BoardBatch:
    """Column-major board tensors: mover/opponent (N,10,10) bool indexed
    [n, row, col]; heights (N,10); rows/cols/connect (N,); chaos (N,) bool."""

    def __init__(self, rows, cols, connect, chaos, device):
        n = len(rows)
        self.device = device
        self.max_connect = max(map(int, connect), default=1)
        self.any_chaos = any(map(bool, chaos))
        self.rows = torch.as_tensor(rows, dtype=torch.int64, device=device)
        self.cols = torch.as_tensor(cols, dtype=torch.int64, device=device)
        self.connect = torch.as_tensor(connect, dtype=torch.int64, device=device)
        self.chaos = torch.as_tensor(chaos, dtype=torch.bool, device=device)
        self.mover = torch.zeros((n, CANVAS, CANVAS), dtype=torch.bool, device=device)
        self.opponent = torch.zeros((n, CANVAS, CANVAS), dtype=torch.bool, device=device)
        self.heights = torch.zeros((n, CANVAS), dtype=torch.int64, device=device)
        self.pieces = torch.zeros((n,), dtype=torch.int64, device=device)

    def __len__(self):
        return len(self.rows)

    def region(self):
        r = _indices(self.device)
        return (r[None, :, None] < self.rows[:, None, None]) & (r[None, None, :] < self.cols[:, None, None])

    def legal(self):
        r = _indices(self.device)
        drops = (r[None, :] < self.cols[:, None]) & (self.heights < self.rows[:, None])
        transforms = self.chaos[:, None].expand(-1, 3)
        return torch.cat([drops, transforms], dim=1)

    def planes(self, rep1, rep2):
        n = len(self)
        region = self.region().float()
        shape = (n, CANVAS, CANVAS)
        return torch.stack([
            self.mover.float(), self.opponent.float(), region,
            (self.connect.float() / 10.0)[:, None, None].expand(shape),
            self.chaos.float()[:, None, None].expand(shape),
            rep1.float()[:, None, None].expand(shape),
            rep2.float()[:, None, None].expand(shape),
        ], dim=1)

    def select(self, indices):
        """The sub-batch of the given games, in the given order. Self-play
        uses it to drop finished games so the tensors only ever carry live
        ones."""
        picked = BoardBatch.__new__(BoardBatch)
        picked.device = self.device
        picked.max_connect = self.max_connect
        picked.any_chaos = self.any_chaos
        for name in ("rows", "cols", "connect", "chaos", "mover", "opponent", "heights", "pieces"):
            setattr(picked, name, getattr(self, name).index_select(0, indices))
        return picked

    def padded(self, width):
        """This batch followed by dummy games up to `width`: boards with no
        rows, no columns and no legal action, so a search never touches them.
        Padding to a few fixed widths is what lets the search replay graphs."""
        extra = width - len(self)
        picked = BoardBatch.__new__(BoardBatch)
        picked.device = self.device
        picked.max_connect = self.max_connect
        picked.any_chaos = self.any_chaos
        for name in ("rows", "cols", "connect", "chaos", "mover", "opponent", "heights", "pieces"):
            value = getattr(self, name)
            filler = torch.zeros((extra,) + tuple(value.shape[1:]), dtype=value.dtype, device=self.device)
            setattr(picked, name, torch.cat([value, filler]))
        return picked

    def clone(self):
        b = BoardBatch.__new__(BoardBatch)
        b.device = self.device
        b.max_connect = self.max_connect
        b.any_chaos = self.any_chaos
        for name in ("rows", "cols", "connect", "chaos", "mover", "opponent", "heights", "pieces"):
            setattr(b, name, getattr(self, name).clone())
        return b

    def position_hash(self, keys, side):
        """Position identity for the repetition rule: the stones (sum of
        the keys under them), the board shape and the side to move. The
        planes are mover-relative, so the same stones with the other side
        to move already hash differently, except on an empty board, which
        `side` (True when the second player is to move; a bool per game or
        one for all) tells apart: a flip there passes the turn without
        placing. Integer sums keep every bit; the float64 sum this replaces
        lost its low bits under the shape offsets."""
        side = torch.as_tensor(side, dtype=torch.int64, device=self.device)
        stones = ((self.mover.long() * keys[0]).sum((1, 2))
                  + (self.opponent.long() * keys[1]).sum((1, 2)))
        tag = self.rows + (CANVAS + 1) * self.cols + (CANVAS + 1) ** 2 * side
        return stones + tag * (1 << (HASH_BITS + 7))


def _shift(mask, dr, dc):
    """mask shifted so that out[r, c] = mask[r - dr, c - dc] (zeros outside)."""
    out = torch.zeros_like(mask)
    r0, r1 = max(dr, 0), CANVAS + min(dr, 0)
    c0, c1 = max(dc, 0), CANVAS + min(dc, 0)
    out[:, r0:r1, c0:c1] = mask[:, r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


def has_line(mask, connect, max_connect=MAX_CONNECT):
    """True per game if `mask` holds a run of length connect[n] in any of
    the four directions (vertical, horizontal, both diagonals)."""
    result = torch.zeros(mask.shape[0], dtype=torch.bool, device=mask.device)
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        run = mask.clone()
        found = torch.zeros_like(result)
        for length in range(2, min(MAX_CONNECT, max_connect) + 1):
            run = run & _shift(mask, dr * (length - 1), dc * (length - 1))
            found |= (connect == length) & run.flatten(1).any(1)
        result |= found
    result |= (connect == 1) & mask.flatten(1).any(1)
    return result


def _hflip(plane, cols):
    """Reverse columns within each game's region."""
    c = _indices(plane.device)
    src = (cols[:, None] - 1 - c[None, :]).clamp(min=0, max=CANVAS - 1)     # (N,10)
    valid = c[None, :] < cols[:, None]
    gathered = plane.gather(2, src[:, None, :].expand(-1, CANVAS, -1))
    return gathered & valid[:, None, :]


def _vflip(plane, rows):
    """Reverse rows within each game's region."""
    r = _indices(plane.device)
    src = (rows[:, None] - 1 - r[None, :]).clamp(min=0, max=CANVAS - 1)
    valid = r[None, :] < rows[:, None]
    gathered = plane.gather(1, src[:, :, None].expand(-1, -1, CANVAS))
    return gathered & valid[:, :, None]


def _column_reverse(plane, heights):
    """Reverse the occupied part of every column: out[r, c] = in[h_c-1-r]
    for r < h_c, empty above."""
    r = _indices(plane.device)
    src = (heights[:, None, :] - 1 - r[None, :, None]).clamp(min=0, max=CANVAS - 1)   # (N,10,10)
    valid = r[None, :, None] < heights[:, None, :]
    return plane.gather(1, src) & valid


def _gravity(mover, opponent):
    """Compact both colours to the bottom of every column, preserving the
    stack order of the combined occupancy (a piece never falls through a
    piece of the other colour)."""
    occupied = (mover | opponent).long()
    pos = (occupied.cumsum(1) - 1).clamp(min=0)
    out_m = torch.zeros_like(occupied)
    out_o = torch.zeros_like(occupied)
    out_m.scatter_add_(1, pos, mover.long())
    out_o.scatter_add_(1, pos, opponent.long())
    return out_m > 0, out_o > 0


def step(board: BoardBatch, action):
    """Applies one action per game. Returns (child, outcome) where outcome
    is per game for the mover who acted; for terminal games the child's
    contents are unspecified.

    Every write is a masked write over the whole batch: no boolean-mask
    indexing, so the shapes are fixed and nothing is read back to the host,
    which is what lets a search simulation replay as a CUDA graph."""
    n = len(board)
    idx = _game_indices(n, board.device)
    child = board.clone()
    outcome = torch.full((n,), NOT_TERMINAL, dtype=torch.int64, device=board.device)

    is_drop = action < 10
    # --- drops -------------------------------------------------------------
    col = action.clamp(min=0, max=9)
    height = board.heights[idx, col]
    # Illegal drops (full column, or column outside the board) may be
    # requested for masked-out games; keep their indexing in bounds.
    # The bound is the board's own height: a full column on a board
    # shorter than the canvas would otherwise take a stone above the
    # region, which has_line could read as a win.
    can = is_drop & (height < board.rows) & (col < board.cols)
    stone = torch.zeros_like(board.mover)
    stone[idx, height.clamp(max=CANVAS - 1), col] = can
    grown = board.mover | stone
    # Everything below is gated on `can`: a drop that is not actually
    # playable must leave both the board and the outcome untouched,
    # rather than advancing a height or reporting a full board.
    line = has_line(grown, board.connect, board.max_connect) & can
    full = (board.pieces + 1 == board.rows * board.cols) & can & ~line
    outcome = torch.where(line, torch.full_like(outcome, WIN), outcome)
    outcome = torch.where(full, torch.full_like(outcome, DRAW), outcome)
    moving = can & ~line & ~full
    moving3 = moving[:, None, None]
    child.mover = torch.where(moving3, board.opponent, board.mover)
    child.opponent = torch.where(moving3, grown, board.opponent)
    child.heights[idx, col] += moving.long()
    child.pieces = board.pieces + moving.long()

    # --- transforms --------------------------------------------------------
    if board.any_chaos:
        is_transform = ~is_drop
        flip, cw, ccw = action == FLIP, action == ROT_CW, action == ROT_CCW
        # The flip turns each column upside down: the stack order within
        # every column reverses, columns stay where they are. Rotations
        # re-fall under gravity and swap the board's dimensions. All three
        # are computed for the whole batch and selected per game.
        flipped_m = _column_reverse(board.mover, board.heights)
        flipped_o = _column_reverse(board.opponent, board.heights)
        cw_m, cw_o = _gravity(_hflip(board.mover, board.cols).transpose(1, 2),
                              _hflip(board.opponent, board.cols).transpose(1, 2))
        ccw_m, ccw_o = _gravity(_vflip(board.mover, board.rows).transpose(1, 2),
                                _vflip(board.opponent, board.rows).transpose(1, 2))
        flip3, cw3, ccw3 = flip[:, None, None], cw[:, None, None], ccw[:, None, None]
        next_mover = torch.where(flip3, flipped_m, torch.where(cw3, cw_m, torch.where(ccw3, ccw_m, board.mover)))
        next_opponent = torch.where(flip3, flipped_o, torch.where(cw3, cw_o, torch.where(ccw3, ccw_o, board.opponent)))
        rotated = cw | ccw
        next_rows = torch.where(rotated, board.cols, board.rows)
        next_cols = torch.where(rotated, board.rows, board.cols)

        mover_line = has_line(next_mover, board.connect, board.max_connect) & is_transform
        opponent_line = has_line(next_opponent, board.connect, board.max_connect) & is_transform
        outcome = torch.where(is_transform & mover_line & opponent_line, torch.full_like(outcome, LOSS), outcome)
        outcome = torch.where(is_transform & mover_line & ~opponent_line, torch.full_like(outcome, WIN), outcome)
        outcome = torch.where(is_transform & ~mover_line & opponent_line, torch.full_like(outcome, LOSS), outcome)
        turning = is_transform & ~mover_line & ~opponent_line
        turning3 = turning[:, None, None]
        child.mover = torch.where(turning3, next_opponent, child.mover)
        child.opponent = torch.where(turning3, next_mover, child.opponent)
        child.rows = torch.where(turning, next_rows, child.rows)
        child.cols = torch.where(turning, next_cols, child.cols)
        occupied = child.mover | child.opponent
        child.heights = torch.where(turning[:, None], occupied.long().sum(1), child.heights)

    return child, outcome
