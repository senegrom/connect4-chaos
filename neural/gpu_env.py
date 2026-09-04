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


class BoardBatch:
    """Column-major board tensors: mover/opponent (N,10,10) bool indexed
    [n, row, col]; heights (N,10); rows/cols/connect (N,); chaos (N,) bool."""

    def __init__(self, rows, cols, connect, chaos, device):
        n = len(rows)
        self.device = device
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
        r = torch.arange(CANVAS, device=self.device)
        return (r[None, :, None] < self.rows[:, None, None]) & (r[None, None, :] < self.cols[:, None, None])

    def legal(self):
        r = torch.arange(CANVAS, device=self.device)
        drops = (r[None, :] < self.cols[:, None]) & (self.heights < self.rows[:, None])
        transforms = self.chaos[:, None].expand(-1, 3)
        return torch.cat([drops, transforms], dim=1)

    def planes(self, rep1, rep2):
        n = len(self)
        region = self.region().float()
        ones = torch.ones((n, CANVAS, CANVAS), device=self.device)
        return torch.stack([
            self.mover.float(), self.opponent.float(), region,
            ones * (self.connect.float() / 10.0)[:, None, None],
            ones * self.chaos.float()[:, None, None],
            ones * rep1.float()[:, None, None],
            ones * rep2.float()[:, None, None],
        ], dim=1)

    def select(self, indices):
        """The sub-batch of the given games, in the given order. Self-play
        uses it to drop finished games so the tensors only ever carry live
        ones."""
        picked = BoardBatch.__new__(BoardBatch)
        picked.device = self.device
        for name in ("rows", "cols", "connect", "chaos", "mover", "opponent", "heights", "pieces"):
            setattr(picked, name, getattr(self, name).index_select(0, indices))
        return picked

    def clone(self):
        b = BoardBatch.__new__(BoardBatch)
        b.device = self.device
        for name in ("rows", "cols", "connect", "chaos", "mover", "opponent", "heights", "pieces"):
            setattr(b, name, getattr(self, name).clone())
        return b

    def position_hash(self, keys):
        """keys: (2,10,10) float64 random. Shape is folded in."""
        h = (self.mover.double() * keys[0]).sum((1, 2)) + (self.opponent.double() * keys[1]).sum((1, 2))
        return h + self.rows.double() * 1e6 + self.cols.double() * 1e7


def _shift(mask, dr, dc):
    """mask shifted so that out[r, c] = mask[r - dr, c - dc] (zeros outside)."""
    out = torch.zeros_like(mask)
    r0, r1 = max(dr, 0), CANVAS + min(dr, 0)
    c0, c1 = max(dc, 0), CANVAS + min(dc, 0)
    out[:, r0:r1, c0:c1] = mask[:, r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


def has_line(mask, connect):
    """True per game if `mask` holds a run of length connect[n] in any of
    the four directions (vertical, horizontal, both diagonals)."""
    result = torch.zeros(mask.shape[0], dtype=torch.bool, device=mask.device)
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        run = mask.clone()
        found = torch.zeros_like(result)
        for length in range(2, MAX_CONNECT + 1):
            run = run & _shift(mask, dr * (length - 1), dc * (length - 1))
            found |= (connect == length) & run.flatten(1).any(1)
        result |= found
    result |= (connect == 1) & mask.flatten(1).any(1)
    return result


def _hflip(plane, cols):
    """Reverse columns within each game's region."""
    c = torch.arange(CANVAS, device=plane.device)
    src = (cols[:, None] - 1 - c[None, :]).clamp(min=0, max=CANVAS - 1)     # (N,10)
    valid = c[None, :] < cols[:, None]
    gathered = plane.gather(2, src[:, None, :].expand(-1, CANVAS, -1))
    return gathered & valid[:, None, :]


def _vflip(plane, rows):
    """Reverse rows within each game's region."""
    r = torch.arange(CANVAS, device=plane.device)
    src = (rows[:, None] - 1 - r[None, :]).clamp(min=0, max=CANVAS - 1)
    valid = r[None, :] < rows[:, None]
    gathered = plane.gather(1, src[:, :, None].expand(-1, -1, CANVAS))
    return gathered & valid[:, :, None]


def _column_reverse(plane, heights):
    """Reverse the occupied part of every column: out[r, c] = in[h_c-1-r]
    for r < h_c, empty above."""
    r = torch.arange(CANVAS, device=plane.device)
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
    contents are unspecified."""
    n = len(board)
    idx = torch.arange(n, device=board.device)
    child = board.clone()
    outcome = torch.full((n,), NOT_TERMINAL, dtype=torch.int64, device=board.device)

    is_drop = action < 10
    # --- drops -------------------------------------------------------------
    if is_drop.any():
        col = action.clamp(max=9)
        row = board.heights[idx, col]
        # Illegal drops (full column, or column outside the board) may be
        # requested for masked-out games; keep their indexing in bounds.
        # The bound is the board's own height: a full column on a board
        # shorter than the canvas would otherwise take a stone above the
        # region, which has_line could read as a win.
        can = is_drop & (row < board.rows) & (col < board.cols)
        grown = board.mover.clone()
        grown[idx[can], row[can], col[can]] = True
        # Everything below is gated on `can`: a drop that is not actually
        # playable must leave both the board and the outcome untouched,
        # rather than advancing a height or reporting a full board.
        line = has_line(grown, board.connect) & can
        full = (board.pieces + 1 == board.rows * board.cols) & can & ~line
        outcome[line] = WIN
        outcome[full] = DRAW
        moving = can & ~line & ~full
        child.mover[moving] = board.opponent[moving]
        child.opponent[moving] = grown[moving]
        child.heights[idx[moving], col[moving]] += 1
        child.pieces[moving] += 1

    # --- transforms --------------------------------------------------------
    is_transform = ~is_drop
    if is_transform.any():
        next_mover = board.mover.clone()
        next_opponent = board.opponent.clone()
        next_rows = board.rows.clone()
        next_cols = board.cols.clone()

        f = action == FLIP
        if f.any():
            # The flip turns each column upside down: the stack order within
            # every column reverses, columns stay where they are.
            next_mover[f] = _column_reverse(board.mover, board.heights)[f]
            next_opponent[f] = _column_reverse(board.opponent, board.heights)[f]

        cw = action == ROT_CW
        if cw.any():
            m, o = _gravity(_hflip(board.mover, board.cols).transpose(1, 2),
                            _hflip(board.opponent, board.cols).transpose(1, 2))
            next_mover[cw], next_opponent[cw] = m[cw], o[cw]
            next_rows[cw], next_cols[cw] = board.cols[cw], board.rows[cw]

        ccw = action == ROT_CCW
        if ccw.any():
            m, o = _gravity(_vflip(board.mover, board.rows).transpose(1, 2),
                            _vflip(board.opponent, board.rows).transpose(1, 2))
            next_mover[ccw], next_opponent[ccw] = m[ccw], o[ccw]
            next_rows[ccw], next_cols[ccw] = board.cols[ccw], board.rows[ccw]

        mover_line = has_line(next_mover, board.connect) & is_transform
        opponent_line = has_line(next_opponent, board.connect) & is_transform
        outcome[is_transform & mover_line & opponent_line] = LOSS
        outcome[is_transform & mover_line & ~opponent_line] = WIN
        outcome[is_transform & ~mover_line & opponent_line] = LOSS
        moving = is_transform & ~mover_line & ~opponent_line
        child.mover[moving] = next_opponent[moving]
        child.opponent[moving] = next_mover[moving]
        child.rows[moving] = next_rows[moving]
        child.cols[moving] = next_cols[moving]
        occupied = child.mover | child.opponent
        child.heights[moving] = occupied[moving].long().sum(1)

    return child, outcome
