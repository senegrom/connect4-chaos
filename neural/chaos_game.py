"""Chaos/classic Connect-k game core on Python big-int bitboards.

A faithful port of the move semantics in native/perfect-chaos-paired.cpp
(and scripts/perfect-chaos-remote-lookup.mjs), with no 64-bit ceiling:
column stride is rows+1, so a 10x10 board is a 110-bit word and the same
shift-chain line detection applies unchanged. Mover-relative throughout:
`mover` is always the side to move, and a move hands the swapped masks to
the child. Cross-validated against the solved tables via
neural/pair_tables.py before any training use.
"""

from __future__ import annotations

from dataclasses import dataclass

WIN = 1
DRAW = 0
LOSS = -1
NOT_TERMINAL = 2

MAX_SIDE_NET = 10  # the network canvas; solvers stop at 7


@dataclass(frozen=True)
class State:
    rows: int
    columns: int
    mover: int
    opponent: int
    heights: tuple
    pieces: int
    mover_count: int

    @property
    def stride(self) -> int:
        return self.rows + 1


def empty_state(rows: int, columns: int) -> State:
    return State(rows, columns, 0, 0, (0,) * columns, 0, 0)


def mask_has_line(mask: int, rows: int, connect: int) -> bool:
    stride = rows + 1
    for shift in (1, stride, stride + 1, stride - 1):
        run = mask
        step = 1
        while step < connect and run:
            run &= mask >> (shift * step)
            step += 1
        if run:
            return True
    return False


def _reverse_segment(segment: int, height: int) -> int:
    reversed_bits = 0
    for bit in range(height):
        if (segment >> bit) & 1:
            reversed_bits |= 1 << (height - 1 - bit)
    return reversed_bits


@dataclass(frozen=True)
class Edge:
    """One legal move: either terminal (value for the parent's mover) or a
    child state (whose own mover is the parent's opponent)."""
    action: str            # 'drop0'..'drop9', 'flip', 'rotate_cw', 'rotate_ccw'
    terminal: int          # WIN/DRAW/LOSS for the parent's mover, or NOT_TERMINAL
    same_layer: bool
    child: State | None


def successors(state: State, connect: int, chaos: bool) -> list:
    rows, columns, stride = state.rows, state.columns, state.stride
    edges = []

    for column in range(columns):
        height = state.heights[column]
        if height >= rows:
            continue
        grown = state.mover | (1 << (column * stride + height))
        action = f"drop{column}"
        if mask_has_line(grown, rows, connect):
            edges.append(Edge(action, WIN, False, None))
            continue
        if state.pieces + 1 == rows * columns:
            edges.append(Edge(action, DRAW, False, None))
            continue
        child_heights = list(state.heights)
        child_heights[column] += 1
        edges.append(Edge(action, NOT_TERMINAL, False, State(
            rows, columns, state.opponent, grown, tuple(child_heights),
            state.pieces + 1, state.pieces - state.mover_count,
        )))

    if not chaos:
        return edges

    def settle(action, next_mover, next_opponent, next_rows, next_columns, next_heights):
        mover_line = mask_has_line(next_mover, next_rows, connect)
        opponent_line = mask_has_line(next_opponent, next_rows, connect)
        if mover_line or opponent_line:
            value = LOSS if (mover_line and opponent_line) else (WIN if mover_line else LOSS)
            edges.append(Edge(action, value, True, None))
            return
        edges.append(Edge(action, NOT_TERMINAL, True, State(
            next_rows, next_columns, next_opponent, next_mover, tuple(next_heights),
            state.pieces, state.pieces - state.mover_count,
        )))

    flipped_mover = 0
    flipped_opponent = 0
    for column in range(columns):
        height = state.heights[column]
        base = column * stride
        occupied = (1 << height) - 1
        segment = (state.mover >> base) & occupied
        reversed_bits = _reverse_segment(segment, height)
        flipped_mover |= reversed_bits << base
        flipped_opponent |= (occupied ^ reversed_bits) << base
    settle('flip', flipped_mover, flipped_opponent, rows, columns, state.heights)

    target_stride = columns + 1
    for clockwise in (True, False):
        rotated_mover = 0
        rotated_opponent = 0
        rotated_heights = [0] * rows
        for target_column in range(rows):
            height = 0
            if clockwise:
                for source_column in range(columns - 1, -1, -1):
                    if state.heights[source_column] <= target_column:
                        continue
                    bit = 1 << (target_column * target_stride + height)
                    if (state.mover >> (source_column * stride + target_column)) & 1:
                        rotated_mover |= bit
                    else:
                        rotated_opponent |= bit
                    height += 1
            else:
                source_row = rows - 1 - target_column
                for source_column in range(columns):
                    if state.heights[source_column] <= source_row:
                        continue
                    bit = 1 << (target_column * target_stride + height)
                    if (state.mover >> (source_column * stride + source_row)) & 1:
                        rotated_mover |= bit
                    else:
                        rotated_opponent |= bit
                    height += 1
            rotated_heights[target_column] = height
        settle('rotate_cw' if clockwise else 'rotate_ccw',
               rotated_mover, rotated_opponent, columns, rows, rotated_heights)

    return edges


def position_key(state: State) -> tuple:
    """Repetition identity: same board, same shape, same side structure."""
    return (state.rows, state.columns, state.mover, state.opponent)


def to_planes(state: State, connect: int, chaos: bool,
              repeated_once: bool = False, repeated_twice: bool = False):
    """Network input as a list of MAX_SIDE_NET x MAX_SIDE_NET float planes:
    mover, opponent, on-board mask, connect/10, chaos flag, rep1, rep2."""
    size = MAX_SIDE_NET
    planes = [[[0.0] * size for _ in range(size)] for _ in range(7)]
    stride = state.stride
    for column in range(state.columns):
        for row in range(state.rows):
            bit = 1 << (column * stride + row)
            if state.mover & bit:
                planes[0][row][column] = 1.0
            elif state.opponent & bit:
                planes[1][row][column] = 1.0
            planes[2][row][column] = 1.0
    for row in range(size):
        for column in range(size):
            planes[3][row][column] = connect / 10.0
            planes[4][row][column] = 1.0 if chaos else 0.0
            planes[5][row][column] = 1.0 if repeated_once else 0.0
            planes[6][row][column] = 1.0 if repeated_twice else 0.0
    return planes


ACTIONS = tuple([f"drop{c}" for c in range(MAX_SIDE_NET)] + ['flip', 'rotate_cw', 'rotate_ccw'])
ACTION_INDEX = {name: index for index, name in enumerate(ACTIONS)}
