"""Reads the pair solver's C4PAIR2 checkpoints as a labelled dataset.

Ports the slot arithmetic of native/perfect-chaos-paired.cpp (third
implementation after C++ and scripts/perfect-chaos-remote-lookup.mjs;
each validates the others): mirror-canonical compositions, pair colour
ranking, drops and transforms. On top of that: mmap-backed value lookup
via the .ranks sidecars (scripts/build-pair-rank-sidecars.py), uniform
sampling over reachable states, and exact policy targets from child
lookups. Solver tables exist only for boards up to 7x7.
"""

from __future__ import annotations

import mmap
import random
import struct
from math import comb
from pathlib import Path

from .chaos_game import (
    DRAW, LOSS, NOT_TERMINAL, WIN, State, mask_has_line, successors,
)

HEADER = struct.Struct("<8s4BHHQ")
HEADER_BYTES = 24
GROUP_WORDS = 2048

MAX_CELLS = 49
_BINOMIAL = [[0] * (MAX_CELLS + 1) for _ in range(MAX_CELLS + 1)]
for _n in range(MAX_CELLS + 1):
    _BINOMIAL[_n][0] = 1
    for _k in range(1, _n + 1):
        _BINOMIAL[_n][_k] = _BINOMIAL[_n - 1][_k - 1] + (_BINOMIAL[_n - 1][_k] if _k <= _n - 1 else 0)


def pair_of(pieces: int, mover_count: int) -> int:
    return max(mover_count, pieces - mover_count)


def _colour_rank(word: int) -> int:
    rank = 0
    seen = 0
    while word:
        position = (word & -word).bit_length() - 1
        seen += 1
        rank += _BINOMIAL[position][seen]
        word &= word - 1
    return rank


def _colour_unrank(rank: int, ones: int) -> int:
    word = 0
    for remaining in range(ones, 0, -1):
        position = remaining - 1
        while _BINOMIAL[position + 1][remaining] <= rank:
            position += 1
        rank -= _BINOMIAL[position][remaining]
        word |= 1 << position
    return word


class _Block:
    def __init__(self, rows: int, columns: int):
        self.rows = rows
        self.columns = columns
        self.canon = [[] for _ in range(rows * columns + 1)]
        self.rank_of = {}
        heights = [0] * columns
        while True:
            canonical = True
            for c in range(columns):
                mirrored = heights[columns - 1 - c]
                if heights[c] != mirrored:
                    canonical = heights[c] < mirrored
                    break
            if canonical:
                pieces = sum(heights)
                code = 0
                for c in range(columns):
                    code |= heights[c] << (3 * c)
                self.rank_of[code] = len(self.canon[pieces])
                self.canon[pieces].append(code)
            column = columns - 1
            while column >= 0 and heights[column] == rows:
                heights[column] = 0
                column -= 1
            if column < 0:
                break
            heights[column] += 1


class Geometry:
    def __init__(self, rows: int, columns: int, connect: int):
        self.rows, self.columns, self.connect = rows, columns, connect
        self.cell_count = rows * columns
        self.blocks = [_Block(rows, columns)]
        if rows != columns:
            self.blocks.append(_Block(columns, rows))

    def pair_colour_slots(self, pieces: int, pair_id: int) -> int:
        high = _BINOMIAL[pieces][pair_id]
        if pair_id * 2 == pieces:
            return high
        return high + _BINOMIAL[pieces][pieces - pair_id]

    def block_pair_slots(self, block: int, pieces: int, pair_id: int) -> int:
        return len(self.blocks[block].canon[pieces]) * self.pair_colour_slots(pieces, pair_id)

    def block_pair_offset(self, block: int, pieces: int, pair_id: int) -> int:
        return sum(self.block_pair_slots(b, pieces, pair_id) for b in range(block))

    def pair_slots(self, pieces: int, pair_id: int) -> int:
        return sum(self.block_pair_slots(b, pieces, pair_id) for b in range(len(self.blocks)))

    def block_index_for(self, rows: int, columns: int) -> int:
        for index, block in enumerate(self.blocks):
            if block.rows == rows and block.columns == columns:
                return index
        raise ValueError("shape outside geometry")


def canonical_pair_slot(geometry: Geometry, state: State, pair_id: int) -> int:
    block_index = geometry.block_index_for(state.rows, state.columns)
    block = geometry.blocks[block_index]
    columns, stride = block.columns, block.rows + 1
    heights, mover, pieces = state.heights, state.mover, state.pieces

    order = 0
    for column in range(columns):
        direct, mirrored = heights[column], heights[columns - 1 - column]
        if direct != mirrored:
            order = -1 if direct < mirrored else 1
            break
    if order == 0:
        for column in range(columns - 1, -1, -1):
            width = (1 << heights[column]) - 1
            direct = (mover >> (column * stride)) & width
            mirrored = (mover >> ((columns - 1 - column) * stride)) & width
            if direct != mirrored:
                order = 1 if direct > mirrored else -1
                break

    colours = 0
    offset = 0
    code = 0
    if order <= 0:
        for c in range(columns):
            code |= heights[c] << (3 * c)
            width = (1 << heights[c]) - 1
            colours |= ((mover >> (c * stride)) & width) << offset
            offset += heights[c]
    else:
        for column in range(columns):
            source = columns - 1 - column
            code |= heights[source] << (3 * column)
            width = (1 << heights[source]) - 1
            colours |= ((mover >> (source * stride)) & width) << offset
            offset += heights[source]
    rank = block.rank_of[code]

    ones = colours.bit_count()
    base = 0
    if ones != pair_id:
        if ones != pieces - pair_id:
            raise ValueError("colour word outside its pair")
        base = _BINOMIAL[pieces][pair_id]
    return (geometry.block_pair_offset(block_index, pieces, pair_id)
            + rank * geometry.pair_colour_slots(pieces, pair_id)
            + base + _colour_rank(colours))


def decode_pair_slot(geometry: Geometry, pieces: int, pair_id: int, slot: int) -> State:
    block_index = 0
    while (block_index + 1 < len(geometry.blocks)
           and slot >= geometry.block_pair_offset(block_index + 1, pieces, pair_id)):
        block_index += 1
    block = geometry.blocks[block_index]
    slot -= geometry.block_pair_offset(block_index, pieces, pair_id)

    colour_slots = geometry.pair_colour_slots(pieces, pair_id)
    composition_rank, sub = divmod(slot, colour_slots)
    mover_count = pair_id
    if sub >= _BINOMIAL[pieces][pair_id]:
        sub -= _BINOMIAL[pieces][pair_id]
        mover_count = pieces - pair_id
    colours = _colour_unrank(sub, mover_count)

    code = block.canon[pieces][composition_rank]
    heights = tuple((code >> (3 * c)) & 7 for c in range(block.columns))
    stride = block.rows + 1
    mover = 0
    opponent = 0
    rest = colours
    for column in range(block.columns):
        height = heights[column]
        occupied = (1 << height) - 1
        segment = rest & occupied
        rest >>= height
        mover |= segment << (column * stride)
        opponent |= (occupied ^ segment) << (column * stride)
    return State(block.rows, block.columns, mover, opponent, heights, pieces, mover_count)


class PairTable:
    """mmap-backed value lookup over one solved board's block files."""

    def __init__(self, directory, rows: int, columns: int, connect: int):
        self.directory = Path(directory)
        self.geometry = Geometry(rows, columns, connect)
        self._maps = {}

    def _mapped(self, name: str):
        if name not in self._maps:
            handle = open(self.directory / name, 'rb')
            self._maps[name] = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        return self._maps[name]

    def has_block(self, pieces: int, pair_id: int) -> bool:
        return (self.directory / f"pair-{pieces}-{pair_id}.bits").exists()

    def block_count(self, pieces: int, pair_id: int) -> int:
        ranks = self._mapped(f"pair-{pieces}-{pair_id}.ranks")
        bits = self._mapped(f"pair-{pieces}-{pair_id}.bits")
        words = HEADER.unpack(bits[:HEADER_BYTES])[7]
        last_group = (len(ranks) // 8) - 1
        base = struct.unpack_from("<Q", ranks, last_group * 8)[0]
        tail = bits[HEADER_BYTES + last_group * GROUP_WORDS * 8:HEADER_BYTES + words * 8]
        return base + int.from_bytes(tail, 'little').bit_count()

    def rank(self, pieces: int, pair_id: int, slot: int) -> int:
        bits = self._mapped(f"pair-{pieces}-{pair_id}.bits")
        ranks = self._mapped(f"pair-{pieces}-{pair_id}.ranks")
        word_index = slot // 64
        group = word_index // GROUP_WORDS
        rank = struct.unpack_from("<Q", ranks, group * 8)[0]
        start = HEADER_BYTES + group * GROUP_WORDS * 8
        span = bits[start:HEADER_BYTES + word_index * 8]
        rank += int.from_bytes(span, 'little').bit_count()
        word = struct.unpack_from("<Q", bits, HEADER_BYTES + word_index * 8)[0]
        bit = slot % 64
        if not (word >> bit) & 1:
            raise KeyError(f"slot {slot} of pair {pieces}-{pair_id} not reachable")
        return rank + (word & ((1 << bit) - 1)).bit_count()

    def value_at(self, pieces: int, pair_id: int, slot: int) -> int:
        rank = self.rank(pieces, pair_id, slot)
        values = self._mapped(f"pair-{pieces}-{pair_id}.values")
        return values[HEADER_BYTES + rank] - 1

    def value_of(self, state: State) -> int:
        pair_id = pair_of(state.pieces, state.mover_count)
        return self.value_at(pair_id=pair_id, pieces=state.pieces,
                             slot=canonical_pair_slot(self.geometry, state, pair_id))

    def edge_value_for_mover(self, edge) -> int:
        if edge.terminal != NOT_TERMINAL:
            return edge.terminal
        from_child = self.value_of(edge.child)
        return DRAW if from_child == DRAW else -from_child

    def sample_state(self, rng: random.Random):
        """Uniform over reachable states: pick a block weighted by its
        reachable count, then rejection-sample set bits inside it."""
        if not hasattr(self, '_block_weights'):
            blocks = []
            for pieces in range(self.geometry.cell_count + 1):
                for pair_id in range((pieces + 1) // 2, pieces + 1):
                    if self.has_block(pieces, pair_id):
                        count = self.block_count(pieces, pair_id)
                        if count:
                            blocks.append((pieces, pair_id, count))
            self._block_weights = blocks
            self._total = sum(b[2] for b in blocks)
        pick = rng.randrange(self._total)
        for pieces, pair_id, count in self._block_weights:
            if pick < count:
                break
            pick -= count
        bits = self._mapped(f"pair-{pieces}-{pair_id}.bits")
        slots = self.geometry.pair_slots(pieces, pair_id)
        while True:
            slot = rng.randrange(slots)
            word = struct.unpack_from("<Q", bits, HEADER_BYTES + (slot // 64) * 8)[0]
            if (word >> (slot % 64)) & 1:
                state = decode_pair_slot(self.geometry, pieces, pair_id, slot)
                return state, self.value_at(pieces, pair_id, slot)

    def labels(self, state: State):
        """(wdl_value, optimal_action_names) with exact child evaluations."""
        value = self.value_of(state)
        best = []
        for edge in successors(state, self.geometry.connect, chaos=True):
            if self.edge_value_for_mover(edge) == value:
                best.append(edge.action)
        return value, best
