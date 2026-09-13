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

from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass
import mmap
import os
import random
import re
import struct
from math import comb
from pathlib import Path

from .chaos_game import (
    DRAW, LOSS, NOT_TERMINAL, WIN, State, mask_has_line, successors,
)

HEADER = struct.Struct("<8s4BHHQ")
HEADER_BYTES = 24
GROUP_WORDS = 2048
MAX_MAPPED_BLOCKS = 32

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


@dataclass
class _PairBlock:
    bits: mmap.mmap
    ranks: mmap.mmap
    values: mmap.mmap | None
    slots: int
    count: int

    def close(self):
        for data in (self.bits, self.ranks, self.values):
            if data is not None:
                data.close()


class PairTable:
    """Validated mmap-backed lookup over an immutable solved table directory.

    Each block is checked once, before it is cached. Validation scans in bounded
    chunks, including every rank prefix and value byte; it never copies a whole
    block into RAM. Do not modify files in place while a reader is open.
    """

    def __init__(self, directory, rows: int, columns: int, connect: int,
                 chaos: bool = True):
        if (type(rows) is not int or type(columns) is not int
                or not 1 <= rows <= 7 or not 1 <= columns <= 7):
            raise ValueError("Pair-table dimensions must be integers from 1 to 7")
        if type(connect) is not int or not 1 <= connect <= max(rows, columns):
            raise ValueError("Pair-table connect length does not fit the board")
        if type(chaos) is not bool:
            raise ValueError("Pair-table chaos mode must be a bool")
        self.directory = Path(directory)
        self.geometry = Geometry(rows, columns, connect)
        self.chaos = chaos
        self._blocks = OrderedDict()
        self._checks = {}

    def close(self):
        for block in self._blocks.values():
            block.close()
        self._blocks.clear()
        self._checks.clear()
        if hasattr(self, '_block_weights'):
            del self._block_weights

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def _indices(self, pieces, pair_id):
        if (type(pieces) is not int or not 0 <= pieces <= self.geometry.cell_count
                or type(pair_id) is not int or not (pieces + 1) // 2 <= pair_id <= pieces):
            raise ValueError("Invalid pair-table layer or pair index")

    def _map(self, stack, path, size, header=None):
        # Check the opened file, not a separately stat'ed path. Close every map
        # if any companion file fails; failed loads must remain retryable.
        with path.open('rb') as handle:
            stat = os.fstat(handle.fileno())
            if stat.st_size != size:
                raise ValueError(f"{path}: incorrect payload size (expected {size} bytes)")
            if header is not None and handle.read(HEADER_BYTES) != HEADER.pack(*header):
                raise ValueError(f"{path}: table identity/header mismatch; check rules, layer and pair")
            data = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        stack.callback(data.close)
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        return data, signature

    @staticmethod
    def _count(bits, ranks, slots, prefix):
        count = 0
        for group in range(len(ranks) // 8):
            if struct.unpack_from('<Q', ranks, 8 * group)[0] != count:
                raise ValueError(f"{prefix}.ranks: rank prefix disagrees with the bitset")
            start = HEADER_BYTES + group * GROUP_WORDS * 8
            stop = min(start + GROUP_WORDS * 8, len(bits))
            count += int.from_bytes(bits[start:stop], 'little').bit_count()
        if slots % 64 and struct.unpack_from('<Q', bits, len(bits) - 8)[0] >> (slots % 64):
            raise ValueError(f"{prefix}.bits: set bits beyond the geometry's slot count")
        return count

    def _block(self, pieces, pair_id):
        self._indices(pieces, pair_id)
        key = (pieces, pair_id)
        if key in self._blocks:
            self._blocks.move_to_end(key)
            return self._blocks[key]
        slots = self.geometry.pair_slots(pieces, pair_id)
        words = (slots + 63) // 64
        groups = (words + GROUP_WORDS - 1) // GROUP_WORDS
        prefix = self.directory / f"pair-{pieces}-{pair_id}"
        identity = (b"C4PAIR2\0", self.geometry.rows, self.geometry.columns,
                    self.geometry.connect)
        kind = 0 if self.chaos else 2
        with ExitStack() as stack:
            bits, bits_id = self._map(stack, prefix.with_suffix('.bits'), HEADER_BYTES + 8 * words,
                                     (*identity, kind, pieces, pair_id, words))
            ranks, ranks_id = self._map(stack, prefix.with_suffix('.ranks'), 8 * groups)
            checked = self._checks.get(key)
            same_index = checked is not None and checked[0][:2] == (bits_id, ranks_id)
            count = checked[1] if same_index else self._count(bits, ranks, slots, prefix)
            values_path = prefix.with_suffix('.values')
            values = values_id = None
            # The native solver intentionally emits no values file for a block
            # with zero reachable states. A nonempty unresolved block is invalid.
            if count or values_path.exists():
                values, values_id = self._map(stack, values_path, HEADER_BYTES + count,
                                   (*identity, kind + 1, pieces, pair_id, count))
                if not same_index or checked[0][2] != values_id:
                    for start in range(HEADER_BYTES, len(values), 1 << 20):
                        if values[start:start + (1 << 20)].translate(None, b'\0\1\2'):
                            raise ValueError(f"{values_path}: invalid or unresolved WDL value")
            block = _PairBlock(bits, ranks, values, slots, count)
            # A large table has hundreds of files. Keep file descriptors bounded
            # without rescanning immutable blocks each time random sampling returns.
            while len(self._blocks) >= MAX_MAPPED_BLOCKS:
                _, oldest = self._blocks.popitem(last=False)
                oldest.close()
            self._checks[key] = ((bits_id, ranks_id, values_id), count)
            self._blocks[key] = block
            stack.pop_all()  # ownership transfers only after complete validation
            return block

    def has_block(self, pieces: int, pair_id: int) -> bool:
        self._indices(pieces, pair_id)
        return (self.directory / f"pair-{pieces}-{pair_id}.bits").exists()

    def validate(self):
        """Preflight all supplied blocks before sampling can publish a shard."""
        blocks = []
        try:
            for path in sorted(self.directory.glob('pair-*.bits')):
                match = re.fullmatch(r'pair-(0|[1-9][0-9]*)-(0|[1-9][0-9]*)\.bits', path.name)
                if match is None:
                    raise ValueError(f"{path}: invalid pair-table filename")
                pieces, pair_id = map(int, match.groups())
                count = self.block_count(pieces, pair_id)
                if count:
                    blocks.append((pieces, pair_id, count))
            if not blocks:
                raise ValueError(f"{self.directory}: no solved reachable pair-table states")
        except Exception:
            self.close()
            raise
        # Preserve the original numeric order of sampling weights.
        self._block_weights = sorted(blocks)
        self._total = sum(count for _, _, count in blocks)

    def block_count(self, pieces: int, pair_id: int) -> int:
        return self._block(pieces, pair_id).count

    def rank(self, pieces: int, pair_id: int, slot: int) -> int:
        block = self._block(pieces, pair_id)
        if type(slot) is not int or not 0 <= slot < block.slots:
            raise ValueError("Slot is outside the pair-table block")
        bits, ranks = block.bits, block.ranks
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
        return self._blocks[pieces, pair_id].values[HEADER_BYTES + rank] - 1

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
            self.validate()
        pick = rng.randrange(self._total)
        for pieces, pair_id, count in self._block_weights:
            if pick < count:
                break
            pick -= count
        bits = self._block(pieces, pair_id).bits
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
        for edge in successors(state, self.geometry.connect, chaos=self.chaos):
            if self.edge_value_for_mover(edge) == value:
                best.append(edge.action)
        return value, best
