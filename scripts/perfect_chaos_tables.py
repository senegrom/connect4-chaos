"""Strict binary table helpers shared by Perfect Chaos proof scripts."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

FRONTIER_MAGIC = b"C4CFRN1\0"
POLICY_MAGIC = b"C4CPOL1\0"
FRONTIER_RECORD_SIZE = 19
POLICY_RECORD_SIZE = 20
ROLE_CODES = {"red": 1, "yellow": 2}
ROLE_NAMES = {value: key for key, value in ROLE_CODES.items()}


@dataclass(frozen=True)
class Table:
    role: int
    boundary: int
    records: tuple[bytes, ...]


def record_key(record: bytes) -> tuple[int, int, int, int, int]:
    mover, opponent = struct.unpack_from("<QQ", record, 0)
    rows = record[16]
    columns = record[17]
    # Frontier records store aiTurn at byte 18. Policy records store action type,
    # but policy states are always AI-turn states.
    ai_turn = record[18] if len(record) == FRONTIER_RECORD_SIZE else 1
    return rows, columns, ai_turn, mover, opponent


def action_key(record: bytes) -> tuple[int, int]:
    if len(record) != POLICY_RECORD_SIZE:
        raise RuntimeError("Only policy records contain actions.")
    return record[18], record[19]


def _mirror_mask(mask: int, rows: int, columns: int) -> int:
    stride = rows + 1
    mirrored = 0
    for column in range(columns):
        target = columns - 1 - column
        for row in range(rows):
            source_bit = 1 << (column * stride + row)
            if mask & source_bit:
                mirrored |= 1 << (target * stride + row)
    return mirrored


def _validate_record(record: bytes, record_size: int, boundary: int) -> None:
    mover, opponent = struct.unpack_from("<QQ", record, 0)
    rows = record[16]
    columns = record[17]
    if (rows, columns) not in {(6, 7), (7, 6)}:
        raise RuntimeError("Perfect Chaos tables must use a 6×7 or 7×6 orientation.")
    stride = rows + 1
    used_bits = columns * stride
    if used_bits > 63 or (mover | opponent) >> used_bits:
        raise RuntimeError("Perfect Chaos record exceeds the sentinel mask boundary.")
    if mover & opponent:
        raise RuntimeError("Perfect Chaos mover and opponent masks overlap.")
    occupied = mover | opponent
    for column in range(columns):
        sentinel = 1 << (column * stride + rows)
        if occupied & sentinel:
            raise RuntimeError("Perfect Chaos record sets a sentinel bit.")
        column_bits = (occupied >> (column * stride)) & ((1 << rows) - 1)
        if column_bits & (column_bits + 1):
            raise RuntimeError("Perfect Chaos record violates gravity.")
    reflected_mover = _mirror_mask(mover, rows, columns)
    reflected_opponent = _mirror_mask(opponent, rows, columns)
    if reflected_mover < mover or (
        reflected_mover == mover and reflected_opponent < opponent
    ):
        raise RuntimeError("Perfect Chaos record is not horizontally canonical.")
    piece_count = occupied.bit_count()
    if record_size == FRONTIER_RECORD_SIZE:
        if record[18] not in (0, 1):
            raise RuntimeError("Perfect Chaos frontier has an invalid side-to-move flag.")
        if piece_count != boundary:
            raise RuntimeError("Perfect Chaos frontier record has the wrong piece count.")
    else:
        action_type = record[18]
        action_column = record[19]
        if piece_count >= boundary:
            raise RuntimeError("Perfect Chaos policy record lies at or beyond its frontier.")
        if action_type > 3:
            raise RuntimeError("Perfect Chaos policy record has an invalid action type.")
        if action_type == 0:
            if action_column >= columns:
                raise RuntimeError("Perfect Chaos policy drop column is out of range.")
        elif action_column != 0:
            raise RuntimeError("Perfect Chaos transform action must use column zero.")


def read_table(path: Path, magic: bytes, record_size: int) -> Table:
    data = path.read_bytes()
    if len(data) < 16 or data[:8] != magic:
        raise RuntimeError(f"Invalid table magic: {path}")
    if data[8] != 1 or data[11] != record_size:
        raise RuntimeError(f"Unsupported table header: {path}")
    role = data[9]
    boundary = data[10]
    count = struct.unpack_from("<I", data, 12)[0]
    if role not in ROLE_NAMES or boundary > 42:
        raise RuntimeError(f"Invalid table metadata: {path}")
    if len(data) != 16 + count * record_size:
        raise RuntimeError(f"Table length mismatch: {path}")
    records = tuple(
        data[16 + index * record_size : 16 + (index + 1) * record_size]
        for index in range(count)
    )
    for record in records:
        _validate_record(record, record_size, boundary)
    keys = [record_key(record) for record in records]
    if any(first >= second for first, second in zip(keys, keys[1:])):
        raise RuntimeError(f"Table records are not strictly sorted: {path}")
    return Table(role=role, boundary=boundary, records=records)


def merge_records(
    records: Iterable[bytes],
    record_size: int,
) -> tuple[list[bytes], int]:
    selected: dict[tuple[int, int, int, int, int], bytes] = {}
    conflicts = 0
    for record in records:
        if len(record) != record_size:
            raise RuntimeError("Record has the wrong size.")
        key = record_key(record)
        existing = selected.get(key)
        if record_size == POLICY_RECORD_SIZE and existing is not None:
            if action_key(record) != action_key(existing):
                conflicts += 1
            if action_key(record) < action_key(existing):
                selected[key] = record
        else:
            selected[key] = record
    return [selected[key] for key in sorted(selected)], conflicts


def encode_table(
    magic: bytes,
    role: int,
    boundary: int,
    record_size: int,
    records: Iterable[bytes],
) -> bytes:
    ordered, conflicts = merge_records(records, record_size)
    if record_size == POLICY_RECORD_SIZE and conflicts:
        raise RuntimeError("Conflicting Perfect Chaos policy actions.")
    header = bytearray(16)
    header[:8] = magic
    header[8] = 1
    header[9] = role
    header[10] = boundary
    header[11] = record_size
    struct.pack_into("<I", header, 12, len(ordered))
    return bytes(header) + b"".join(ordered)


def write_table(
    path: Path,
    magic: bytes,
    role: int,
    boundary: int,
    record_size: int,
    records: Iterable[bytes],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_table(magic, role, boundary, record_size, records))


def file_summary(path: Path) -> dict[str, int | str]:
    payload = path.read_bytes()
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
