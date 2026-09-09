"""Stable, position-disjoint validation, shared by exact data and replay.

Reserve one tenth of canonical positions, independently of shard names, RNG
seeds and append order. Horizontal reflections share a partition because
training uses that augmentation. Repetition variants are grouped together
conservatively; rules and mover-relative colours are part of the identity.
Rotations/flips are game moves (with gravity), not augmentation symmetries.

The version and hash must remain fixed when extending an existing dataset.
Legacy shards are filtered by the same rule at load time. This cannot undo
leakage in a model already trained on the old file-based split.
"""
from __future__ import annotations

from hashlib import blake2b
import math
import struct

import torch

SPLIT_VERSION = "position-blake2b-v1"
SPLIT_CHUNK = 4096
CANVAS = 10


def _reserved(rows, cols, connect, chaos, mover, opponent):
    payload = struct.pack("4B", rows, cols, connect, int(chaos))
    payload += mover.to_bytes(13, "big") + opponent.to_bytes(13, "big")
    digest = blake2b(payload, digest_size=8, person=b"c4-split-v1").digest()
    return int.from_bytes(digest, "big") % 10 == 0


def state_is_validation(state, connect, chaos):
    """Classify a scalar game State, without allocating its neural planes."""
    mask = (1 << state.rows) - 1
    normal = [0, 0]
    mirrored = [0, 0]
    for colour, word in enumerate((state.mover, state.opponent)):
        for col in range(state.columns):
            segment = (word >> (col * state.stride)) & mask
            normal[colour] |= segment << (col * CANVAS)
            mirrored[colour] |= segment << ((state.columns - 1 - col) * CANVAS)
    mover, opponent = min(tuple(normal), tuple(mirrored))
    return _reserved(state.rows, state.columns, connect, chaos, mover, opponent)


def validation_mask(planes, scale=None):
    """CPU bool mask for encoded float or scaled uint8 neural positions.

    Pack bits with tensor operations, then hash a small fixed-size identity per
    row. Chunking bounds scratch memory even for very large replay files.
    Ignore repetition planes so a replay variant of an exact validation
    position can never enter training.
    """
    if planes.device.type != "cpu" or planes.ndim != 4 or tuple(planes.shape[1:]) != (7, 10, 10):
        raise ValueError("Expected CPU planes with shape (N, 7, 10, 10)")
    scale = float(scale if scale is not None else (10 if planes.dtype == torch.uint8 else 1))
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("Invalid neural plane scale")
    result = torch.empty(len(planes), dtype=torch.bool)
    weights = torch.bitwise_left_shift(torch.ones(50, dtype=torch.int64), torch.arange(50))
    columns = torch.arange(CANVAS)[None, :]
    for start in range(0, len(planes), SPLIT_CHUNK):
        chunk = planes[start:start + SPLIT_CHUNK]
        n = len(chunk)
        rows = (chunk[:, 2, :, 0] > 0).sum(dim=1)
        cols = (chunk[:, 2, 0, :] > 0).sum(dim=1)
        connects = (chunk[:, 3, 0, 0].float() * (10 / scale)).round().long()
        modes = (chunk[:, 4, 0, 0] > 0).long()
        if bool(((rows < 1) | (rows > 10) | (cols < 1) | (cols > 10)
                 | (connects < 1) | (connects > 10)).any()):
            raise ValueError("Invalid board rules in neural planes")
        pieces = (chunk[:, :2] > 0) & (chunk[:, 2:3] > 0)
        source = torch.where(columns < cols[:, None], cols[:, None] - 1 - columns, columns)
        mirror = pieces.gather(3, source[:, None, None, :].expand_as(pieces))
        def limbs(bits):
            packed = bits.transpose(2, 3).reshape(n, 2, 2, 50)
            return (packed.long() * weights).sum(dim=3).reshape(n, 4)
        records = torch.cat((torch.stack((rows, cols, connects, modes), dim=1),
                             limbs(pieces), limbs(mirror)), dim=1).tolist()
        for offset, record in enumerate(records):
            r, c, k, mode, ml, mh, ol, oh, rml, rmh, rol, roh = record
            mover, opponent = min((ml | (mh << 50), ol | (oh << 50)),
                                  (rml | (rmh << 50), rol | (roh << 50)))
            result[start + offset] = _reserved(r, c, k, mode, mover, opponent)
    return result


SAMPLE_FIELDS = ("planes", "legal", "policy", "wdl", "q", "validation", "root_value")


def select_samples(shard, selection):
    """Slice every training tensor together while preserving shard metadata."""
    return {key: value[selection] if key in SAMPLE_FIELDS else value
            for key, value in shard.items()}
