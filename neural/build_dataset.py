"""Samples exactly-labelled training shards from solved pair tables.

Each shard is a torch .pt file of tensors: planes (N,7,10,10) float32,
legal (N,13) bool, policy (N,13) float32 (uniform over exactly-optimal
actions), wdl (N,) int64 (0 loss, 1 draw, 2 win). Sampling is uniform
over reachable states; labelling evaluates every child against the table,
so building shards is the CPU-heavy step and training stays GPU-bound.

Usage:
  python -m neural.build_dataset <out_dir> <samples_per_config> \
      <dir:rows:cols:connect> [...]
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch

from .chaos_game import ACTION_INDEX, ACTIONS, to_planes, successors
from .pair_tables import PairTable

SHARD = 25_000


def build(out_dir: Path, samples: int, spec: str, seed: int) -> None:
    directory, rows, columns, connect = spec.rsplit(":", 3)
    rows, columns, connect = int(rows), int(columns), int(connect)
    table = PairTable(directory, rows, columns, connect)
    rng = random.Random(seed)
    tag = f"{rows}x{columns}c{connect}"

    done = 0
    shard_index = 0
    started = time.time()
    while done < samples:
        count = min(SHARD, samples - done)
        planes = torch.zeros((count, 7, 10, 10), dtype=torch.float32)
        legal = torch.zeros((count, 13), dtype=torch.bool)
        policy = torch.zeros((count, 13), dtype=torch.float32)
        wdl = torch.zeros((count,), dtype=torch.int64)
        for i in range(count):
            state, value = table.sample_state(rng)
            edges = successors(state, connect, chaos=True)
            best = [e.action for e in edges
                    if table.edge_value_for_mover(e) == value]
            planes[i] = torch.tensor(to_planes(state, connect, chaos=True),
                                     dtype=torch.float32)
            for edge in edges:
                legal[i][ACTION_INDEX[edge.action]] = True
            weight = 1.0 / len(best)
            for action in best:
                policy[i][ACTION_INDEX[action]] = weight
            wdl[i] = value + 1
        out = out_dir / f"{tag}-{shard_index:04d}.pt"
        torch.save({"planes": planes, "legal": legal, "policy": policy,
                    "wdl": wdl, "config": (rows, columns, connect)}, out)
        done += count
        shard_index += 1
        rate = done / max(1.0, time.time() - started)
        print(f"[{tag}] {done}/{samples} ({rate:.0f} samples/s)", flush=True)


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = int(sys.argv[2])
    for index, spec in enumerate(sys.argv[3:]):
        build(out_dir, samples, spec, seed=977 + index)
    print("dataset complete")


if __name__ == "__main__":
    main()
