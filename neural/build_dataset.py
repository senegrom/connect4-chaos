"""Samples exactly-labelled training shards from solved pair tables.

Each shard is a torch .pt file of tensors: planes (N,7,10,10) float32,
legal (N,13) bool, policy (N,13) float32 (uniform over exactly-optimal
actions), wdl (N,) int64 (0 loss, 1 draw, 2 win). Sampling is uniform
over reachable states; labelling evaluates every child against the table,
so building shards is the CPU-heavy step and training stays GPU-bound.

Usage:
  python -m neural.build_dataset <out_dir> <samples_per_config> \
      <dir:rows:cols:connect:mode> [...]   (mode: chaos | classic)

DATASET_START_INDEX numbers the first shard (default 0); set it past
the shards a config already has to extend it rather than overwrite it.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import torch

from .chaos_game import ACTION_INDEX, ACTIONS, to_planes, successors
from .pair_tables import PairTable

SHARD = 25_000


def build(out_dir: Path, samples: int, spec: str, seed: int, start_index: int = 0) -> None:
    directory, rows, columns, connect, mode = spec.rsplit(":", 4)
    rows, columns, connect = int(rows), int(columns), int(connect)
    chaos = mode != 'classic'
    table = PairTable(directory, rows, columns, connect, chaos=chaos)
    rng = random.Random(seed)
    tag = f"{rows}x{columns}c{connect}{mode}"

    done = 0
    # Shards are numbered from start_index, so a later run extends a config
    # instead of rewriting it (shard 0000 of each config is the held-out
    # evaluation set and must never be regenerated). The seed moves with the
    # index so the new shards sample fresh positions.
    shard_index = start_index
    started = time.time()
    while done < samples:
        count = min(SHARD, samples - done)
        planes = torch.zeros((count, 7, 10, 10), dtype=torch.float32)
        legal = torch.zeros((count, 13), dtype=torch.bool)
        policy = torch.zeros((count, 13), dtype=torch.float32)
        wdl = torch.zeros((count,), dtype=torch.int64)
        # Exact value of every legal action for the mover (0 loss, 1 draw,
        # 2 win); 3 marks illegal actions and is ignored by the loss.
        q = torch.full((count, 13), 3, dtype=torch.int64)
        for i in range(count):
            state, value = table.sample_state(rng)
            edges = successors(state, connect, chaos=chaos)
            best = []
            for edge in edges:
                for_mover = table.edge_value_for_mover(edge)
                index = ACTION_INDEX[edge.action]
                legal[i][index] = True
                q[i][index] = for_mover + 1
                if for_mover == value:
                    best.append(index)
            planes[i] = torch.tensor(to_planes(state, connect, chaos=chaos),
                                     dtype=torch.float32)
            weight = 1.0 / len(best)
            for index in best:
                policy[i][index] = weight
            wdl[i] = value + 1
        out = out_dir / f"{tag}-{shard_index:04d}.pt"
        if out.exists():
            raise SystemExit(f"{out} already exists; set DATASET_START_INDEX past the existing shards")
        torch.save({"planes": planes, "legal": legal, "policy": policy,
                    "wdl": wdl, "q": q, "config": (rows, columns, connect)}, out)
        done += count
        shard_index += 1
        rate = done / max(1.0, time.time() - started)
        print(f"[{tag}] {done}/{samples} ({rate:.0f} samples/s)", flush=True)


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = int(sys.argv[2])
    start_index = int(os.environ.get("DATASET_START_INDEX", "0"))
    for index, spec in enumerate(sys.argv[3:]):
        build(out_dir, samples, spec, seed=977 + index + 7919 * start_index,
              start_index=start_index)
    print("dataset complete")


if __name__ == "__main__":
    main()
