"""Perfect distillation: trains the policy/value net on exact shards.

Losses: cross-entropy of the masked policy against the exactly-optimal
action distribution, cross-entropy of the W/D/L head against the exact
value. Reports, per config, value accuracy and blunder rate (argmax
action not in the optimal set) on held-out shards.

Usage: python -m neural.distill <shard_dir> <out_dir> [steps] [batch]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch import nn

from .model import PolicyValueNet


def mirror_batch(planes, legal, policy):
    """Horizontal mirror: flip columns; drops remap c -> 9-c; the two
    rotations swap (mirror conjugates them); flip is self-conjugate."""
    planes = torch.flip(planes, dims=(3,))
    order = list(range(9, -1, -1)) + [10, 12, 11]
    index = torch.tensor(order)
    return planes, legal[:, index], policy[:, index]


def load_shards(shard_dir: Path):
    train, held = [], []
    for path in sorted(shard_dir.glob("*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=True)
        (held if path.stem.endswith("0000") else train).append(shard)
    if not train:
        train, held = held, train
    return train, held


def main() -> None:
    shard_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20_000
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 512

    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, held = load_shards(shard_dir)
    planes = torch.cat([s["planes"] for s in train])
    legal = torch.cat([s["legal"] for s in train])
    policy = torch.cat([s["policy"] for s in train])
    wdl = torch.cat([s["wdl"] for s in train])
    print(f"train samples: {len(planes)}, held shards: {len(held)}, device: {device}")

    net = PolicyValueNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    generator = torch.Generator().manual_seed(20260901)

    started = time.time()
    for step in range(1, steps + 1):
        picks = torch.randint(0, len(planes), (batch,), generator=generator)
        b_planes, b_legal = planes[picks], legal[picks]
        b_policy, b_wdl = policy[picks], wdl[picks]
        if step % 2 == 0:
            b_planes, b_legal, b_policy = mirror_batch(b_planes, b_legal, b_policy)
        b_planes, b_legal = b_planes.to(device), b_legal.to(device)
        b_policy, b_wdl = b_policy.to(device), b_wdl.to(device)

        logits, values = net(b_planes, b_legal)
        log_probs = torch.log_softmax(logits, dim=1)
        policy_loss = -(b_policy * log_probs.masked_fill(~b_legal, 0.0)).sum(dim=1).mean()
        value_loss = nn.functional.cross_entropy(values, b_wdl)
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()

        if step % 500 == 0 or step == steps:
            print(f"step {step}/{steps} loss={loss.item():.4f} "
                  f"(policy {policy_loss.item():.4f}, value {value_loss.item():.4f}) "
                  f"{(time.time() - started):.0f}s", flush=True)

    net.eval()
    with torch.no_grad():
        for shard in held:
            h_planes = shard["planes"].to(device)
            h_legal = shard["legal"].to(device)
            logits, values = net(h_planes, h_legal)
            value_accuracy = (values.argmax(dim=1).cpu() == shard["wdl"]).float().mean()
            best = logits.argmax(dim=1).cpu()
            optimal = shard["policy"].gather(1, best.unsqueeze(1)).squeeze(1) > 0
            rows, columns, connect = shard["config"]
            print(f"[held {rows}x{columns} c{connect}] value accuracy "
                  f"{value_accuracy:.4f}, blunder rate {1.0 - optimal.float().mean():.4f}",
                  flush=True)

    torch.save({"model": net.state_dict(), "steps": steps}, out_dir / "distilled.pt")
    print(f"saved {out_dir / 'distilled.pt'}")


if __name__ == "__main__":
    main()
