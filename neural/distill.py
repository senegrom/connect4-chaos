"""Perfect distillation: trains the policy/value/Q net on exact shards.

Losses: cross-entropy of the masked policy against the exactly-optimal
action distribution, cross-entropy of the W/D/L head against the exact
value, and cross-entropy of the per-action Q head against the exact
value of every legal action. Reports, per held-out shard, value accuracy
and two blunder rates: the policy argmax's and the Q-argmax's (choosing
the action whose predicted outcome distribution has the best expectation).

Usage: python -m neural.distill <shard_dir> <out_dir> [steps] [batch]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

from .model import PolicyValueNet

MIRROR_ORDER = torch.tensor(list(range(9, -1, -1)) + [10, 12, 11])
OUTCOME_SCORE = torch.tensor([-1.0, 0.0, 1.0])   # loss, draw, win


def mirror_batch(planes, legal, policy, q):
    """Horizontal mirror: flip columns; drops remap c -> 9-c; the two
    rotations swap (mirror conjugates them); flip is self-conjugate."""
    planes = torch.flip(planes, dims=(3,))
    return planes, legal[:, MIRROR_ORDER], policy[:, MIRROR_ORDER], q[:, MIRROR_ORDER]


def load_shards(shard_dirs):
    """Exact shards (dir/*.pt, first shard of each config held out) plus any
    self-play replay shards; replay carries q=3 everywhere so only its
    policy and outcome supervise. Several directories may be given,
    separated by ';'."""
    # DISTILL_HOLDOUT_CONFIGS="6x6c4classic,5x6c4chaos" holds out every shard
    # of those configs: the board-level generalization test (no position
    # of that board is ever trained on).
    holdout = {tag for tag in os.environ.get("DISTILL_HOLDOUT_CONFIGS", "").split(",") if tag}
    train, held = [], []
    for shard_dir in str(shard_dirs).split(";"):
        for path in sorted(Path(shard_dir).glob("*.pt")):
            shard = torch.load(path, map_location="cpu", weights_only=True)
            if "q" not in shard:
                raise SystemExit(f"{path} predates the Q head; rebuild the dataset")
            replay = shard.get("source") == "selfplay"
            tag = path.stem.rsplit("-", 1)[0]
            whole_board_held = tag in holdout
            if whole_board_held and not path.stem.endswith("0000"):
                continue   # keep one shard per held-out board for evaluation
            (held if ((path.stem.endswith("0000") or whole_board_held) and not replay)
             else train).append(shard)
    if not train:
        train, held = held, train
    return train, held


def q_choice(q_logits, legal):
    """Action with the best expected outcome under the Q head."""
    expectation = (torch.softmax(q_logits, dim=2) * OUTCOME_SCORE.to(q_logits.device)).sum(dim=2)
    return expectation.masked_fill(~legal, float('-inf')).argmax(dim=1)


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
    q = torch.cat([s["q"] for s in train])
    print(f"train samples: {len(planes)}, held shards: {len(held)}, device: {device}")

    net = PolicyValueNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    generator = torch.Generator().manual_seed(20260901)

    started = time.time()
    for step in range(1, steps + 1):
        picks = torch.randint(0, len(planes), (batch,), generator=generator)
        b_planes, b_legal = planes[picks], legal[picks]
        b_policy, b_wdl, b_q = policy[picks], wdl[picks], q[picks]
        if step % 2 == 0:
            b_planes, b_legal, b_policy, b_q = mirror_batch(b_planes, b_legal, b_policy, b_q)
        b_planes, b_legal = b_planes.to(device), b_legal.to(device)
        b_policy, b_wdl, b_q = b_policy.to(device), b_wdl.to(device), b_q.to(device)

        logits, values, q_logits = net(b_planes, b_legal)
        log_probs = torch.log_softmax(logits, dim=1)
        policy_loss = -(b_policy * log_probs.masked_fill(~b_legal, 0.0)).sum(dim=1).mean()
        value_loss = nn.functional.cross_entropy(values, b_wdl)
        q_loss = nn.functional.cross_entropy(
            q_logits.reshape(-1, 3), b_q.reshape(-1), ignore_index=3)
        loss = policy_loss + value_loss + q_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()

        if step % 500 == 0 or step == steps:
            print(f"step {step}/{steps} loss={loss.item():.4f} "
                  f"(policy {policy_loss.item():.4f}, value {value_loss.item():.4f}, "
                  f"q {q_loss.item():.4f}) {(time.time() - started):.0f}s", flush=True)

    net.eval()
    with torch.no_grad():
        for shard in held:
            h_planes = shard["planes"].to(device)
            h_legal = shard["legal"].to(device)
            logits, values, q_logits = net(h_planes, h_legal)
            value_accuracy = (values.argmax(dim=1).cpu() == shard["wdl"]).float().mean()
            optimal = shard["policy"] > 0
            policy_pick = logits.argmax(dim=1).cpu()
            q_pick = q_choice(q_logits, h_legal).cpu()
            policy_ok = optimal.gather(1, policy_pick.unsqueeze(1)).squeeze(1).float().mean()
            q_ok = optimal.gather(1, q_pick.unsqueeze(1)).squeeze(1).float().mean()
            rows, columns, connect = shard["config"]
            print(f"[held {rows}x{columns} c{connect}] value accuracy {value_accuracy:.4f}, "
                  f"blunder rate policy {1.0 - policy_ok:.4f} / q {1.0 - q_ok:.4f}",
                  flush=True)

    torch.save({"model": net.state_dict(), "steps": steps}, out_dir / "distilled.pt")
    print(f"saved {out_dir / 'distilled.pt'}")


if __name__ == "__main__":
    main()
