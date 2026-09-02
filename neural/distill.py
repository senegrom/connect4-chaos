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


def decode_planes(planes):
    """Shard planes are float32 (exact shards) or uint8 scaled by 10
    (self-play shards); either way float16 is what training uses."""
    if planes.dtype == torch.uint8:
        return planes.half() / 10
    return planes.half()


def load_shards(shard_dirs):
    """Exact shards (dir/*.pt, first shard of each config held out) plus any
    self-play replay shards; replay carries q=3 everywhere so only its
    policy and outcome supervise. Several directories may be given,
    separated by ';'."""
    # DISTILL_HOLDOUT_CONFIGS="6x6c4classic,5x6c4chaos" holds out every shard
    # of those configs: the board-level generalization test (no position
    # of that board is ever trained on).
    holdout = {tag for tag in os.environ.get("DISTILL_HOLDOUT_CONFIGS", "").split(",") if tag}
    # Shards are memory-mapped: nothing is read until it is copied into the
    # training buffers, so loading costs no float32 peak in host RAM.
    train, held = [], []
    for shard_dir in str(shard_dirs).split(";"):
        for path in sorted(Path(shard_dir).glob("*.pt")):
            shard = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if "q" not in shard:
                raise SystemExit(f"{path} predates the Q head; rebuild the dataset")
            shard["mtime"] = path.stat().st_mtime
            replay = shard.get("source") == "selfplay"
            tag = path.stem.rsplit("-", 1)[0]
            whole_board_held = tag in holdout
            if whole_board_held and not path.stem.endswith("0000"):
                continue   # keep one shard per held-out board for evaluation
            (held if ((path.stem.endswith("0000") or whole_board_held) and not replay)
             else train).append(shard)
    if not train:
        train, held = held, train
    # Replay window (AlphaZero-style): only the newest DISTILL_REPLAY_WINDOW
    # self-play positions train; older shards age out, which also bounds
    # host RAM as the actor keeps producing.
    window = int(os.environ.get("DISTILL_REPLAY_WINDOW", "4000000"))
    replay_shards = sorted((s for s in train if s.get("source") == "selfplay"),
                           key=lambda s: s["mtime"], reverse=True)
    kept, total = [], 0
    for s in replay_shards:
        if total >= window:
            break
        kept.append(s)
        total += len(s["planes"])
    dropped = len(replay_shards) - len(kept)
    if dropped:
        print(f"replay window {window}: keeping newest {len(kept)} shards ({total} positions), "
              f"dropping {dropped} older shards")
    kept_ids = {id(s) for s in kept}
    train = [s for s in train if s.get("source") != "selfplay" or id(s) in kept_ids]
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
    # Planes live in host RAM as float16 inside preallocated buffers (no
    # concatenation copies): ~2M positions cost ~3 GB instead of ~11 GB.
    total = sum(len(s["planes"]) for s in train)
    planes = torch.empty((total, 7, 10, 10), dtype=torch.float16)
    legal = torch.empty((total, 13), dtype=torch.bool)
    policy = torch.empty((total, 13), dtype=torch.float32)
    wdl = torch.empty((total,), dtype=torch.int64)
    q = torch.empty((total, 13), dtype=torch.int64)
    cursor = 0
    for s in train:
        count = len(s["planes"])
        s["count"] = count
        planes[cursor:cursor + count] = decode_planes(s["planes"])
        legal[cursor:cursor + count] = s["legal"]
        policy[cursor:cursor + count] = s["policy"]
        wdl[cursor:cursor + count] = s["wdl"]
        q[cursor:cursor + count] = s["q"]
        cursor += count
        for key in ("planes", "legal", "policy", "wdl", "q"):
            s[key] = None      # release the mapping once copied
    # Replay-majority batches: DISTILL_REPLAY_FRACTION of every batch comes
    # from self-play shards (the only data for boards without tables), the
    # rest from exact shards. Falls back to all-exact when no replay exists.
    is_replay = torch.cat([torch.full((s["count"],), s.get("source") == "selfplay")
                           for s in train])
    replay_idx = is_replay.nonzero().squeeze(1)
    exact_idx = (~is_replay).nonzero().squeeze(1)
    replay_fraction = float(os.environ.get("DISTILL_REPLAY_FRACTION", "0.75"))
    if len(replay_idx) == 0 or len(exact_idx) == 0:
        replay_fraction = 1.0 if len(exact_idx) == 0 else 0.0
    print(f"train samples: {len(planes)} (exact {len(exact_idx)}, replay {len(replay_idx)}, "
          f"replay fraction {replay_fraction:.2f}), held shards: {len(held)}, device: {device}")

    init = os.environ.get("DISTILL_INIT")
    if init:
        payload = torch.load(init, map_location=device, weights_only=True)
        net = PolicyValueNet(*payload.get("arch", (192, 12, 48))).to(device)
        net.load_state_dict(payload["model"])
        print(f"warm start from {init} arch={payload.get('arch', (192, 12, 48))}")
    else:
        net = PolicyValueNet().to(device)
    print(f"architecture: {net.channels} channels x {net.blocks} blocks, "
          f"{sum(p.numel() for p in net.parameters())/1e6:.2f}M params")
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(os.environ.get("DISTILL_LR", "1e-3")),
                                  weight_decay=1e-4)
    # bf16 autocast for the forward pass (losses stay float32): the same
    # step costs roughly half the GPU time. DISTILL_FP32=1 disables it.
    use_amp = device == "cuda" and os.environ.get("DISTILL_FP32", "") != "1"
    torch.backends.cudnn.benchmark = True
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    generator = torch.Generator().manual_seed(20260901)

    started = time.time()
    for step in range(1, steps + 1):
        n_replay = int(round(batch * replay_fraction))
        picks = torch.cat([
            replay_idx[torch.randint(0, max(1, len(replay_idx)), (n_replay,), generator=generator)]
            if n_replay else torch.empty(0, dtype=torch.int64),
            exact_idx[torch.randint(0, max(1, len(exact_idx)), (batch - n_replay,), generator=generator)]
            if batch - n_replay else torch.empty(0, dtype=torch.int64),
        ])
        b_planes, b_legal = planes[picks], legal[picks]
        b_policy, b_wdl, b_q = policy[picks], wdl[picks], q[picks]
        if step % 2 == 0:
            b_planes, b_legal, b_policy, b_q = mirror_batch(b_planes, b_legal, b_policy, b_q)
        b_planes, b_legal = b_planes.to(device).float(), b_legal.to(device)
        b_policy, b_wdl, b_q = b_policy.to(device), b_wdl.to(device), b_q.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits, values, q_logits = net(b_planes, b_legal)
        logits, values, q_logits = logits.float(), values.float(), q_logits.float()
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

    # Save before evaluating: the checkpoint must never depend on the
    # evaluation surviving a crowded GPU.
    torch.save({"model": net.state_dict(), "steps": steps,
                "arch": (net.channels, net.blocks, net.head_channels)}, out_dir / "distilled.pt")
    print(f"saved {out_dir / 'distilled.pt'}", flush=True)

    net.eval()
    with torch.no_grad():
        for shard in held:
            value_hits = policy_hits = q_hits = 0
            optimal = shard["policy"] > 0
            for start in range(0, len(shard["planes"]), 4096):
                h_planes = decode_planes(shard["planes"][start:start + 4096]).to(device).float()
                h_legal = shard["legal"][start:start + 4096].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits, values, q_logits = net(h_planes, h_legal)
                logits, values, q_logits = logits.float(), values.float(), q_logits.float()
                value_hits += (values.argmax(dim=1).cpu() == shard["wdl"][start:start + 4096]).sum().item()
                policy_pick = logits.argmax(dim=1).cpu()
                q_pick = q_choice(q_logits, h_legal).cpu()
                chunk_optimal = optimal[start:start + 4096]
                policy_hits += chunk_optimal.gather(1, policy_pick.unsqueeze(1)).sum().item()
                q_hits += chunk_optimal.gather(1, q_pick.unsqueeze(1)).sum().item()
            count = len(shard["planes"])
            rows, columns, connect = shard["config"]
            print(f"[held {rows}x{columns} c{connect}] value accuracy {value_hits / count:.4f}, "
                  f"blunder rate policy {1.0 - policy_hits / count:.4f} / q {1.0 - q_hits / count:.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
