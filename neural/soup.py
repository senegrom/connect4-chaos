"""Average several checkpoints into one network of the same size.

Successive generations warm-start from one another and each generation ends
its own cosine schedule at a learning rate near zero, so the published
checkpoints are points along a single trajectory rather than independent
models. Averaging their weights (a "model soup", the same idea as stochastic
weight averaging) therefore tends to land in the middle of the basin they are
circling, which is usually flatter and generalises better than any one of
them - and unlike averaging their outputs it costs nothing at inference: one
network, one forward pass, the same ONNX file the browser downloads today.

BatchNorm is the one part that cannot simply be averaged. Its running mean
and variance describe the activations of the weights that produced them, so
after averaging they are recomputed from real positions (exact shards plus
self-play replay, the mix the learner trains on).

Usage:
  python -m neural.soup <out.pt> <shard_dirs> <model.pt> <model.pt> [more...]
  SOUP_BATCHES (default 200) sets how many batches the pass runs, and
  SOUP_POOL (default 800000) how many positions it draws them from.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from .distill import decode_planes, quantize_planes
from .data_split import validation_mask
from .model import PolicyValueNet


def average_state(paths, device="cpu"):
    """Mean of the checkpoints' float tensors; integer buffers come from the
    last one, since averaging a counter is meaningless."""
    payloads = [torch.load(path, map_location=device, weights_only=True) for path in paths]
    archs = {tuple(payload.get("arch", (192, 12, 48))) for payload in payloads}
    if len(archs) != 1:
        raise SystemExit(f"cannot average different architectures: {sorted(archs)}")
    states = [payload["model"] for payload in payloads]
    keys = set(states[0])
    if any(set(state) != keys for state in states):
        raise SystemExit("checkpoints do not share the same parameter names")
    averaged = {}
    for key in states[0]:
        values = [state[key] for state in states]
        if values[0].is_floating_point():
            averaged[key] = torch.stack([value.float() for value in values]).mean(dim=0).to(values[0].dtype)
        else:
            averaged[key] = values[-1].clone()
    return averaged, archs.pop(), payloads


def calibration_data(shard_dirs, pool=800_000, exact_share=0.25):
    """A bounded sample of the positions the learner trains on.

    Only the running statistics of BatchNorm are being estimated, so a few
    hundred thousand positions are ample; reading every exact shard to draw
    them cost twenty minutes. Replay comes first because it is where the
    large boards live, and the reserved validation positions are excluded so
    a soup is never calibrated on the data it will be scored against.
    """
    exact_target = int(pool * exact_share)
    replay, exact = [], []
    for shard_dir in str(shard_dirs).split(";"):
        for path in sorted(Path(shard_dir).glob("*.pt")):
            is_replay = path.stem.startswith("gpu-sp-")
            if not is_replay and path.stem.endswith("0000"):
                continue                       # the held-out shard of a board
            (replay if is_replay else exact).append(path)
    chosen = []
    for paths, target in ((replay, pool - exact_target), (exact, exact_target)):
        taken = 0
        for path in paths:
            if taken >= target:
                break
            shard = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            planes = quantize_planes(shard["planes"], shard.get("planes_scale"))
            keep = ~(shard["validation"].bool() if "validation" in shard
                     else validation_mask(planes, 10))
            planes, legal = planes[keep], shard["legal"][keep]
            if len(planes) > target - taken:
                planes, legal = planes[:target - taken], legal[:target - taken]
            chosen.append((planes.clone(), legal.clone()))
            taken += len(planes)
    if not chosen:
        raise SystemExit("no positions available to recalibrate BatchNorm")
    return torch.cat([p for p, _ in chosen]), torch.cat([l for _, l in chosen])


@torch.no_grad()
def recalibrate(net, planes, legal, device, batches=200, batch_size=1024, seed=20260908):
    """Recomputes BatchNorm running statistics for the averaged weights.

    momentum=None makes each BatchNorm accumulate a cumulative average over
    the pass, so the result depends on which positions are seen and not on
    the order they arrive in."""
    modules = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    if not modules:
        return 0
    for module in modules:
        module.reset_running_stats()
        module.momentum = None
    generator = torch.Generator().manual_seed(seed)
    net.train()
    seen = 0
    for _ in range(batches):
        picks = torch.randint(0, len(planes), (batch_size,), generator=generator)
        net(decode_planes(planes[picks]).to(device).float(), legal[picks].to(device))
        seen += batch_size
    net.eval()
    return seen


def main():
    out_path, shard_dirs, model_paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    if len(model_paths) < 2:
        raise SystemExit("give at least two checkpoints to average")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    averaged, arch, payloads = average_state(model_paths, device="cpu")
    net = PolicyValueNet(*arch).to(device)
    net.load_state_dict(averaged)
    names = ", ".join(Path(path).name for path in model_paths)
    print(f"averaged {len(model_paths)} checkpoints ({names}), arch {arch}", flush=True)
    planes, legal = calibration_data(shard_dirs, pool=int(os.environ.get("SOUP_POOL", "800000")))
    print(f"calibration pool: {len(planes)} positions", flush=True)
    seen = recalibrate(net, planes, legal, device, batches=int(os.environ.get("SOUP_BATCHES", "200")))
    print(f"recalibrated BatchNorm over {seen} sampled positions", flush=True)
    payload = {
        "model": {key: value.cpu() for key, value in net.state_dict().items()},
        "steps": max(int(p.get("steps", 0)) for p in payloads),
        "arch": arch,
        "data_split_version": payloads[-1].get("data_split_version", ""),
        "holdout_configs": payloads[-1].get("holdout_configs", ""),
        "soup": [Path(path).name for path in model_paths],
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".pt.partial")
    torch.save(payload, temporary)
    temporary.replace(out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
