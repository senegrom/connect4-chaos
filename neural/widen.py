"""Widen a checkpoint's trunk without changing what it computes.

A network that has plateaued can be given more capacity while keeping every
generation of training it already has: the wider network starts out
computing exactly the same function as the narrow one, and the new
channels grow from there under the normal learner.

The construction (Net2Net-style): every weight that *reads* a new channel
on the way to an old one is zero - the old channels of the stream only
ever see old channels through the copied weights, so the heads, which read
old channels only, are unchanged. Every weight that *produces* a new
channel keeps its fresh random initialisation, so no new channel is a
constant (BatchNorm in training mode would otherwise divide a zero
variance by its epsilon and amplify the first gradients ~300x). The zero
readers still receive gradient, because their inputs and their outputs'
gradients are both non-zero, so training switches the new channels on.

Usage:
  python -m neural.widen <in.pt> <out.pt> <channels>
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from .model import ACTIONS, CANVAS, PLANES, PolicyValueNet


def _copy_bn(src: nn.BatchNorm2d, dst: nn.BatchNorm2d, old: int) -> None:
    dst.weight[:old] = src.weight
    dst.bias[:old] = src.bias
    dst.running_mean[:old] = src.running_mean
    dst.running_var[:old] = src.running_var
    dst.num_batches_tracked.copy_(src.num_batches_tracked)


@torch.no_grad()
def widen(net: PolicyValueNet, channels: int) -> PolicyValueNet:
    old = net.channels
    if channels <= old:
        raise ValueError(f"{channels} channels is not wider than {old}")
    wide = PolicyValueNet(channels, net.blocks, net.head_channels)
    wide.stem[0].weight[:old] = net.stem[0].weight            # new stem filters stay random
    _copy_bn(net.stem[1], wide.stem[1], old)
    for block, wide_block in zip(net.tower, wide.tower):
        wide_block.conv1.weight[:old, :old] = block.conv1.weight
        wide_block.conv1.weight[:old, old:] = 0               # old features do not read the new stream
        _copy_bn(block.norm1, wide_block.norm1, old)
        wide_block.conv2.weight[:old, :old] = block.conv2.weight
        wide_block.conv2.weight[:old, old:] = 0               # old outputs do not read new features
        _copy_bn(block.norm2, wide_block.norm2, old)
    wide.column_features[0].weight[:, :old] = net.column_features[0].weight
    wide.column_features[0].weight[:, old:] = 0
    wide.column_features[1].load_state_dict(net.column_features[1].state_dict())
    wide.drop_logit.load_state_dict(net.drop_logit.state_dict())
    wide.drop_q.load_state_dict(net.drop_q.state_dict())
    wide.global_features[0].weight[:, :old] = net.global_features[0].weight
    wide.global_features[0].weight[:, old:] = 0
    wide.global_features[0].bias.copy_(net.global_features[0].bias)
    wide.transform_logit.load_state_dict(net.transform_logit.state_dict())
    wide.transform_q.load_state_dict(net.transform_q.state_dict())
    wide.value.load_state_dict(net.value.state_dict())
    return wide


@torch.no_grad()
def largest_difference(net: nn.Module, wide: nn.Module, seed: int = 7, batch: int = 64) -> float:
    generator = torch.Generator().manual_seed(seed)
    planes = torch.rand((batch, PLANES, CANVAS, CANVAS), generator=generator)
    legal = torch.rand((batch, ACTIONS), generator=generator) > 0.3
    legal[:, 0] = True
    worst = 0.0
    for a, b in zip(net.eval()(planes, legal), wide.eval()(planes, legal)):
        finite = torch.isfinite(a)
        worst = max(worst, float((a[finite] - b[finite]).abs().max()))
    return worst


def main() -> None:
    source, target, channels = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    payload = torch.load(source, map_location="cpu", weights_only=True)
    arch = tuple(payload.get("arch", (192, 12, 48)))
    net = PolicyValueNet(*arch)
    net.load_state_dict(payload["model"])
    wide = widen(net, channels)
    gap = largest_difference(net, wide)
    before = sum(p.numel() for p in net.parameters())
    after = sum(p.numel() for p in wide.parameters())
    print(f"{arch[0]} -> {channels} channels, {arch[1]} blocks: {before / 1e6:.1f}M -> {after / 1e6:.1f}M "
          f"parameters ({after / before:.2f}x); largest output difference {gap:.2e}", flush=True)
    if gap > 1e-4:
        raise SystemExit("the widened network does not reproduce the original")
    out = dict(payload)
    out["model"] = {key: value.clone() for key, value in wide.state_dict().items()}
    out["arch"] = (channels, arch[1], arch[2])
    out["widened_from"] = source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, target)
    print(f"saved {target} ({target.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
