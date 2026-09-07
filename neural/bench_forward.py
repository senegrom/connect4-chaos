"""Times the inference forward at self-play batch widths, three ways: the
folded network under bf16 autocast (what the actor runs), the same network
with cuDNN's fused conv+bias+ReLU kernels in explicit bf16, and the folded
network under torch.compile. Prints ms per forward and whether the fused
variants agree with the reference.

Usage: python -m neural.bench_forward <model.pt>    (needs CUDA)
"""

from __future__ import annotations

import copy
import sys
import time

import torch
from torch import nn

from .gpu_env import BoardBatch
from .gpu_selfplay import _prepare_network, all_shapes, forward
from .test_gpu_mcts import play

ONE = (1, 1)


class FusedResidual(nn.Module):
    """conv+bias+ReLU and conv+bias+residual+ReLU as single cuDNN calls."""

    def __init__(self, block):
        super().__init__()
        self.w1, self.b1 = nn.Parameter(block.conv1.weight.data), nn.Parameter(block.conv1.bias.data)
        self.w2, self.b2 = nn.Parameter(block.conv2.weight.data), nn.Parameter(block.conv2.bias.data)

    def forward(self, x):
        out = torch.cudnn_convolution_relu(x, self.w1, self.b1, ONE, ONE, ONE, 1)
        return torch.cudnn_convolution_add_relu(out, self.w2, x, 1.0, self.b2, ONE, ONE, ONE, 1)


class FusedNet(nn.Module):
    def __init__(self, folded):
        super().__init__()
        stem = folded.stem[0]
        self.stem_w, self.stem_b = nn.Parameter(stem.weight.data), nn.Parameter(stem.bias.data)
        self.tower = nn.ModuleList(FusedResidual(block) for block in folded.tower)
        column = folded.column_features[0]
        self.col_w, self.col_b = nn.Parameter(column.weight.data), nn.Parameter(column.bias.data)
        self.drop_logit, self.drop_q = copy.deepcopy(folded.drop_logit), copy.deepcopy(folded.drop_q)
        self.global_features = copy.deepcopy(folded.global_features)
        self.transform_logit, self.transform_q, self.value = (
            copy.deepcopy(folded.transform_logit), copy.deepcopy(folded.transform_q), copy.deepcopy(folded.value))

    def forward(self, planes, legal_mask):
        trunk = torch.cudnn_convolution_relu(planes, self.stem_w, self.stem_b, ONE, ONE, ONE, 1)
        for block in self.tower:
            trunk = block(trunk)
        columns = torch.cudnn_convolution_relu(trunk, self.col_w, self.col_b, ONE, (0, 0), ONE, 1)
        columns = columns.mean(dim=2).transpose(1, 2)
        drop_logits = self.drop_logit(columns).squeeze(-1)
        drop_q = self.drop_q(columns)
        pooled = self.global_features(trunk.mean(dim=(2, 3)))
        policy = torch.cat([drop_logits, self.transform_logit(pooled)], dim=1)
        policy = policy.masked_fill(~legal_mask, float("-inf"))
        q = torch.cat([drop_q, self.transform_q(pooled).view(-1, 3, 3)], dim=1)
        return policy, self.value(pooled), q


def report(label, outputs, exact):
    """Per head: mean and max deviation from fp32 where fp32 is finite, and
    how often the policy argmax agrees."""
    parts = []
    for name, a, b in zip(("policy", "value", "q"), outputs, exact):
        finite = torch.isfinite(b)
        diff = (a.float() - b.float())[finite].abs()
        parts.append(f"{name} mean {diff.mean():.4f} max {diff.max():.3f}")
    agree = float((outputs[0].float().argmax(dim=1) == exact[0].argmax(dim=1)).float().mean())
    print(f"    {label}: {'; '.join(parts)}; policy argmax agreement {agree:.1%}")


def bench(label, fn, planes, legal, repeats=20):
    for _ in range(3):
        fn(planes, legal)
    torch.cuda.synchronize()
    started = time.time()
    for _ in range(repeats):
        out = fn(planes, legal)
    torch.cuda.synchronize()
    ms = (time.time() - started) / repeats * 1000
    print(f"  {label:34s} {ms:7.2f} ms", flush=True)
    return out, ms


def main():
    device = "cuda"
    payload = torch.load(sys.argv[1], map_location=device, weights_only=True)
    folded = _prepare_network(payload, device)
    fused = FusedNet(folded).to(device).to(torch.bfloat16).to(memory_format=torch.channels_last).eval()
    compiled = None
    try:
        compiled = torch.compile(folded, dynamic=True)
    except Exception as exc:  # pragma: no cover - compiler dependent
        print(f"torch.compile unavailable: {type(exc).__name__}: {exc}")
    shapes = all_shapes()
    for width in (8192, 4096, 1024, 256, 64):
        picks = [shapes[i % len(shapes)] for i in range(width)]
        board = BoardBatch([p[0] for p in picks], [p[1] for p in picks],
                           [p[2] for p in picks], [p[3] for p in picks], device)
        board = play(board, [0, 1, 0])
        zeros = torch.zeros(width, dtype=torch.bool, device=device)
        planes, legal = board.planes(zeros, zeros), board.legal()
        print(f"width {width}:")
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False):
                exact = folded(planes.contiguous(memory_format=torch.channels_last), legal)
            reference, base = bench("folded, autocast bf16 (actor)", lambda p, l: forward(folded, p, l),
                                    planes, legal)
            report("autocast bf16 vs fp32", reference, exact)
            nhwc = planes.to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
            try:
                fused_out, fused_ms = bench("fused cuDNN conv+bias+relu, bf16",
                                            lambda p, l: fused(p, l), nhwc, legal)
                report("fused bf16 vs fp32", fused_out, exact)
                print(f"    fused speedup {base / fused_ms:.2f}x")
            except Exception as exc:
                print(f"    fused kernels failed: {type(exc).__name__}: {str(exc)[:200]}")
            if compiled is not None:
                try:
                    started = time.time()
                    forward(compiled, planes, legal)
                    torch.cuda.synchronize()
                    print(f"    (compile/warm-up {time.time() - started:.1f}s)")
                    _out, compiled_ms = bench("folded, torch.compile dynamic", lambda p, l: forward(compiled, p, l),
                                              planes, legal)
                    print(f"    compiled speedup {base / compiled_ms:.2f}x")
                except Exception as exc:
                    print(f"    compiled forward failed: {type(exc).__name__}: {str(exc)[:200]}")
                    compiled = None


if __name__ == "__main__":
    main()
