"""Policy/value/Q network for variable-board Connect-k (canvas 10x10).

Input: 7 planes (mover, opponent, on-board mask, connect/10, chaos flag,
repeated-once, repeated-twice). Every head is size-agnostic: drop logits
come from per-column features (the tower's columns pooled over rows),
transform logits and the value from the pooled trunk, so the same weights
serve a 4x4 and a 10x10 board without a canvas-position bias.

Heads, all from the mover's perspective:
  policy  (N,13)    masked action logits: 10 drops, flip, rotate cw/ccw
  value   (N,3)     loss / draw / win of the position
  q       (N,13,3)  loss / draw / win after each action - the exact
                    per-action supervision the solver tables provide
"""

from __future__ import annotations

import copy

import torch
from torch import nn

PLANES = 7
CANVAS = 10
ACTIONS = 13
DROPS = 10


class Residual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = torch.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return torch.relu(out + x)


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 256, blocks: int = 20, head_channels: int = 64):
        super().__init__()
        self.channels, self.blocks, self.head_channels = channels, blocks, head_channels
        self.stem = nn.Sequential(
            nn.Conv2d(PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[Residual(channels) for _ in range(blocks)])
        # Per-column features: 1x1 conv, then pool over rows -> (N, H, DROPS).
        self.column_features = nn.Sequential(
            nn.Conv2d(channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(),
        )
        self.drop_logit = nn.Linear(head_channels, 1)
        self.drop_q = nn.Linear(head_channels, 3)
        # Global features: pooled trunk -> transforms, value, transform q.
        self.global_features = nn.Sequential(
            nn.Linear(channels, 128),
            nn.ReLU(),
        )
        self.transform_logit = nn.Linear(128, 3)
        self.transform_q = nn.Linear(128, 3 * 3)
        self.value = nn.Linear(128, 3)

    def forward(self, planes, legal_mask):
        trunk = self.tower(self.stem(planes))                 # (N, C, rows, cols)
        columns = self.column_features(trunk).mean(dim=2)      # (N, H, cols)
        columns = columns.transpose(1, 2)                       # (N, cols, H)
        drop_logits = self.drop_logit(columns).squeeze(-1)      # (N, DROPS)
        drop_q = self.drop_q(columns)                           # (N, DROPS, 3)

        pooled = self.global_features(trunk.mean(dim=(2, 3)))   # (N, 128)
        transform_logits = self.transform_logit(pooled)         # (N, 3)
        transform_q = self.transform_q(pooled).view(-1, 3, 3)   # (N, 3, 3)

        policy = torch.cat([drop_logits, transform_logits], dim=1)
        policy = policy.masked_fill(~legal_mask, float('-inf'))
        q = torch.cat([drop_q, transform_q], dim=1)             # (N, 13, 3)
        return policy, self.value(pooled), q


def _fold(conv: nn.Conv2d, norm: nn.BatchNorm2d) -> nn.Conv2d:
    scale = norm.weight / torch.sqrt(norm.running_var + norm.eps)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                      conv.stride, conv.padding, bias=True).to(conv.weight.device)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale[:, None, None, None])
        fused.bias.copy_(norm.bias - norm.running_mean * scale)
    return fused


def fold_batchnorm(net: PolicyValueNet) -> PolicyValueNet:
    """A copy of `net` for inference with every BatchNorm folded into the
    convolution before it. Exact in eval mode (the normalisation is an
    affine map of the conv output), and it removes 25 kernels from every
    forward pass, which the self-play actor runs tens of millions of times
    per generation."""
    folded = copy.deepcopy(net).eval()
    folded.stem = nn.Sequential(_fold(folded.stem[0], folded.stem[1]), nn.ReLU())
    for block in folded.tower:
        block.conv1 = _fold(block.conv1, block.norm1)
        block.conv2 = _fold(block.conv2, block.norm2)
        block.norm1 = nn.Identity()
        block.norm2 = nn.Identity()
    folded.column_features = nn.Sequential(
        _fold(folded.column_features[0], folded.column_features[1]), nn.ReLU())
    return folded


_ONE = (1, 1)


class FusedResidual(nn.Module):
    """A folded residual block as two cuDNN calls: conv+bias+ReLU, then
    conv+bias+residual+ReLU. CUDA only, inference only."""

    def __init__(self, block: Residual):
        super().__init__()
        self.weight1 = nn.Parameter(block.conv1.weight.detach().clone(), requires_grad=False)
        self.bias1 = nn.Parameter(block.conv1.bias.detach().clone(), requires_grad=False)
        self.weight2 = nn.Parameter(block.conv2.weight.detach().clone(), requires_grad=False)
        self.bias2 = nn.Parameter(block.conv2.bias.detach().clone(), requires_grad=False)

    def forward(self, x):
        out = torch.cudnn_convolution_relu(x, self.weight1, self.bias1, _ONE, _ONE, _ONE, 1)
        return torch.cudnn_convolution_add_relu(out, self.weight2, x, 1.0, self.bias2,
                                                _ONE, _ONE, _ONE, 1)


class FusedInferenceNet(nn.Module):
    """PolicyValueNet for CUDA inference: BatchNorm folded and every
    convolution fused with its bias and ReLU into a single cuDNN kernel, in
    one dtype throughout (bf16 by default). Same outputs as the network under
    bf16 autocast to within bf16 noise, at 1.7x the speed; the self-play
    actor evaluates tens of millions of positions per generation."""

    own_dtype = True        # never wrap this in autocast; it casts its own input

    def __init__(self, net: PolicyValueNet, dtype=torch.bfloat16):
        super().__init__()
        folded = fold_batchnorm(net)
        stem = folded.stem[0]
        self.dtype = dtype
        self.stem_weight = nn.Parameter(stem.weight.detach().clone(), requires_grad=False)
        self.stem_bias = nn.Parameter(stem.bias.detach().clone(), requires_grad=False)
        self.tower = nn.ModuleList(FusedResidual(block) for block in folded.tower)
        column = folded.column_features[0]
        self.column_weight = nn.Parameter(column.weight.detach().clone(), requires_grad=False)
        self.column_bias = nn.Parameter(column.bias.detach().clone(), requires_grad=False)
        self.drop_logit, self.drop_q = folded.drop_logit, folded.drop_q
        self.global_features = folded.global_features
        self.transform_logit, self.transform_q, self.value = (
            folded.transform_logit, folded.transform_q, folded.value)
        self.to(dtype).to(memory_format=torch.channels_last).eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, planes, legal_mask):
        planes = planes.to(self.dtype).contiguous(memory_format=torch.channels_last)
        trunk = torch.cudnn_convolution_relu(planes, self.stem_weight, self.stem_bias, _ONE, _ONE, _ONE, 1)
        for block in self.tower:
            trunk = block(trunk)
        columns = torch.cudnn_convolution_relu(trunk, self.column_weight, self.column_bias,
                                               _ONE, (0, 0), _ONE, 1)
        columns = columns.mean(dim=2).transpose(1, 2)           # (N, cols, H)
        drop_logits = self.drop_logit(columns).squeeze(-1)      # (N, DROPS)
        drop_q = self.drop_q(columns)                           # (N, DROPS, 3)
        pooled = self.global_features(trunk.mean(dim=(2, 3)))   # (N, 128)
        policy = torch.cat([drop_logits, self.transform_logit(pooled)], dim=1)
        policy = policy.masked_fill(~legal_mask, float('-inf'))
        q = torch.cat([drop_q, self.transform_q(pooled).view(-1, 3, 3)], dim=1)
        return policy, self.value(pooled), q
