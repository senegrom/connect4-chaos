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
    def __init__(self, channels: int = 96, blocks: int = 8, head_channels: int = 32):
        super().__init__()
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
