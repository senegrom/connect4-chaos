"""Policy/value network for variable-board Connect-k (canvas 10x10).

Input: 7 planes (mover, opponent, on-board mask, connect/10, chaos flag,
repeated-once, repeated-twice). Heads: 13 masked action logits (10 drops,
flip, rotate cw/ccw) and a 3-way win/draw/loss distribution from the
mover's perspective — the labels are exact, so the value head is a
classifier, not a regressor.
"""

from __future__ import annotations

import torch
from torch import nn

PLANES = 7
CANVAS = 10
ACTIONS = 13


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
    def __init__(self, channels: int = 96, blocks: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[Residual(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * CANVAS * CANVAS, ACTIONS),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(CANVAS * CANVAS, 128),
            nn.ReLU(),
            nn.Linear(128, 3),   # loss, draw, win for the mover
        )

    def forward(self, planes, legal_mask):
        trunk = self.tower(self.stem(planes))
        policy = self.policy_head(trunk)
        policy = policy.masked_fill(~legal_mask, float('-inf'))
        return policy, self.value_head(trunk)
