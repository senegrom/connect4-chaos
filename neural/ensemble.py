"""Average the predictions of several checkpoints, as one network object.

This is the other kind of ensemble: not one averaged set of weights (see
neural/soup.py) but every member evaluated and their answers pooled. It is
strictly more expensive - N forward passes per position - so the honest
comparison is against a single network given N times the search, which the
arena can do with a different simulation budget per side.

Probabilities are averaged, not logits: the heads are categorical, and the
mean of two distributions is a distribution while the mean of two logit
vectors depends on arbitrary per-network offsets. The result is returned as
log-probabilities so callers can keep applying softmax and legality masks
exactly as they do to a single network's output.
"""

from __future__ import annotations

import sys

import torch
from torch import nn

MIN_PROBABILITY = 1e-9      # keeps log() finite for actions no member likes


class Ensemble(nn.Module):
    """Members are evaluated in turn; their probabilities are averaged."""

    def __init__(self, members):
        super().__init__()
        if len(members) < 2:
            raise ValueError("an ensemble needs at least two members")
        self.members = nn.ModuleList(members)
        # Members prepared for CUDA carry their own dtype and layout, so the
        # caller must not wrap this in autocast (see gpu_selfplay.forward).
        self.own_dtype = all(getattr(member, "own_dtype", False) for member in members)

    def forward(self, planes, legal):
        policy = value = q = None
        for member in self.members:
            member_policy, member_value, member_q = member(planes, legal)
            # A masked action is -inf for every member, so its mean stays 0.
            policy_probability = torch.softmax(member_policy.float(), dim=1)
            value_probability = torch.softmax(member_value.float(), dim=1)
            q_probability = torch.softmax(member_q.float(), dim=2)
            policy = policy_probability if policy is None else policy + policy_probability
            value = value_probability if value is None else value + value_probability
            q = q_probability if q is None else q + q_probability
        count = len(self.members)
        policy = (policy / count).clamp(min=MIN_PROBABILITY).log()
        return policy.masked_fill(~legal, float("-inf")), (value / count).log(), (q / count).log()


def load_ensemble(paths, device):
    """Builds an ensemble from checkpoint paths, each prepared for inference
    exactly as a single network would be."""
    from .gpu_selfplay import _prepare_network        # imported late: it imports the search

    members = []
    for path in paths:
        payload = torch.load(path, map_location=device, weights_only=True)
        members.append(_prepare_network(payload, device))
    ensemble = Ensemble(members)
    ensemble.eval()
    return ensemble


def main():
    """Reports how much the members disagree, as a sanity check that the
    ensemble is really pooling different opinions."""
    from .gpu_env import BoardBatch
    from .gpu_selfplay import all_shapes, forward

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sys.argv[1].split(",")
    ensemble = load_ensemble(paths, device)
    shapes = all_shapes()
    picks = [shapes[i % len(shapes)] for i in range(512)]
    board = BoardBatch([p[0] for p in picks], [p[1] for p in picks],
                       [p[2] for p in picks], [p[3] for p in picks], device)
    zeros = torch.zeros(len(board), dtype=torch.bool, device=device)
    planes, legal = board.planes(zeros, zeros), board.legal()
    with torch.no_grad():
        choices = []
        for member in ensemble.members:
            logits, _value, _q = forward(member, planes, legal)
            choices.append(logits.argmax(dim=1))
        pooled, _value, _q = forward(ensemble, planes, legal)
        pooled_choice = pooled.argmax(dim=1)
    agree = torch.stack([choice == choices[0] for choice in choices]).all(dim=0)
    with_first = (pooled_choice == choices[0]).float().mean()
    print(f"{len(paths)} members over {len(board)} openings: all agree on "
          f"{float(agree.float().mean()):.1%}; the pooled move matches member 1 on "
          f"{float(with_first):.1%}", flush=True)


if __name__ == "__main__":
    main()
