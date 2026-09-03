"""Blunder rate of the whole player - network plus search - on solved boards.

The held-out tables tell us the exact value of every legal action, and the
shards store that. Until now they only ever scored the raw policy, which is
not what plays: a move comes from a search. This reconstructs the positions
from a shard, runs the real search on them, and reports how often the move
it settles on is not exactly optimal.

It also sweeps the exploration constant, since that costs nothing to change
and a better setting is worth more than a doubling of simulations.

Usage:
  python -m neural.search_quality <model.pt> <shard.pt> [sims] [positions]
"""

from __future__ import annotations

import sys

import torch

from . import gpu_mcts
from .arena import load
from .distill import decode_planes
from .gpu_env import CANVAS, BoardBatch
from .gpu_mcts import search, visit_policy
from .gpu_selfplay import forward


def boards_from_planes(planes, device):
    """Rebuilds a batch from the plane stack a shard stores."""
    planes = planes.to(device).float()
    mover, opponent, region = planes[:, 0] > 0.5, planes[:, 1] > 0.5, planes[:, 2] > 0.5
    rows = region[:, :, 0].sum(dim=1).long()
    cols = region[:, 0, :].sum(dim=1).long()
    board = BoardBatch.__new__(BoardBatch)
    board.device = device
    board.rows, board.cols = rows, cols
    board.connect = (planes[:, 3, 0, 0] * 10).round().long()
    board.chaos = planes[:, 4, 0, 0] > 0.5
    board.mover, board.opponent = mover, opponent
    occupied = (mover | opponent)
    board.heights = occupied.sum(dim=1).long()             # stones per column
    board.pieces = occupied.sum(dim=(1, 2)).long()
    assert int(board.rows.max()) <= CANVAS and int(board.cols.max()) <= CANVAS
    return board


@torch.no_grad()
def blunder_rate(net, shard, sims, limit, device, c_puct=None):
    """Share of positions where the chosen move is not exactly optimal.

    sims=0 asks the policy head alone, which is the number the training
    reports; anything higher is what actually plays."""
    previous = gpu_mcts.C_PUCT
    if c_puct is not None:
        gpu_mcts.C_PUCT = c_puct
    try:
        optimal = shard["policy"][:limit] > 0
        wrong = 0
        counted = 0
        for start in range(0, min(limit, len(shard["wdl"])), 512):
            stop = start + 512
            planes = decode_planes(shard["planes"][start:stop])
            board = boards_from_planes(planes, device)
            zeros = torch.zeros(len(board), dtype=torch.bool, device=device)
            legal = board.legal()
            if sims > 0:
                visits, _value = search(net, forward, board, zeros, zeros, sims, add_noise=False)
                choice = visit_policy(visits, legal).argmax(dim=1).cpu()
            else:
                logits, _wdl, _q = forward(net, board.planes(zeros, zeros), legal)
                choice = logits.masked_fill(~legal, float("-inf")).argmax(dim=1).cpu()
            chunk = optimal[start:stop]
            wrong += int((~chunk.gather(1, choice.unsqueeze(1)).squeeze(1)).sum())
            counted += len(choice)
        return wrong / max(1, counted), counted
    finally:
        gpu_mcts.C_PUCT = previous


def main():
    model_path, shard_path = sys.argv[1], sys.argv[2]
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load(model_path, device)
    shard = torch.load(shard_path, map_location="cpu", weights_only=True)
    rows, cols, connect = shard["config"]
    print(f"{shard_path.split(chr(92))[-1]}: {rows}x{cols} c{connect}, {limit} positions")
    for budget in (0, 32, sims, 512):
        rate, counted = blunder_rate(net, shard, budget, limit, device)
        label = "policy head alone" if budget == 0 else f"{budget:4d} simulations"
        print(f"  {label:>18}: blunder rate {rate:.4f} over {counted}", flush=True)


if __name__ == "__main__":
    main()
