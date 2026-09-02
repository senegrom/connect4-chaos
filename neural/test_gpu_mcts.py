"""Checks batched PUCT search: tactics, bookkeeping, and playing strength.

  1. bookkeeping - every simulation lands on the root, visits stay legal
  2. tactics     - an immediate win is taken, an immediate threat is blocked
  3. strength    - search beats the raw policy of the same network

The tactical positions are reached by playing real moves through the
environment, so no hand-built board can be inconsistent.

Usage: python -m neural.test_gpu_mcts <model.pt> [device] [sims]
"""

from __future__ import annotations

import sys

import torch

from .gpu_env import BoardBatch, DRAW, NOT_TERMINAL, step
from .gpu_mcts import sample_actions, search, visit_policy
from .gpu_selfplay import forward
from .model import PolicyValueNet

MAX_PLIES = 220


def load(model_path, device):
    payload = torch.load(model_path, map_location=device, weights_only=True)
    net = PolicyValueNet(*payload.get("arch", (192, 12, 48))).to(device)
    net.load_state_dict(payload["model"])
    net.eval()
    return net


def play(board, actions):
    """Applies a sequence of identical actions across the batch."""
    for action in actions:
        board, outcome = step(board, torch.full((len(board),), action,
                                                dtype=torch.int64, device=board.device))
        assert bool((outcome == NOT_TERMINAL).all()), f"setup move {action} ended the game"
    return board


def flags(board):
    zeros = torch.zeros(len(board), dtype=torch.bool, device=board.device)
    return zeros, zeros


def test_bookkeeping(net, device, sims):
    board = BoardBatch([6] * 8, [7] * 8, [4] * 8, [False] * 8, device)
    rep1, rep2 = flags(board)
    visits, _value = search(net, forward, board, rep1, rep2, sims)
    legal = board.legal()
    assert bool((visits.sum(dim=1) == sims).all()), f"visit total {visits.sum(dim=1)} != {sims}"
    assert not bool(visits.masked_fill(legal, 0).any()), "visits landed on illegal actions"
    policy = visit_policy(visits, legal)
    assert torch.allclose(policy.sum(dim=1), torch.ones(len(board), device=device), atol=1e-5)
    print(f"bookkeeping: {sims} simulations per tree, all on legal actions")


def test_tactics(net, device, sims):
    # Four in a row wins; the mover owns columns 0,1,2 on the bottom row.
    board = BoardBatch([6], [7], [4], [False], device)
    board = play(board, [0, 6, 1, 5, 2, 4])
    rep1, rep2 = flags(board)
    visits, _value = search(net, forward, board, rep1, rep2, sims)
    choice = int(visits.argmax(dim=1).item())
    win_share = float(visits[0, 3] / visits[0].sum())
    assert choice == 3, f"winning drop is column 3, search chose {choice}"

    # The opponent has three in column 6; the mover must block there.
    board = BoardBatch([6], [7], [4], [False], device)
    board = play(board, [6, 0, 6, 1, 6])
    rep1, rep2 = flags(board)
    visits, _value = search(net, forward, board, rep1, rep2, sims)
    choice = int(visits.argmax(dim=1).item())
    block_share = float(visits[0, 6] / visits[0].sum())
    assert choice == 6, f"must block column 6, search chose {choice}"
    print(f"tactics: takes the win ({win_share:.0%} of visits), "
          f"blocks the threat ({block_share:.0%} of visits)")


OPENING_PLIES = 6         # sampled from the policy so the games differ


@torch.no_grad()
def match(net, device, sims, games, shape, seed):
    """Search plays the raw policy; search moves first in half the games.

    Both players are deterministic, so without variety every game of a
    colour would be the same game. The first OPENING_PLIES moves are
    therefore sampled from the network's own policy, which makes each game
    an independent position drawn from a plausible opening distribution."""
    rows, cols, connect, chaos = shape
    board = BoardBatch([rows] * games, [cols] * games, [connect] * games, [chaos] * games, device)
    torch.manual_seed(seed)
    searcher_first = torch.arange(games, device=device) % 2 == 0
    searcher_moves = searcher_first.clone()      # true when the side to move searches
    active = torch.ones(games, dtype=torch.bool, device=device)
    result = torch.full((games,), 9, dtype=torch.int64, device=device)   # 9 = unfinished
    zeros = torch.zeros(games, dtype=torch.bool, device=device)

    for _ply in range(MAX_PLIES):
        if not bool(active.any()):
            break
        legal = board.legal()
        logits, _wdl, _q = forward(net, board.planes(zeros, zeros), legal)
        masked = logits.masked_fill(~legal, float("-inf"))
        if _ply < OPENING_PLIES:
            choice = torch.multinomial(torch.softmax(masked, dim=1), 1).squeeze(1)
        else:
            visits, _value = search(net, forward, board, zeros, zeros, sims, add_noise=False)
            choice = torch.where(searcher_moves, visits.argmax(dim=1), masked.argmax(dim=1))

        child, outcome = step(board, choice)
        finished = active & (outcome != NOT_TERMINAL)
        # Outcome is for the player who just moved.
        result = torch.where(finished, torch.where(searcher_moves, outcome, -outcome), result)
        result = torch.where(finished & (outcome == DRAW), torch.zeros_like(result), result)
        active = active & ~finished
        board = child
        searcher_moves = ~searcher_moves

    decided = result != 9
    wins = int(((result == 1) & decided).sum())
    losses = int(((result == -1) & decided).sum())
    draws = int(((result == 0) & decided).sum())
    unfinished = games - int(decided.sum())
    score = (wins + 0.5 * draws) / max(1, wins + losses + draws)
    print(f"  {rows}x{cols} c{connect} {'chaos' if chaos else 'classic'}: search scores "
          f"{score:.1%} ({wins}W/{draws}D/{losses}L" + (f", {unfinished} unfinished)" if unfinished else ")"))
    return wins, draws, losses


def test_strength(net, device, sims, games):
    print(f"strength: {sims}-simulation search vs the same network's raw policy")
    shapes = [(6, 7, 4, False), (6, 7, 4, True), (8, 8, 4, True)]
    totals = [0, 0, 0]
    for index, shape in enumerate(shapes):
        wins, draws, losses = match(net, device, sims, games, shape, seed=1234 + index)
        totals = [totals[0] + wins, totals[1] + draws, totals[2] + losses]
    played = sum(totals)
    score = (totals[0] + 0.5 * totals[1]) / max(1, played)
    print(f"  overall: {score:.1%} for search ({totals[0]}W/{totals[1]}D/{totals[2]}L)")
    assert score > 0.5, "search must be at least as strong as the raw policy"


def main():
    model_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else ("cuda" if torch.cuda.is_available() else "cpu")
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    games = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    net = load(model_path, device)
    test_bookkeeping(net, device, sims)
    test_tactics(net, device, sims)
    test_strength(net, device, sims, games)
    print("GPU MCTS OK")


if __name__ == "__main__":
    main()
