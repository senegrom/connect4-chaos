"""Head-to-head matches between two checkpoints, batched on the GPU.

This is the measurement the held-out tables cannot give: those boards are
solved, and a quarter of every training batch supervises them directly, so
their blunder rates track table coverage rather than play on the boards
that only self-play reaches. The arena plays generation against generation
on large boards, including shapes never used for self-play, and reports a
score per board.

Both sides run the same search budget, so a match compares networks, not
search budgets. Games follow the real rules: threefold repetition is a
draw, and colours alternate so first-player advantage cancels.

Usage:
  python -m neural.arena <model_a.pt> <model_b.pt> [games] [sims] [shapes] [seed]
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

import torch

from .gpu_env import BoardBatch, DRAW, NOT_TERMINAL, step
from .gpu_mcts import search, visit_policy
from .gpu_selfplay import forward, parse_shapes
from .model import PolicyValueNet

MAX_PLIES = 300           # far beyond any real game; repetition ends them
OPENING_PLIES = 4         # sampled, so the games of a colour are not identical

# Boards wide enough that no exact table exists, spanning both rule sets,
# both connect lengths and lopsided shapes. The tag "*" marks shapes the
# actors never play, which is where generalisation shows.
DEFAULT_SHAPES = ("6x7c4chaos,6x7c4classic,7x7c4chaos,8x8c5chaos,5x10c4chaos,"
                  "10x10c4classic,9x7c4classic,7x9c5chaos,"
                  "4x9c4chaos,9x9c5chaos,10x7c4chaos,8x6c4classic,6x10c5classic,10x9c4chaos")


def load(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    net = PolicyValueNet(*payload.get("arch", (192, 12, 48))).to(device)
    net.load_state_dict(payload["model"])
    net.eval()
    return net


@torch.no_grad()
def _choose(net, board, rep1, rep2, sims, sampling):
    """One move per game from a search on `board`."""
    legal = board.legal()
    if sims > 0:
        visits, _value = search(net, forward, board, rep1, rep2, sims, add_noise=False)
        policy = visit_policy(visits, legal)
    else:
        logits, _wdl, _q = forward(net, board.planes(rep1, rep2), legal)
        policy = torch.softmax(logits.masked_fill(~legal, float("-inf")), dim=1)
    if sampling:
        return torch.multinomial(policy.clamp(min=1e-12), 1).squeeze(1)
    return policy.argmax(dim=1)


@torch.no_grad()
def play(net_a, net_b, shapes, games: int, sims: int, seed: int, device):
    """Plays `games` games per shape and returns per-shape results for A."""
    torch.manual_seed(seed)
    picks = [shapes[i % len(shapes)] for i in range(games * len(shapes))]
    board = BoardBatch([p[0] for p in picks], [p[1] for p in picks],
                       [p[2] for p in picks], [p[3] for p in picks], device)
    total = len(board)
    keys = torch.rand((2, 10, 10), dtype=torch.float64, device=device)
    histories = [dict() for _ in range(total)]
    result = [None] * total
    # A moves first in every other game.
    a_first = torch.arange(total, device=device) % 2 == 0
    live = torch.arange(total, device=device)
    started = time.time()

    for ply in range(MAX_PLIES):
        if len(live) == 0:
            break
        alive = live.tolist()
        width = len(alive)
        hashes = board.position_hash(keys).cpu().tolist()
        counts = torch.tensor([histories[alive[i]].get(hashes[i], 0) for i in range(width)],
                              device=device)
        rep1, rep2 = counts >= 1, counts >= 2
        # A is to move where (A moved first) == (the ply is even).
        a_moves = a_first[live] == (ply % 2 == 0)
        choice = torch.zeros(width, dtype=torch.int64, device=device)
        sampling = ply < OPENING_PLIES
        for net, mask in ((net_a, a_moves), (net_b, ~a_moves)):
            if not bool(mask.any()):
                continue
            index = mask.nonzero().squeeze(1)
            picked = _choose(net, board.select(index), rep1[index], rep2[index], sims, sampling)
            choice[index] = picked

        for i in range(width):
            game = alive[i]
            histories[game][hashes[i]] = histories[game].get(hashes[i], 0) + 1

        child, outcome = step(board, choice)
        outcome_cpu = outcome.cpu().tolist()
        child_hashes = child.position_hash(keys).cpu().tolist()
        a_moved = a_moves.cpu().tolist()
        keep = []
        for i in range(width):
            game = alive[i]
            if outcome_cpu[i] == 1:                       # the mover just won
                result[game] = 1 if a_moved[i] else -1
            elif outcome_cpu[i] == DRAW:                  # board full
                result[game] = 0
            elif histories[game].get(child_hashes[i], 0) >= 2:
                result[game] = 0                          # threefold repetition
            else:
                keep.append(i)
        if len(keep) < width:
            index = torch.tensor(keep, dtype=torch.int64, device=device)
            board = child.select(index)
            live = live[index]
        else:
            board = child

    unfinished = sum(1 for value in result if value is None)
    tally = defaultdict(lambda: [0, 0, 0])                # wins, draws, losses for A
    for game, shape in enumerate(picks):
        if result[game] is None:
            continue
        rows, cols, connect, chaos = shape
        key = f"{rows}x{cols}c{connect}{'chaos' if chaos else 'classic'}"
        tally[key][0 if result[game] == 1 else (1 if result[game] == 0 else 2)] += 1
    return dict(tally), unfinished, time.time() - started


def report(tally, unfinished, seconds, label_a="A", label_b="B"):
    lines = []
    totals = [0, 0, 0]
    for key in sorted(tally):
        wins, draws, losses = tally[key]
        played = wins + draws + losses
        score = (wins + 0.5 * draws) / max(1, played)
        totals = [totals[i] + tally[key][i] for i in range(3)]
        lines.append(f"  {key:18s} {score:6.1%}  ({wins}W/{draws}D/{losses}L)")
    played = sum(totals)
    overall = (totals[0] + 0.5 * totals[1]) / max(1, played)
    header = (f"arena {label_a} vs {label_b}: {overall:.1%} over {played} games, "
              f"{seconds:.0f}s" + (f", {unfinished} unfinished" if unfinished else ""))
    return overall, "\n".join([header] + lines)


def main():
    model_a, model_b = sys.argv[1], sys.argv[2]
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    sims = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    spec = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_SHAPES
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 7
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net_a, net_b = load(model_a, device), load(model_b, device)
    tally, unfinished, seconds = play(net_a, net_b, parse_shapes(spec), games, sims, seed, device)
    _overall, text = report(tally, unfinished, seconds,
                            model_a.split("\\")[-1], model_b.split("\\")[-1])
    print(text, flush=True)


if __name__ == "__main__":
    main()
