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
# Both sides play deterministically after the opening, so games only differ
# through it: too few opening plies and a match is a handful of distinct
# lines repeated, which reads as a landslide either way. The report counts
# the distinct openings so that clustering is visible rather than implied.
OPENING_PLIES = 8
OPENING_TEMPERATURE = 1.3

# Every playable board, so a model cannot look stronger by trading one
# shape against another; fewer games per board keeps the total sane.
DEFAULT_SHAPES = "all"


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
        spread = policy.clamp(min=1e-12) ** (1.0 / OPENING_TEMPERATURE)
        return torch.multinomial(spread, 1).squeeze(1)
    return policy.argmax(dim=1)


@torch.no_grad()
def play(net_a, net_b, shapes, games: int, sims: int, seed: int, device, sims_b=None):
    """Plays `games` games per shape and returns per-shape results for A.

    `sims_b` gives B a different search budget, which is how the value of
    search itself is measured: the same network on both sides, thinking
    for different lengths."""
    sims_b = sims if sims_b is None else sims_b
    torch.manual_seed(seed)
    picks = [shapes[i % len(shapes)] for i in range(games * len(shapes))]
    board = BoardBatch([p[0] for p in picks], [p[1] for p in picks],
                       [p[2] for p in picks], [p[3] for p in picks], device)
    total = len(board)
    keys = torch.rand((2, 10, 10), dtype=torch.float64, device=device)
    histories = [dict() for _ in range(total)]
    result = [None] * total
    opening = [[] for _ in range(total)]        # to count distinct opening lines
    # Colour must not track the board: shapes cycle with the index, so
    # keying colour on the same parity gave every board a single colour
    # (with an even shape count) and turned first-player advantage into an
    # apparent skill gap. Colour flips per lap through the shape list.
    laps = torch.arange(total, device=device) // len(shapes)
    a_first = laps % 2 == 0
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
        for net, mask, budget in ((net_a, a_moves, sims), (net_b, ~a_moves, sims_b)):
            if not bool(mask.any()):
                continue
            index = mask.nonzero().squeeze(1)
            picked = _choose(net, board.select(index), rep1[index], rep2[index], budget, sampling)
            choice[index] = picked

        choice_cpu = choice.cpu().tolist()
        for i in range(width):
            game = alive[i]
            histories[game][hashes[i]] = histories[game].get(hashes[i], 0) + 1
            if ply < OPENING_PLIES:
                opening[game].append(choice_cpu[i])

        child, outcome = step(board, choice)
        outcome_cpu = outcome.cpu().tolist()
        child_hashes = child.position_hash(keys).cpu().tolist()
        a_moved = a_moves.cpu().tolist()
        keep = []
        for i in range(width):
            game = alive[i]
            if outcome_cpu[i] != NOT_TERMINAL:
                # Outcome is for the player who just moved: WIN 1, DRAW 0,
                # LOSS -1. A chaos transform can complete a line for the
                # opponent, so LOSS is a real ending and must be scored;
                # enumerating only WIN and DRAW let those games run on with
                # the colours reversed.
                mover = outcome_cpu[i]
                result[game] = mover if a_moved[i] else -mover
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
    lines = defaultdict(set)
    for game, shape in enumerate(picks):
        rows, cols, connect, chaos = shape
        key = f"{rows}x{cols}c{connect}{'chaos' if chaos else 'classic'}"
        lines[key].add(tuple(opening[game]))
        if result[game] is None:
            continue
        tally[key][0 if result[game] == 1 else (1 if result[game] == 0 else 2)] += 1
    distinct = {key: len(value) for key, value in lines.items()}
    return dict(tally), unfinished, time.time() - started, distinct


def report(tally, unfinished, seconds, label_a="A", label_b="B", distinct=None):
    """A summary, not a line per board: with every shape in play a full
    listing runs to hundreds of lines and the headline scrolls away."""
    distinct = distinct or {}
    totals = [0, 0, 0]
    scored = []
    groups = {"chaos": [0, 0, 0], "classic": [0, 0, 0],
              "small (<=30 cells)": [0, 0, 0], "large (>30 cells)": [0, 0, 0]}
    for key, (wins, draws, losses) in tally.items():
        played = wins + draws + losses
        if not played:
            continue
        totals = [totals[i] + (wins, draws, losses)[i] for i in range(3)]
        scored.append(((wins + 0.5 * draws) / played, key, wins, draws, losses))
        rows, rest = key.split("x", 1)
        cols = rest.split("c", 1)[0]
        cells = int(rows) * int(cols)
        for name in ("chaos" if "chaos" in key else "classic",
                     "small (<=30 cells)" if cells <= 30 else "large (>30 cells)"):
            groups[name] = [groups[name][i] + (wins, draws, losses)[i] for i in range(3)]

    played = sum(totals)
    overall = (totals[0] + 0.5 * totals[1]) / max(1, played)
    lines = [f"arena {label_a} vs {label_b}: {overall:.1%} over {played} games on "
             f"{len(scored)} boards, {seconds:.0f}s"
             + (f", {unfinished} unfinished" if unfinished else "")]
    for name, (wins, draws, losses) in groups.items():
        group_played = wins + draws + losses
        if group_played:
            score = (wins + 0.5 * draws) / group_played
            lines.append(f"  {name:20s} {score:6.1%}  ({wins}W/{draws}D/{losses}L)")
    scored.sort()
    for label, rows in (("worst boards", scored[:5]), ("best boards", scored[-5:])):
        lines.append(f"  {label}:")
        for score, key, wins, draws, losses in rows:
            spread = f", {distinct[key]} openings" if key in distinct else ""
            lines.append(f"    {key:16s} {score:6.1%}  ({wins}W/{draws}D/{losses}L{spread})")
    return overall, chr(10).join(lines)


def main():
    model_a, model_b = sys.argv[1], sys.argv[2]
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    sims = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    spec = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_SHAPES
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 7
    # A different budget for B measures what search itself is worth: the
    # same network on both sides, thinking for different lengths.
    sims_b = int(sys.argv[7]) if len(sys.argv) > 7 else sims
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net_a, net_b = load(model_a, device), load(model_b, device)
    tally, unfinished, seconds, distinct = play(net_a, net_b, parse_shapes(spec),
                                                games, sims, seed, device, sims_b)
    _overall, text = report(tally, unfinished, seconds,
                            model_a.split("\\")[-1], model_b.split("\\")[-1], distinct)
    print(text, flush=True)


if __name__ == "__main__":
    main()
