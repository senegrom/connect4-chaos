"""Head-to-head matches between two checkpoints, batched on the GPU.

This is the measurement the held-out tables cannot give: those boards are
solved, and a quarter of every training batch supervises them directly, so
their blunder rates track table coverage rather than play on the boards
that only self-play reaches. The arena plays generation against generation
on large boards, including shapes never used for self-play, and reports a
score per board.

Both sides run the same search budget, so a match compares networks, not
search budgets. Games follow the real rules: threefold repetition is a
draw, and every game is played twice - the second replays the first's
opening move for move with the colours swapped, so each network meets the
same position from either side. Balancing colours alone still left the
match at the mercy of which side drew the kinder openings, noise as large
as the differences these matches exist to resolve; with the pairing, a
network against itself scores exactly 50%.

Usage:
  python -m neural.arena <model_a.pt> <model_b.pt> [games] [sims] [shapes] [seed]
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

import torch

from .gpu_env import BoardBatch, DRAW, NOT_TERMINAL, hash_keys, step
from .gpu_history import DenseHistory, DenseHistoryView, history_counts
from .gpu_mcts import search, visit_policy
from .gpu_selfplay import _prepare_network, forward, parse_shapes

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
    """One checkpoint, or several separated by commas evaluated as an
    ensemble (neural/ensemble.py)."""
    paths = [part for part in str(path).split(",") if part]
    if len(paths) > 1:
        from .ensemble import load_ensemble
        return load_ensemble(paths, device)
    payload = torch.load(paths[0], map_location=device, weights_only=True)
    return _prepare_network(payload, device)


@torch.no_grad()
def _choose(net, board, rep1, rep2, sims, sampling, side, history, keys):
    """One move per game from a search on `board`."""
    legal = board.legal()
    if sims > 0:
        visits, _value = search(net, forward, board, rep1, rep2, sims, add_noise=False,
                                side=side, history=history, keys=keys)
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
    if games % 2:
        raise ValueError("games per board must be even: every opening is played twice")
    torch.manual_seed(seed)
    # Games come in adjacent pairs of the same board, so a pair shares an
    # index but for its last bit.
    picks = [shapes[(i // 2) % len(shapes)] for i in range(games * len(shapes))]
    board = BoardBatch([p[0] for p in picks], [p[1] for p in picks],
                       [p[2] for p in picks], [p[3] for p in picks], device)
    total = len(board)
    keys = hash_keys(device)
    history = DenseHistory(total, MAX_PLIES + 1, device)
    result = torch.full((total,), 9, dtype=torch.int8, device=device)  # 9 = unfinished
    opening = torch.full((OPENING_PLIES, total), -1, dtype=torch.int8, device=device)
    # Each opening is played twice, once from each side: the first game of a
    # pair searches its opening, the second replays those moves and swaps the
    # colours. Balancing colours alone leaves whichever side drew the kinder
    # openings ahead, and that noise is the same size as the differences
    # these matches are asked to resolve.
    a_first = torch.arange(total, device=device) % 2 == 0
    live = torch.arange(total, device=device)
    started = time.time()

    for ply in range(MAX_PLIES):
        if len(live) == 0:
            break
        width = len(live)
        side = ply % 2 == 1                     # every action passes the turn
        hashes = board.position_hash(keys, side)
        search_history = history.search_view(live)
        counts = history_counts(search_history, hashes)
        rep1, rep2 = counts >= 1, counts >= 2
        # A is to move where (A moved first) == (the ply is even).
        a_moves = a_first[live] == (ply % 2 == 0)
        choice = torch.zeros(width, dtype=torch.int64, device=device)
        sampling = ply < OPENING_PLIES
        # Only the first game of a pair searches its opening; the second
        # replays it move for move, so both reach the same position and each
        # network sees it once from either side. The twin cannot have ended
        # earlier - it has played the same moves on the same board - and the
        # replayed half of the opening costs no search at all.
        replaying = ((live % 2) == 1) if sampling else torch.zeros_like(a_moves)
        for net, mask, budget in ((net_a, a_moves & ~replaying, sims),
                                  (net_b, ~a_moves & ~replaying, sims_b)):
            if not bool(mask.any()):
                continue
            index = mask.nonzero().squeeze(1)
            subset_history = DenseHistoryView(search_history.hashes[index],
                                              search_history.lengths[index])
            picked = _choose(net, board.select(index), rep1[index], rep2[index], budget, sampling,
                             side, subset_history, keys)
            choice[index] = picked
        if sampling and bool(replaying.any()):
            # The first game of the pair has already written this ply.
            opening[ply, live[~replaying]] = choice[~replaying].to(torch.int8)
            choice[replaying] = opening[ply, live[replaying] - 1].to(torch.int64)

        history.append_or_reset(live, hashes, choice < 10)
        if ply < OPENING_PLIES:
            opening[ply, live] = choice.to(torch.int8)

        child, outcome = step(board, choice)
        child_hashes = child.position_hash(keys, not side)
        repeated = history.counts(live, child_hashes) >= 2
        terminal = outcome != NOT_TERMINAL
        finished = terminal | repeated
        terminal_score = torch.where(a_moves, outcome, -outcome).to(torch.int8)
        scores = torch.where(terminal, terminal_score, torch.zeros_like(terminal_score))
        result[live[finished]] = scores[finished]
        keep = (~finished).nonzero(as_tuple=False).squeeze(1)
        board = child.select(keep)
        live = live[keep]

    result_cpu = result.cpu().tolist()
    opening_cpu = opening.transpose(0, 1).cpu().tolist()
    unfinished = sum(1 for value in result_cpu if value == 9)
    tally = defaultdict(lambda: [0, 0, 0])                # wins, draws, losses for A
    lines = defaultdict(set)
    for game, shape in enumerate(picks):
        rows, cols, connect, chaos = shape
        key = f"{rows}x{cols}c{connect}{'chaos' if chaos else 'classic'}"
        lines[key].add(tuple(action for action in opening_cpu[game] if action >= 0))
        value = result_cpu[game]
        if value == 9:
            continue
        tally[key][0 if value == 1 else (1 if value == 0 else 2)] += 1
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
