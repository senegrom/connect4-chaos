"""Blunder rate of the whole player - network plus search - on solved boards.

The held-out tables tell us the exact value of every legal action, and the
shards store that. The training log only ever scores the raw policy, which
is not what plays: a move comes from a search. This reconstructs the
positions from a shard, runs the real search on them, and reports how
often the move it settles on is not exactly optimal.

Given a directory instead of a shard, it scores the held-out shard of
every solved board (the reserved positions from its first shard) and
pools the rates by rule set. Legacy shards use the same position filter
as training; historical checkpoints may still have seen the old split. That is the
number to compare checkpoints by.

It also sweeps the exploration constant, since that costs nothing to change
and a better setting is worth more than a doubling of simulations.

Usage:
  python -m neural.search_quality <model.pt> <shard.pt | shard_dir> [sims] [positions]
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import torch

from . import gpu_mcts
from .arena import load
from .distill import decode_planes, filtered_chunks
from .data_split import SAMPLE_FIELDS
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
    # Cache Python-side batch invariants once. MCTS/environment hot paths use
    # these instead of synchronising CUDA merely to discover a bound or
    # whether any transform can be legal.
    board.max_connect = int(board.connect.max().item())
    board.any_chaos = bool(board.chaos.any().item())
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
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Position limit must be a positive integer")
        effective_count = min(limit, len(shard["wdl"]))
        if any(len(shard[key]) != len(shard["wdl"]) for key in ("planes", "policy", "legal", "q")):
            raise ValueError("Shard tensors have inconsistent sample counts")
        optimal = shard["policy"][:effective_count] > 0
        wrong = 0
        counted = 0
        for start in range(0, effective_count, 512):
            stop = min(start + 512, effective_count)
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


@torch.no_grad()
def head_stats(net, shard, limit, device):
    """How sure the raw heads are on the same positions, as sums over the
    positions counted: policy entropy over legal moves (nats), probability
    of the top move, probability mass on the exactly-optimal moves, W/D/L
    head hits against the exact result, and per-action Q head hits on the
    legal actions whose exact result is known (the search seeds unvisited
    children from that head). Blunder rates say whether the
    policy is right; these say how confident it is, which is what the
    search feeds on - a policy that gets sharper while search results get
    worse is starving the tree of alternatives."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Position limit must be a positive integer")
    count = min(limit, len(shard["wdl"]))
    optimal = shard["policy"][:count] > 0
    sums = dict.fromkeys(HEAD_KEYS, 0.0)
    sums["count"] = count
    for start in range(0, count, 512):
        stop = min(start + 512, count)
        board = boards_from_planes(decode_planes(shard["planes"][start:stop]), device)
        zeros = torch.zeros(len(board), dtype=torch.bool, device=device)
        legal = board.legal()
        logits, wdl, q = forward(net, board.planes(zeros, zeros), legal)
        log_probs = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=1)
        probs = log_probs.exp()                    # exactly zero on illegal moves
        entropy = -(probs * log_probs.masked_fill(~legal, 0.0)).sum(dim=1)
        sums["entropy"] += float(entropy.sum())
        sums["top"] += float(probs.max(dim=1).values.sum())
        sums["optimal_mass"] += float((probs.cpu() * optimal[start:stop]).sum())
        sums["value_hits"] += float((wdl.argmax(dim=1).cpu() == shard["wdl"][start:stop].long()).sum())
        q_targets = shard["q"][start:stop].long()
        known = (q_targets != 3) & legal.cpu()
        sums["q_hits"] += float(((q.argmax(dim=2).cpu() == q_targets) & known).sum())
        sums["q_count"] += float(known.sum())
    return sums


HEAD_KEYS = ("entropy", "top", "optimal_mass", "value_hits", "q_hits", "q_count", "count")


def describe_heads(sums):
    n = max(1, sums["count"])
    return (f"| H {sums['entropy'] / n:.3f} top {sums['top'] / n:.3f} "
            f"opt {sums['optimal_mass'] / n:.3f} v-acc {sums['value_hits'] / n:.4f} "
            f"q-acc {sums['q_hits'] / max(1.0, sums['q_count']):.4f}")


def label(budget):
    return "policy" if budget == 0 else f"{budget} sims"


def unique_budgets(budgets):
    """Validate and deduplicate once, including one-shot iterables."""
    values = tuple(budgets)
    if not values or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                         for value in values):
        raise ValueError("Search budgets must be nonnegative integers, with at least one budget")
    return tuple(dict.fromkeys(values))


def evaluation_budgets(sims, *, directory):
    unique_budgets((sims,))
    if sims == 0:
        return (0,)
    return unique_budgets((0, 32, sims, 2 * sims if directory else 512))


def held_out_shards(shard_dir):
    """First-shard validation candidates, filtered by position before scoring."""
    return sorted(Path(shard_dir).glob("*-0000.pt"))


def load_validation_shard(path, limit):
    """Use the trainer's split, including legacy files; bound materialisation."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Position limit must be a positive integer")
    path = Path(path)
    shard = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    holdout = {tag.strip() for tag in os.environ.get("DISTILL_HOLDOUT_CONFIGS", "").split(",") if tag.strip()}
    chunks = list(filtered_chunks(shard, [], validation=True,
                  whole_board_held=path.stem.rsplit("-", 1)[0] in holdout, limit=limit))
    if not chunks:
        raise ValueError(f"{path}: no reserved validation positions; rebuild the validation shard")
    return {key: torch.cat([chunk[key] for chunk in chunks]) if key in SAMPLE_FIELDS else value
            for key, value in shard.items()}


def sweep(net, shard_paths, budgets, limit, device):
    """Blunder rates per shard, then pooled over every shard, over the
    chaos ones and over the classic ones."""
    budgets = unique_budgets(budgets)
    pools = {name: {budget: [0, 0] for budget in budgets} for name in ("all", "chaos", "classic")}
    heads = {name: dict.fromkeys(HEAD_KEYS, 0.0) for name in pools}
    lines = []
    for path in shard_paths:
        shard = load_validation_shard(path, limit)
        tag = path.stem.rsplit("-", 1)[0]
        parts = []
        counted = 0
        for budget in budgets:
            rate, counted = blunder_rate(net, shard, budget, limit, device)
            wrong = round(rate * counted)
            for name in ("all", "chaos" if "chaos" in tag else "classic"):
                pools[name][budget][0] += wrong
                pools[name][budget][1] += counted
            parts.append(f"{label(budget)} {rate:.4f}")
        stats = head_stats(net, shard, limit, device)
        for name in ("all", "chaos" if "chaos" in tag else "classic"):
            for key, value in stats.items():
                heads[name][key] += value
        parts.append(describe_heads(stats))
        lines.append(f"  {tag:16s} {'  '.join(parts)}  ({counted} positions)")
        print(lines[-1], flush=True)
    for name, pool in pools.items():
        if not any(count for _wrong, count in pool.values()):
            continue
        parts = [f"{label(budget)} {wrong / count:.4f}" if count else f"{label(budget)} n/a"
                 for budget, (wrong, count) in pool.items()]
        parts.append(describe_heads(heads[name]))
        counts = {count for _wrong, count in pool.values()}
        count_note = (f"{next(iter(counts))} positions" if len(counts) == 1 else
                      ", ".join(f"{label(budget)}: {count} positions"
                                for budget, (_wrong, count) in pool.items()))
        lines.append(f"pooled {name:8s} {'  '.join(parts)}  ({count_note})")
        print(lines[-1], flush=True)
    return lines


def main():
    model_path, target = sys.argv[1], sys.argv[2]
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    device = "cuda" if torch.cuda.is_available() else "cpu"
    directory = Path(target).is_dir()
    budgets = evaluation_budgets(sims, directory=directory)
    net = load(model_path, device)
    if directory:
        shards = held_out_shards(target)
        # sims=0 scores the raw heads alone: seconds instead of minutes.
        print(f"{Path(model_path).name}: {len(shards)} held-out shards, "
              f"{limit} positions each, budgets {'/'.join(str(b) for b in budgets)}", flush=True)
        sweep(net, shards, budgets, limit, device)
        return
    shard = torch.load(target, map_location="cpu", weights_only=True)
    rows, cols, connect = shard["config"]
    print(f"{Path(target).name}: {rows}x{cols} c{connect}, {limit} positions")
    for budget in budgets:
        rate, counted = blunder_rate(net, shard, budget, limit, device)
        print(f"  {label(budget):>10}: blunder rate {rate:.4f} over {counted}", flush=True)


if __name__ == "__main__":
    main()
