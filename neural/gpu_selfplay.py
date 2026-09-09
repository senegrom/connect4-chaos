"""GPU self-play across many board shapes and both rule sets at once.

Thousands of games advance in lockstep on the GPU. Repetition history and
training records stay on-device until the batch finishes, so the actor does not
serialize every ply through Python dictionaries or CPU copies. With
SELFPLAY_TARGET_SIMS set, only a share of plies is searched deeply and teaches
the policy; shallow plies still supervise the value head. The search replays
CUDA graphs (neural/gpu_mcts.py) and the network runs with its BatchNorm
folded away; SELFPLAY_GRAPHS=0 and SELFPLAY_CHANNELS_LAST=0 switch those off,
SELFPLAY_PROFILE=1 prints a timeline of the run.

Usage:
  python -m neural.gpu_selfplay <model.pt> <out_dir> <games> <shapes> [seed]
  shapes: comma list like 6x7c4chaos,6x7c4classic,8x8c4chaos,5x10c4chaos
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import torch

from neural.training_config import DEFAULT_SIMS, validate_selfplay
from .gpu_env import ACTIONS, BoardBatch, DRAW, NOT_TERMINAL, hash_keys, step
from .gpu_history import DenseHistory, history_counts
from .data_split import SPLIT_VERSION, validation_mask
from . import gpu_mcts
from .gpu_mcts import improved_policy, root_value, sample_actions, search, search_root, visit_policy
from .model import FusedInferenceNet, PolicyValueNet, fold_batchnorm

TEMPERATURE_PLIES = 12
OPENING_PLIES = int(os.environ.get("SELFPLAY_OPENING_PLIES", "6"))
OPENING_TEMPERATURE = float(os.environ.get("SELFPLAY_OPENING_TEMPERATURE", "1.6"))
# Every game starts from the same empty board, and after the sampled opening
# the play is greedy, so a generation's games funnel into the lines the
# policy prefers and the next generation sees them again. A share of games
# therefore opens with uniformly random legal moves - between one and
# RANDOM_OPENING_PLIES of them, fewer on tiny boards - before the search
# takes over. The search still runs on those positions and its visit
# distribution is still the target, so the randomness only decides which
# positions get taught, not what they are taught.
RANDOM_OPENING_SHARE = float(os.environ.get("SELFPLAY_RANDOM_OPENING_SHARE", "0.5"))
RANDOM_OPENING_PLIES = int(os.environ.get("SELFPLAY_RANDOM_OPENING_PLIES", "4"))
# "visits": the normalised visit counts of the deep plies teach the policy,
# shallow plies teach only the value head. "gumbel": every ply teaches the
# policy through Gumbel MuZero's improved policy (gpu_mcts.improved_policy).
POLICY_TARGET = os.environ.get("SELFPLAY_POLICY_TARGET", "visits")
if POLICY_TARGET not in ("visits", "gumbel"):
    raise ValueError("SELFPLAY_POLICY_TARGET must be 'visits' or 'gumbel'")
GUMBEL_C_VISIT = float(os.environ.get("SELFPLAY_GUMBEL_C_VISIT", "50"))
GUMBEL_C_SCALE = float(os.environ.get("SELFPLAY_GUMBEL_C_SCALE", "1.0"))
MAX_PLIES = 220
AUTOCAST = os.environ.get("SELFPLAY_FP32", "") != "1"
CHANNELS_LAST = os.environ.get("SELFPLAY_CHANNELS_LAST", "1") != "0"
FUSED = os.environ.get("SELFPLAY_FUSED", "1") != "0"
PROFILE = os.environ.get("SELFPLAY_PROFILE", "") == "1"
torch.backends.cudnn.benchmark = True


def forward(net, planes, legal):
    """Network forward; outputs returned as float32. A FusedInferenceNet
    runs in its own dtype with fused cuDNN kernels; any other network runs
    under bf16 autocast (weight cache off: it is not safe under CUDA graph
    capture, and recasting the weights costs nothing measurable)."""
    if getattr(net, "own_dtype", False):
        logits, wdl, q = net(planes, legal)
        return logits.float(), wdl.float(), q.float()
    if planes.is_cuda and CHANNELS_LAST:
        planes = planes.contiguous(memory_format=torch.channels_last)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=AUTOCAST and planes.is_cuda, cache_enabled=False):
        logits, wdl, q = net(planes, legal)
    return logits.float(), wdl.float(), q.float()


def all_shapes(rows=range(4, 11), cols=range(1, 11), connects=(3, 4, 5)):
    """Every playable board from 4x1 to 10x10, both rule sets."""
    shapes = []
    for row_count in rows:
        for col_count in cols:
            for connect in connects:
                if connect > max(row_count, col_count) or row_count * col_count < connect:
                    continue
                shapes.append((row_count, col_count, connect, True))
                shapes.append((row_count, col_count, connect, False))
    return shapes


def parse_shapes(spec: str):
    """A comma list like ``6x7c4chaos,8x8c5classic``, or ``all``."""
    if spec.strip() == "all":
        return all_shapes()
    shapes = []
    for item in spec.split(","):
        item = item.strip()
        dims, rest = item.split("x")
        cols, rest = rest.split("c", 1)
        connect = int("".join(ch for ch in rest if ch.isdigit()))
        mode = "classic" if "classic" in rest else "chaos"
        shapes.append((int(dims), int(cols), connect, mode != "classic"))
    return shapes


SIMS = int(os.environ.get("SELFPLAY_SIMS", str(DEFAULT_SIMS)))
if SIMS <= 0:
    raise SystemExit("SELFPLAY_SIMS must be positive; the two-ply mode was removed.")
TARGET_SIMS = int(os.environ.get("SELFPLAY_TARGET_SIMS", "0"))
TARGET_SHARE = float(os.environ.get("SELFPLAY_TARGET_SHARE", "0.25"))


def _prepare_network(payload, device):
    """The checkpoint as an inference network. On CUDA: BatchNorm folded
    and every conv+bias+ReLU fused into one cuDNN kernel in bf16
    (SELFPLAY_FUSED=0 keeps the plain folded network under autocast,
    SELFPLAY_FP32=1 keeps full precision). No torch.compile: the search
    replays whole simulations as CUDA graphs, and compiling the network for
    every batch width the actor passes through cost more than it saved
    (measured 2.7x slower)."""
    net = PolicyValueNet(*payload.get("arch", (192, 12, 48))).to(device)
    net.load_state_dict(payload["model"])
    if str(device) == "cuda" and AUTOCAST and FUSED:
        return FusedInferenceNet(net)
    net = fold_batchnorm(net)
    if str(device) == "cuda" and CHANNELS_LAST:
        net.to(memory_format=torch.channels_last)
    return net


def _finish_shard(record_planes, record_legal, record_policy, record_valid,
                  outcome_final, end_ply, record_root=None):
    """Flatten completed games and derive mover-relative WDL targets on GPU."""
    completed = end_ply >= 0
    valid = record_valid & completed[None, :]
    if not bool(valid.any()):
        raise RuntimeError("Self-play produced no completed training positions")
    plies = torch.arange(record_valid.shape[0], device=record_valid.device)[:, None]
    distance = end_ply[None, :] - plies
    final = outcome_final[None, :].expand_as(distance)
    signed = torch.where(final == DRAW, torch.zeros_like(final),
                         torch.where((distance & 1) == 0, final, -final))
    return {
        "planes": record_planes[valid].cpu(),
        "planes_scale": 10,
        "legal": record_legal[valid].cpu(),
        "policy": record_policy[valid].cpu(),
        "wdl": (signed[valid] + 1).to(torch.uint8).cpu(),
        # The search's own value of each position, for the player to move:
        # a lower-variance companion to the game's outcome.
        **({"root_value": record_root[valid].to(torch.float16).cpu()} if record_root is not None else {}),
        # Replay has no exact per-action targets. Avoid an all-3 int64 tensor
        # (~416 MB at four million rows); the learner materializes 3 lazily.
        "q_default": 3,
        "source": "selfplay",
    }, int(completed.logical_not().sum().item()), int(valid.sum().item())


def run(model_path, out_dir, games_total, shapes, seed=20260902):
    spec = ",".join(f"{r}x{c}c{k}{'chaos' if chaos else 'classic'}" for r, c, k, chaos in shapes)
    validate_selfplay(games_total, SIMS, spec, TARGET_SIMS, TARGET_SHARE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = random.Random(seed)
    payload = torch.load(model_path, map_location=device, weights_only=True)
    net = _prepare_network(payload, device)

    picks = [shapes[i % len(shapes)] for i in range(games_total)]
    rng.shuffle(picks)
    board = BoardBatch([p[0] for p in picks], [p[1] for p in picks],
                       [p[2] for p in picks], [p[3] for p in picks], device)
    n = len(board)
    random_plies = torch.tensor([
        rng.randint(1, max(1, min(RANDOM_OPENING_PLIES, (p[0] * p[1]) // 8)))
        if rng.random() < RANDOM_OPENING_SHARE else 0
        for p in picks], dtype=torch.int64, device=device)
    keys = hash_keys(device)
    history = DenseHistory(n, MAX_PLIES + 1, device)

    # Worst-case storage is ~0.7 GB for 4096 x 220 games, small on an H100.
    # It replaces per-ply GPU->CPU transfers and thousands of Python lists.
    record_planes = torch.empty((MAX_PLIES, n, 7, 10, 10), dtype=torch.uint8, device=device)
    record_legal = torch.empty((MAX_PLIES, n, ACTIONS), dtype=torch.bool, device=device)
    record_policy = torch.empty((MAX_PLIES, n, ACTIONS), dtype=torch.float32, device=device)
    record_valid = torch.zeros((MAX_PLIES, n), dtype=torch.bool, device=device)
    record_root = torch.zeros((MAX_PLIES, n), dtype=torch.float32, device=device)
    outcome_final = torch.full((n,), 9, dtype=torch.int64, device=device)  # 9 = unfinished
    end_ply = torch.full((n,), -1, dtype=torch.int64, device=device)

    live = torch.arange(n, device=device)
    started = time.time()
    timeline = []
    for ply in range(MAX_PLIES):
        if len(live) == 0:
            break
        if PROFILE and ply % 5 == 0:
            if device == "cuda":
                torch.cuda.synchronize()
            timeline.append(f"{ply}:{len(live)}:{time.time() - started:.0f}s")
        side = ply % 2 == 1
        hashes = board.position_hash(keys, side)
        history_view = history.search_view(live)
        rep_counts = history_counts(history_view, hashes)
        rep1, rep2 = rep_counts >= 1, rep_counts >= 2
        legal = board.legal()
        planes = board.planes(rep1, rep2)
        deep = TARGET_SIMS > 0 and rng.random() < TARGET_SHARE
        visits, value_sum, prior, net_value = search_root(
            net, forward, board, rep1, rep2,
            TARGET_SIMS if deep else SIMS,
            side=side, history=history_view, keys=keys,
        )
        if POLICY_TARGET == "gumbel":
            target = improved_policy(prior, visits, value_sum, net_value, legal,
                                     GUMBEL_C_VISIT, GUMBEL_C_SCALE)
        else:
            target = visit_policy(visits, legal)
        searched_value = root_value(visits, value_sum)
        greedy = torch.full((len(live),), ply >= TEMPERATURE_PLIES,
                            dtype=torch.bool, device=device)
        played = target if ply >= OPENING_PLIES else visit_policy(
            visits, legal, OPENING_TEMPERATURE)
        choice = sample_actions(played, greedy)
        opening = random_plies[live] > ply
        if bool(opening.any()):
            uniform = torch.multinomial(legal.float().clamp(min=1e-12), 1).squeeze(1)
            choice = torch.where(opening, uniform, choice)
        if TARGET_SIMS > 0 and not deep and POLICY_TARGET != "gumbel":
            target = torch.zeros_like(target)

        # Compact before recording: four times less storage than float32 planes
        # and no PCIe transfer until the full batch has finished.
        record_planes[ply, live] = (planes * 10).round().to(torch.uint8)
        record_legal[ply, live] = legal
        record_policy[ply, live] = target
        record_valid[ply, live] = True
        record_root[ply, live] = searched_value

        is_drop = choice < 10
        history.append_or_reset(live, hashes, is_drop)
        child, outcome = step(board, choice)
        child_hashes = child.position_hash(keys, not side)
        repeated = (outcome == NOT_TERMINAL) & (history.counts(live, child_hashes) >= 2)
        finished = (outcome != NOT_TERMINAL) | repeated
        finished_games = live[finished]
        outcome_final[finished_games] = torch.where(
            outcome[finished] != NOT_TERMINAL, outcome[finished],
            torch.zeros_like(outcome[finished]))
        end_ply[finished_games] = ply

        keep = (~finished).nonzero(as_tuple=False).squeeze(1)
        board = child.select(keep)
        live = live[keep]
        if ply % 20 == 0:
            print(f"ply {ply}: active {len(live)}/{n}, {time.time() - started:.0f}s", flush=True)

    shard, capped, positions = _finish_shard(
        record_planes, record_legal, record_policy, record_valid, outcome_final, end_ply,
        record_root)
    shard["config"] = (0, 0, 0)
    shard["shapes"] = picks
    # Compute the stable train/validation partition once per generated row.
    # Learner generations can then filter replay with a cheap boolean slice
    # instead of re-running BLAKE2b over the whole rolling window.
    shard["validation"] = validation_mask(shard["planes"], shard["planes_scale"])
    shard["split_version"] = SPLIT_VERSION
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"gpu-sp-{seed}-{int(time.time())}.pt"
    torch.save(shard, out)
    mode = f"mcts {SIMS} sims"
    if TARGET_SIMS > 0:
        mode += f", {TARGET_SHARE:.0%} of plies at {TARGET_SIMS}"
    if RANDOM_OPENING_SHARE > 0 and RANDOM_OPENING_PLIES > 0:
        mode += f", {RANDOM_OPENING_SHARE:.0%} random openings up to {RANDOM_OPENING_PLIES} plies"
    if POLICY_TARGET == "gumbel":
        mode += ", gumbel policy targets on every ply"
    suffix = f", {capped} capped games discarded" if capped else ""
    if PROFILE:
        stats = gpu_mcts.STATS
        print(f"profile: timeline {' '.join(timeline)}; {stats['workspaces']} workspaces, "
              f"{stats['captures']} graphs captured in {stats['capture_seconds']:.1f}s", flush=True)
    print(f"self-play [{mode}]: {n} games, {positions} positions{suffix}, "
          f"{time.time() - started:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    model_path, out_dir, games_total = sys.argv[1], sys.argv[2], int(sys.argv[3])
    shapes = parse_shapes(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 20260902
    run(model_path, out_dir, games_total, shapes, seed)
