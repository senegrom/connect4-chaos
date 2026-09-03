"""GPU self-play across many board shapes and both rule sets at once.

Thousands of games advance in lockstep on the GPU: the environment is
tensorized (neural/gpu_env.py), and the move-selection search is a
two-ply minimax over network values - every legal action's children and
their replies are evaluated in batch, which is exactly the kind of work a
GPU does well. Move choice follows Gumbel AlphaZero: sample early moves
from logits + Gumbel noise + a scaled search score, play the argmax later;
the training target is the completed-Q distribution
softmax(logits + sigma * score) over legal actions. Threefold repetition
draws and feeds the repetition planes. Shards use the standard schema
(q = 3 everywhere, source = selfplay).

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

from .gpu_env import ACTIONS, BoardBatch, DRAW, NOT_TERMINAL, step
from .gpu_mcts import sample_actions, search, visit_policy
from .model import PolicyValueNet

OUTCOME_SCORE = torch.tensor([-1.0, 0.0, 1.0])   # loss, draw, win
SIGMA = 4.0                 # search-score scale added to logits
TEMPERATURE_PLIES = 12      # sample (with Gumbel noise) for this many plies
# Every game starts from the same empty board, so the opening is where a
# batch is most redundant: measured on 6x7 classic, 256 games occupied only
# 8 distinct positions at ply 2. Flattening the visit distribution for the
# first few plies spreads them out; later plies stay at the search's own
# distribution so the targets keep their quality.
OPENING_PLIES = int(os.environ.get("SELFPLAY_OPENING_PLIES", "6"))
OPENING_TEMPERATURE = float(os.environ.get("SELFPLAY_OPENING_TEMPERATURE", "1.6"))
MAX_PLIES = 220             # cycle guard beyond the threefold rule
EVAL_CHUNK = 32768
# Inference runs in bfloat16 on the GPU (tensor cores; ~2x the games per
# hour); SELFPLAY_FP32=1 restores full precision.
AUTOCAST = os.environ.get("SELFPLAY_FP32", "") != "1"
torch.backends.cudnn.benchmark = True


def forward(net, planes, legal):
    """Network forward under bf16 autocast; outputs returned as float32."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=AUTOCAST and planes.is_cuda):
        logits, wdl, q = net(planes, legal)
    return logits.float(), wdl.float(), q.float()


def all_shapes(rows=range(4, 11), cols=range(1, 11), connects=(3, 4, 5)):
    """Every playable board from 4x1 to 10x10, both rule sets.

    A network whose heads are size-agnostic should see the whole space
    rather than a handful of shapes: eighteen fixed boards are a narrow,
    self-similar slice of experience. Boards where no line can fit are
    skipped, since every game there is drawn by construction. Games on tiny
    boards are short, so the position mix still leans towards the large
    boards without any weighting."""
    shapes = []
    for row_count in rows:
        for col_count in cols:
            for connect in connects:
                if connect > max(row_count, col_count):
                    continue
                if row_count * col_count < connect:
                    continue
                shapes.append((row_count, col_count, connect, True))
                shapes.append((row_count, col_count, connect, False))
    return shapes


def parse_shapes(spec: str):
    """A comma list like "6x7c4chaos,8x8c5classic", or "all" for the whole
    space of playable boards."""
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


@torch.no_grad()
def evaluate(net, planes, legal):
    """Returns policy logits (N,13) and mover values (N,) = P(win)-P(loss)."""
    logits_out, values_out = [], []
    for start in range(0, len(planes), EVAL_CHUNK):
        logits, wdl, _q = forward(net, planes[start:start + EVAL_CHUNK], legal[start:start + EVAL_CHUNK])
        dist = torch.softmax(wdl, dim=1)
        logits_out.append(logits)
        values_out.append(dist[:, 2] - dist[:, 0])
    return torch.cat(logits_out), torch.cat(values_out)


FAST_REPLIES = os.environ.get("SELFPLAY_FAST", "") == "1"
# SELFPLAY_SIMS > 0 runs batched PUCT search (neural/gpu_mcts.py) and takes
# the visit distribution as the policy target - deeper and better targets
# than the two-ply lookahead, at one network evaluation per simulation.
SIMS = int(os.environ.get("SELFPLAY_SIMS", "0"))
# Playing every move at the depth a good training target needs is wasteful:
# most moves only have to be reasonable. With SELFPLAY_TARGET_SIMS set, a
# share of plies is searched deeply and teaches the policy, while the rest
# are searched cheaply and teach only the value head through the game's
# outcome. A quarter of plies at 256 simulations costs about the same as
# every ply at 88, and the targets are four times deeper.
TARGET_SIMS = int(os.environ.get("SELFPLAY_TARGET_SIMS", "0"))
TARGET_SHARE = float(os.environ.get("SELFPLAY_TARGET_SHARE", "0.25"))


@torch.no_grad()
def two_ply_scores(net, board: BoardBatch, legal, rep1, rep2):
    """score[n, a] = value for the mover of taking action a, after the
    opponent's best reply. Exact mode evaluates every reply position with
    the value head (169 evaluations per game and ply); SELFPLAY_FAST=1
    instead reads the opponent's reply values off the child's Q head (one
    evaluation per child, 13x cheaper), which is what that head is
    trained to predict."""
    n = len(board)
    device = board.device
    scores = torch.full((n, ACTIONS), -2.0, device=device)
    for a in range(ACTIONS):
        active = legal[:, a]
        if not active.any():
            continue
        child, outcome = step(board, torch.full((n,), a, dtype=torch.int64, device=device))
        terminal = outcome != NOT_TERMINAL
        score_a = outcome.float().clone()          # terminal: outcome for the mover
        alive = active & ~terminal
        if alive.any() and FAST_REPLIES:
            child_legal = child.legal()
            _logits, _wdl, q = forward(net, child.planes(rep1, rep2)[alive], child_legal[alive])
            expectation = (torch.softmax(q, dim=2) * OUTCOME_SCORE.to(device)).sum(dim=2)
            best_reply = expectation.masked_fill(~child_legal[alive], float("-inf")).max(dim=1).values
            score_a[alive] = -best_reply          # the child mover's best reply, negated
        elif alive.any():
            child_legal = child.legal()
            # Opponent (child mover) replies: value of each reply for the child
            # mover; the parent's score is the negation of the best reply.
            reply_best = torch.full((n,), -2.0, device=device)
            for b in range(ACTIONS):
                b_active = child_legal[:, b] & alive
                if not b_active.any():
                    continue
                grandchild, reply_outcome = step(child, torch.full((n,), b, dtype=torch.int64, device=device))
                reply_value = reply_outcome.float().clone()
                deep = b_active & (reply_outcome == NOT_TERMINAL)
                if deep.any():
                    planes = grandchild.planes(rep1, rep2)
                    _logits, leaf = evaluate(net, planes[deep], grandchild.legal()[deep])
                    # leaf is for the grandchild mover (= parent mover); the
                    # child mover's value of reply b is its negation.
                    reply_value[deep] = -leaf
                reply_best = torch.where(b_active, torch.maximum(reply_best, reply_value), reply_best)
            score_a[alive] = -reply_best[alive]
        scores[:, a] = torch.where(active, score_a, torch.full_like(score_a, -2.0))
    return scores


def run(model_path, out_dir, games_total, shapes, seed=20260902):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = random.Random(seed)
    payload = torch.load(model_path, map_location=device, weights_only=True)
    net = PolicyValueNet(*payload.get("arch", (192, 12, 48))).to(device)
    net.load_state_dict(payload["model"])
    net.eval()

    picks = [shapes[i % len(shapes)] for i in range(games_total)]
    rng.shuffle(picks)
    board = BoardBatch([p[0] for p in picks], [p[1] for p in picks],
                       [p[2] for p in picks], [p[3] for p in picks], device)
    n = len(board)
    keys = torch.rand((2, 10, 10), dtype=torch.float64, device=device, generator=None)
    histories = [dict() for _ in range(n)]
    records = [[] for _ in range(n)]          # (planes, legal, target) per game
    outcome_final = [None] * n
    # Finished games are dropped from the tensors: without this the batch
    # keeps its full width to the last ply, and most of the work goes to
    # games that ended long ago (measured: about 2.5x the GPU time).
    live = torch.arange(n, device=device)
    started = time.time()

    for ply in range(MAX_PLIES):
        if len(live) == 0:
            break
        alive = live.tolist()
        width = len(alive)
        hashes = board.position_hash(keys).cpu().tolist()
        rep_counts = torch.tensor([histories[alive[i]].get(hashes[i], 0) for i in range(width)],
                                  device=device)
        rep1, rep2 = rep_counts >= 1, rep_counts >= 2
        legal = board.legal()
        planes = board.planes(rep1, rep2)
        if SIMS > 0:
            deep = TARGET_SIMS > 0 and rng.random() < TARGET_SHARE
            visits, _value_sum = search(net, forward, board, rep1, rep2,
                                        TARGET_SIMS if deep else SIMS)
            target = visit_policy(visits, legal)
            greedy = torch.full((width,), ply >= TEMPERATURE_PLIES, dtype=torch.bool, device=device)
            # The training target stays the search distribution; only the
            # move actually played is drawn from a flatter one early on.
            played = target if ply >= OPENING_PLIES else visit_policy(visits, legal,
                                                                     OPENING_TEMPERATURE)
            choice = sample_actions(played, greedy)
            if TARGET_SIMS > 0 and not deep:
                # Shallow ply: the position still carries the game's outcome
                # for the value head, but an all-zero row tells the trainer
                # not to learn a policy from a search this thin.
                target = torch.zeros_like(target)
        else:
            logits, _values = evaluate(net, planes, legal)
            scores = two_ply_scores(net, board, legal, rep1, rep2)
            completed = logits + SIGMA * scores
            completed = completed.masked_fill(~legal, float("-inf"))
            target = torch.softmax(completed, dim=1)
            if ply < TEMPERATURE_PLIES:
                gumbel = -torch.log(-torch.log(torch.rand_like(completed).clamp(min=1e-9)))
                choice = (completed + gumbel).argmax(dim=1)
            else:
                choice = completed.argmax(dim=1)

        planes_cpu, legal_cpu, target_cpu = planes.cpu(), legal.cpu(), target.cpu()
        for i in range(width):
            game = alive[i]
            records[game].append((planes_cpu[i], legal_cpu[i], target_cpu[i]))
            histories[game][hashes[i]] = histories[game].get(hashes[i], 0) + 1

        child, outcome = step(board, choice)
        outcome_cpu = outcome.cpu().tolist()
        child_hashes = child.position_hash(keys).cpu().tolist()
        keep = []
        for i in range(width):
            game = alive[i]
            if outcome_cpu[i] != NOT_TERMINAL:
                outcome_final[game] = int(outcome_cpu[i])
            elif histories[game].get(child_hashes[i], 0) >= 2:
                outcome_final[game] = DRAW        # third occurrence: draw for the mover
            else:
                keep.append(i)
        if len(keep) < width:
            index = torch.tensor(keep, dtype=torch.int64, device=device)
            board = child.select(index)
            live = live[index]
        else:
            board = child
        if ply % 20 == 0:
            print(f"ply {ply}: active {len(keep)}/{n}, {time.time() - started:.0f}s", flush=True)

    # Games are meant to end by a win, a full board or the threefold rule;
    # anything still running at the cap is labelled a draw it may not have
    # earned, so the count is reported rather than hidden.
    capped = sum(1 for value in outcome_final if value is None)
    for i in range(n):
        if outcome_final[i] is None:
            outcome_final[i] = DRAW               # ply cap

    planes_out, legal_out, policy_out, wdl_out = [], [], [], []
    for i in range(n):
        value = outcome_final[i]
        for planes, legal, target in reversed(records[i]):
            planes_out.append(planes)
            legal_out.append(legal)
            policy_out.append(target)
            wdl_out.append(value + 1)
            value = value if value == DRAW else -value
    shard = {
        # Planes travel as uint8 (value x 10): lossless for the 0/1 planes
        # and the connect/10 plane, a quarter of float32 on disk and wire.
        "planes": (torch.stack(planes_out) * 10).round().to(torch.uint8),
        "planes_scale": 10,
        "legal": torch.stack(legal_out),
        "policy": torch.stack(policy_out),
        "wdl": torch.tensor(wdl_out, dtype=torch.int64),
        "q": torch.full((len(planes_out), 13), 3, dtype=torch.int64),
        "config": (0, 0, 0),
        "source": "selfplay",
        "shapes": picks,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"gpu-sp-{seed}-{int(time.time())}.pt"
    torch.save(shard, out)
    mode = f"mcts {SIMS} sims" if SIMS > 0 else ("2-ply fast" if FAST_REPLIES else "2-ply exact")
    if SIMS > 0 and TARGET_SIMS > 0:
        mode += f", {TARGET_SHARE:.0%} of plies at {TARGET_SIMS}"
    print(f"self-play [{mode}]: {n} games, {len(planes_out)} positions, "
          f"{capped} hit the {MAX_PLIES}-ply cap, {time.time() - started:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    model_path, out_dir, games_total = sys.argv[1], sys.argv[2], int(sys.argv[3])
    shapes = parse_shapes(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 20260902
    run(model_path, out_dir, games_total, shapes, seed)
