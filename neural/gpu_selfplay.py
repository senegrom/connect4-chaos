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
from .model import PolicyValueNet

OUTCOME_SCORE = torch.tensor([-1.0, 0.0, 1.0])   # loss, draw, win
SIGMA = 4.0                 # search-score scale added to logits
TEMPERATURE_PLIES = 12      # sample (with Gumbel noise) for this many plies
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


def parse_shapes(spec: str):
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
    active = torch.ones(n, dtype=torch.bool, device=device)
    started = time.time()

    for ply in range(MAX_PLIES):
        if not active.any():
            break
        hashes = board.position_hash(keys).cpu().tolist()
        rep_counts = torch.tensor([histories[i].get(hashes[i], 0) for i in range(n)], device=device)
        rep1, rep2 = rep_counts >= 1, rep_counts >= 2
        legal = board.legal()
        planes = board.planes(rep1, rep2)
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
        for i in range(n):
            if active[i]:
                records[i].append((planes_cpu[i], legal_cpu[i], target_cpu[i]))
                histories[i][hashes[i]] = histories[i].get(hashes[i], 0) + 1

        child, outcome = step(board, choice)
        outcome_cpu = outcome.cpu()
        child_hashes = child.position_hash(keys).cpu().tolist()
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        for i in range(n):
            if not active[i]:
                continue
            if outcome_cpu[i] != NOT_TERMINAL:
                outcome_final[i] = int(outcome_cpu[i])
                finished[i] = True
            elif histories[i].get(child_hashes[i], 0) >= 2:
                outcome_final[i] = DRAW           # third occurrence: draw for the mover
                finished[i] = True
        active &= ~finished
        board = child
        if ply % 20 == 0:
            print(f"ply {ply}: active {int(active.sum())}/{n}, {time.time() - started:.0f}s", flush=True)

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
        "planes": torch.stack(planes_out),
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
    print(f"self-play: {n} games, {len(planes_out)} positions, {time.time() - started:.0f}s -> {out}",
          flush=True)


if __name__ == "__main__":
    model_path, out_dir, games_total = sys.argv[1], sys.argv[2], int(sys.argv[3])
    shapes = parse_shapes(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 20260902
    run(model_path, out_dir, games_total, shapes, seed)
