"""Batched MCTS self-play for any board up to 10x10, classic or Chaos.

AlphaZero-style: PUCT tree search guided by the network, Dirichlet noise
at the root, visit-count policy targets, temperature moves early. Games
run in lockstep so leaf evaluations batch onto the GPU. Threefold
repetition ends a game as a draw (and feeds the repetition planes).
Output shards use the same tensor schema as neural.build_dataset, so
distillation and replay data mix freely.

Usage:
  python -m neural.selfplay <model.pt> <out_dir> <rows> <cols> <connect> \
      <mode> <games> <sims> [shard_prefix]
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

import torch

from .chaos_game import (
    ACTION_INDEX, DRAW, LOSS, NOT_TERMINAL, WIN,
    empty_state, position_key, successors, to_planes,
)
from .model import PolicyValueNet

C_PUCT = 1.5
DIRICHLET_ALPHA = 0.4
DIRICHLET_FRACTION = 0.25
TEMPERATURE_PLIES = 12


class Node:
    __slots__ = ("edges", "prior", "visits", "value_sum", "children", "terminal")

    def __init__(self, edges, priors):
        self.edges = edges                    # list of Edge
        self.prior = priors                   # list of float, aligned with edges
        self.visits = [0] * len(edges)
        self.value_sum = [0.0] * len(edges)
        self.children = [None] * len(edges)   # Node or None
        self.terminal = None                  # set when this node is terminal


class Game:
    def __init__(self, rows, columns, connect, chaos, rng):
        self.connect = connect
        self.chaos = chaos
        self.rng = rng
        self.state = empty_state(rows, columns)
        self.history = {}
        self.reps = (False, False)
        self.records = []                     # (planes, legal, visit_policy)
        self.outcome = None                   # +1 mover-at-record wins ... per record sign
        self.root = None
        self.ply = 0

    def repetition_flags(self, state):
        seen = self.history.get(position_key(state), 0)
        return (seen >= 1, seen >= 2)


def evaluate_batch(net, device, items):
    """items: list of (state, connect, chaos, rep1, rep2) -> priors, values"""
    planes = torch.tensor([to_planes(s, k, c, r1, r2) for s, k, c, r1, r2 in items],
                          dtype=torch.float32, device=device)
    legal = torch.zeros((len(items), 13), dtype=torch.bool, device=device)
    edge_lists = []
    for row, (state, connect, chaos, _r1, _r2) in enumerate(items):
        edges = successors(state, connect, chaos)
        edge_lists.append(edges)
        for edge in edges:
            legal[row][ACTION_INDEX[edge.action]] = True
    with torch.no_grad():
        logits, wdl = net(planes, legal)
        probs = torch.softmax(logits, dim=1).cpu()
        dist = torch.softmax(wdl, dim=1).cpu()
    values = (dist[:, 2] - dist[:, 0]).tolist()   # P(win) - P(loss) for the mover
    priors = []
    for row, edges in enumerate(edge_lists):
        priors.append([probs[row][ACTION_INDEX[e.action]].item() for e in edges])
    return edge_lists, priors, values


def puct_select(node):
    total = math.sqrt(max(1, sum(node.visits)))
    best, best_score = 0, -1e9
    for index in range(len(node.edges)):
        q = (node.value_sum[index] / node.visits[index]) if node.visits[index] else 0.0
        u = C_PUCT * node.prior[index] * total / (1 + node.visits[index])
        if q + u > best_score:
            best, best_score = index, q + u
    return best


def run_selfplay(model_path, out_dir, rows, columns, connect, chaos,
                 games_total, sims, shard_prefix, seed=20260901):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = PolicyValueNet().to(device)
    payload = torch.load(model_path, map_location=device, weights_only=True)
    net.load_state_dict(payload["model"])
    net.eval()
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    finished = []
    active = [Game(rows, columns, connect, chaos, rng) for _ in range(games_total)]
    started = time.time()

    while active:
        # Expand roots that need a network evaluation.
        pending = [g for g in active if g.root is None]
        if pending:
            items = [(g.state, g.connect, g.chaos, *g.repetition_flags(g.state))
                     for g in pending]
            edge_lists, priors, _values = evaluate_batch(net, device, items)
            for g, edges, prior in zip(pending, edge_lists, priors):
                noise = [rng.gammavariate(DIRICHLET_ALPHA, 1.0) for _ in edges]
                total = sum(noise) or 1.0
                mixed = [(1 - DIRICHLET_FRACTION) * p + DIRICHLET_FRACTION * n / total
                         for p, n in zip(prior, noise)]
                g.root = Node(edges, mixed)

        for _sim in range(sims):
            leaves = []
            for g in active:
                node, path = g.root, []
                value = None
                while True:
                    index = puct_select(node)
                    path.append((node, index))
                    edge = node.edges[index]
                    if edge.terminal != NOT_TERMINAL:
                        value = float(edge.terminal)   # for the mover at `node`
                        break
                    child = node.children[index]
                    if child is None:
                        leaves.append((g, node, index, path))
                        break
                    node = child
                    value = None
                if value is not None:
                    for depth, (n, i) in enumerate(reversed(path)):
                        n.visits[i] += 1
                        n.value_sum[i] += value if depth % 2 == 0 else -value
            if leaves:
                items = [(n.edges[i].child, g.connect, g.chaos,
                          *g.repetition_flags(n.edges[i].child))
                         for g, n, i, _p in leaves]
                edge_lists, priors, values = evaluate_batch(net, device, items)
                for (g, node, index, path), edges, prior, value in zip(
                        leaves, edge_lists, priors, values):
                    node.children[index] = Node(edges, prior)
                    # value is from the CHILD mover's view; the edge's parent
                    # mover sees its negation, and so on up the path.
                    for depth, (n, i) in enumerate(reversed(path)):
                        n.value_sum[i] += -value if depth % 2 == 0 else value
                        n.visits[i] += 1

        # One move per game from visit counts.
        still_active = []
        for g in active:
            visits = g.root.visits
            policy = [0.0] * 13
            total = sum(visits) or 1
            for index, edge in enumerate(g.root.edges):
                policy[ACTION_INDEX[edge.action]] = visits[index] / total
            rep1, rep2 = g.repetition_flags(g.state)
            g.records.append((to_planes(g.state, g.connect, g.chaos, rep1, rep2),
                              [e.action for e in g.root.edges], policy))
            if g.ply < TEMPERATURE_PLIES:
                index = rng.choices(range(len(visits)), weights=[v + 1e-6 for v in visits])[0]
            else:
                index = max(range(len(visits)), key=lambda i: visits[i])
            edge = g.root.edges[index]
            g.history[position_key(g.state)] = g.history.get(position_key(g.state), 0) + 1

            if edge.terminal != NOT_TERMINAL:
                g.outcome = edge.terminal      # for the mover who just played
                finished.append(g)
                continue
            g.state = edge.child
            g.ply += 1
            key = position_key(g.state)
            if g.history.get(key, 0) >= 2:     # third occurrence ends it
                g.outcome = DRAW
                finished.append(g)
                continue
            child = g.root.children[index]
            g.root = child if child is not None else None
            still_active.append(g)
        active = still_active

    # Shards: outcome propagated backward with alternating perspective.
    planes_out, legal_out, policy_out, wdl_out = [], [], [], []
    for g in finished:
        value = g.outcome                      # for the mover at the LAST record
        for planes, legal_actions, policy in reversed(g.records):
            planes_out.append(planes)
            legal_row = [False] * 13
            for action in legal_actions:
                legal_row[ACTION_INDEX[action]] = True
            legal_out.append(legal_row)
            policy_out.append(policy)
            wdl_out.append(value + 1)
            value = value if value == DRAW else -value
    shard = {
        "planes": torch.tensor(planes_out, dtype=torch.float32),
        "legal": torch.tensor(legal_out, dtype=torch.bool),
        "policy": torch.tensor(policy_out, dtype=torch.float32),
        "wdl": torch.tensor(wdl_out, dtype=torch.int64),
        "config": (rows, columns, connect),
    }
    out = out_dir / f"{shard_prefix}-{int(time.time())}.pt"
    torch.save(shard, out)
    elapsed = time.time() - started
    print(f"self-play: {games_total} games, {len(planes_out)} positions, "
          f"{elapsed:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    model_path, out_dir = sys.argv[1], sys.argv[2]
    rows, columns, connect = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    chaos = sys.argv[6] != "classic"
    games_total, sims = int(sys.argv[7]), int(sys.argv[8])
    prefix = sys.argv[9] if len(sys.argv) > 9 else f"sp-{rows}x{columns}"
    run_selfplay(model_path, out_dir, rows, columns, connect, chaos,
                 games_total, sims, prefix)
