"""Batched PUCT search: one AlphaZero-style tree per game, all in tensors.

Thousands of games search in lockstep. A simulation descends every tree at
once over the visit statistics, expands one leaf per game and backs the
value up its path, so a simulation costs one batched network evaluation
and one environment step regardless of how deep the trees have grown.
Search depth therefore grows with the simulation count, which is what a
fixed-depth lookahead cannot do.

Each node keeps its board, so descending is pure indexing: the position of
a child is read, never replayed. That storage is small (a 4096-game forest
with 64 simulations holds its boards in about 80 MB) and it is what keeps
the cost per simulation flat in the depth.

Values are always "for the player to move at this node"; an edge's value
is the negation of the value of the position it leads to.
"""

from __future__ import annotations

import torch

from .gpu_env import ACTIONS, CANVAS, NOT_TERMINAL, BoardBatch, step

C_PUCT = 1.5
DIRICHLET_ALPHA = 0.4
DIRICHLET_FRACTION = 0.25
MAX_DEPTH = 64            # descent guard; trees are far shallower in practice

_SCALARS = ("rows", "cols", "connect", "chaos", "pieces")


class Forest:
    """One tree per game. Edge statistics are [game, node, action]; each
    node also stores the position it stands for."""

    def __init__(self, games: int, sims: int, device):
        capacity = sims + 2
        shape = (games, capacity, ACTIONS)
        self.games, self.capacity, self.device = games, capacity, device
        self.child = torch.full(shape, -1, dtype=torch.int64, device=device)
        self.visits = torch.zeros(shape, device=device)
        self.value_sum = torch.zeros(shape, device=device)
        self.prior = torch.zeros(shape, device=device)
        self.legal = torch.zeros(shape, dtype=torch.bool, device=device)
        # Outcome of a terminal edge, for the mover at its parent node.
        self.edge_terminal = torch.full(shape, NOT_TERMINAL, dtype=torch.int64, device=device)
        self.size = torch.ones((games,), dtype=torch.int64, device=device)
        # Game index for every advanced-indexing read; kept separate from the
        # stored boards, whose own "rows" field is a board height.
        self.rows = torch.arange(games, device=device)
        board_shape = (games, capacity, CANVAS, CANVAS)
        self.mover = torch.zeros(board_shape, dtype=torch.bool, device=device)
        self.opponent = torch.zeros(board_shape, dtype=torch.bool, device=device)
        self.heights = torch.zeros((games, capacity, CANVAS), dtype=torch.int64, device=device)
        self.scalars = {name: torch.zeros((games, capacity),
                                          dtype=torch.bool if name == "chaos" else torch.int64,
                                          device=device)
                        for name in _SCALARS}

    def store(self, node, board: BoardBatch):
        index = (self.rows, node)
        self.mover[index] = board.mover
        self.opponent[index] = board.opponent
        self.heights[index] = board.heights
        for name, store in self.scalars.items():
            store[index] = getattr(board, name)

    def load(self, node) -> BoardBatch:
        index = (self.rows, node)
        board = BoardBatch.__new__(BoardBatch)
        board.device = self.device
        board.mover = self.mover[index]
        board.opponent = self.opponent[index]
        board.heights = self.heights[index]
        for name, store in self.scalars.items():
            setattr(board, name, store[index])
        return board

    def puct(self, node):
        """Action scores at one node per game."""
        index = (self.rows, node)
        visits, value_sum = self.visits[index], self.value_sum[index]
        q = torch.where(visits > 0, value_sum / visits.clamp(min=1), torch.zeros_like(visits))
        total = visits.sum(dim=1, keepdim=True).clamp(min=1).sqrt()
        u = C_PUCT * self.prior[index] * total / (1.0 + visits)
        return (q + u).masked_fill(~self.legal[index], float("-inf"))

    def install(self, node, logits, legal):
        """Writes priors and legality into a node."""
        prior = torch.softmax(logits.masked_fill(~legal, float("-inf")), dim=1)
        self.prior[self.rows, node] = torch.nan_to_num(prior)
        self.legal[self.rows, node] = legal


@torch.no_grad()
def search(net, forward, board: BoardBatch, rep1, rep2, sims: int,
           add_noise: bool = True, generator=None):
    """Runs `sims` simulations from `board`; returns root visits and values.

    `forward(net, planes, legal)` evaluates a batch and returns
    (policy logits, wdl logits, q logits), so callers share one autocast
    policy with the rest of the pipeline.
    """
    games, device = len(board), board.device
    forest = Forest(games, sims, device)
    rows = forest.rows
    root = torch.zeros(games, dtype=torch.int64, device=device)

    root_legal = board.legal()
    logits, _wdl, _q = forward(net, board.planes(rep1, rep2), root_legal)
    forest.install(root, logits, root_legal)
    forest.store(root, board)
    if add_noise:
        noise = torch.distributions.Dirichlet(
            torch.full((ACTIONS,), DIRICHLET_ALPHA, device=device)).sample((games,))
        noise = (noise * root_legal)
        noise = noise / noise.sum(dim=1, keepdim=True).clamp(min=1e-9)
        forest.prior[rows, 0] = ((1 - DIRICHLET_FRACTION) * forest.prior[rows, 0]
                                 + DIRICHLET_FRACTION * noise)

    playable = root_legal.any(dim=1)
    for _ in range(sims):
        node = root.clone()
        alive = playable.clone()
        path_nodes = torch.full((games, MAX_DEPTH), -1, dtype=torch.int64, device=device)
        path_actions = torch.full((games, MAX_DEPTH), -1, dtype=torch.int64, device=device)
        depth = torch.zeros(games, dtype=torch.int64, device=device)
        leaf_value = torch.zeros(games, device=device)
        expanding = torch.zeros(games, dtype=torch.bool, device=device)

        # --- descent: pure indexing over the tree, no environment steps ----
        for level in range(MAX_DEPTH):
            if not bool(alive.any()):
                break
            action = forest.puct(node).argmax(dim=1)
            path_nodes[:, level] = torch.where(alive, node, path_nodes[:, level])
            path_actions[:, level] = torch.where(alive, action, path_actions[:, level])
            depth = torch.where(alive, depth + 1, depth)

            child = forest.child[rows, node, action]
            terminal = forest.edge_terminal[rows, node, action]
            hit_terminal = alive & (terminal != NOT_TERMINAL)
            leaf_value = torch.where(hit_terminal, terminal.float(), leaf_value)
            expanding |= alive & ~hit_terminal & (child < 0)

            descend = alive & ~hit_terminal & (child >= 0)
            node = torch.where(descend, child, node)
            alive = descend

        # --- expansion: one environment step and one evaluation ------------
        last = (depth - 1).clamp(min=0)
        parent = path_nodes[rows, last]
        action = path_actions[rows, last]
        if bool(expanding.any()):
            leaf_board, outcome = step(forest.load(parent), action.clamp(min=0))
            terminal_now = expanding & (outcome != NOT_TERMINAL)
            if bool(terminal_now.any()):
                index = (rows[terminal_now], parent[terminal_now], action[terminal_now])
                forest.edge_terminal[index] = outcome[terminal_now]
                leaf_value = torch.where(terminal_now, outcome.float(), leaf_value)

            fresh = expanding & (outcome == NOT_TERMINAL)
            if bool(fresh.any()):
                leaf_legal = leaf_board.legal()
                logits, wdl, _q = forward(net, leaf_board.planes(rep1, rep2), leaf_legal)
                distribution = torch.softmax(wdl, dim=1)
                child_value = distribution[:, 2] - distribution[:, 0]
                new_index = forest.size.clamp(max=forest.capacity - 1)
                forest.child[rows[fresh], parent[fresh], action[fresh]] = new_index[fresh]
                forest.install(new_index, logits, leaf_legal)
                forest.store(new_index, leaf_board)
                # Nodes belonging to games that did not expand stay unused.
                idle = ~fresh
                forest.legal[rows[idle], new_index[idle]] = False
                forest.prior[rows[idle], new_index[idle]] = 0.0
                forest.size = torch.where(fresh, forest.size + 1, forest.size)
                # The leaf's value is for its own mover; its parent edge negates it.
                leaf_value = torch.where(fresh, -child_value, leaf_value)

        # --- backup: deepest edge takes +leaf_value, alternating upward ----
        for level in range(MAX_DEPTH - 1, -1, -1):
            active = path_actions[:, level] >= 0
            if not bool(active.any()):
                continue
            sign = torch.where(((depth - 1 - level) % 2) == 0, 1.0, -1.0)
            index = (rows[active], path_nodes[active, level], path_actions[active, level])
            forest.visits[index] += 1.0
            forest.value_sum[index] += (sign * leaf_value)[active]

    return forest.visits[rows, 0], forest.value_sum[rows, 0]


def visit_policy(visits, legal, temperature: float = 1.0):
    """Normalised visit distribution over legal actions (the AlphaZero
    policy target); falls back to legal-uniform when nothing was visited,
    which happens only when every move ends the game at once."""
    counts = visits.masked_fill(~legal, 0.0)
    if temperature != 1.0:
        counts = counts.clamp(min=0) ** (1.0 / temperature)
    total = counts.sum(dim=1, keepdim=True)
    uniform = legal.float()
    uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp(min=1)
    return torch.where(total > 0, counts / total.clamp(min=1e-9), uniform)


def root_value(visits, value_sum):
    """Search value of the root position, for the player to move."""
    total = visits.sum(dim=1)
    return torch.where(total > 0, value_sum.sum(dim=1) / total.clamp(min=1), torch.zeros_like(total))


def sample_actions(policy, greedy, generator=None):
    """Samples from the visit distribution, or takes its argmax."""
    picked = torch.multinomial(policy.clamp(min=0) + 1e-12, 1, generator=generator).squeeze(1)
    return torch.where(greedy, policy.argmax(dim=1), picked)
