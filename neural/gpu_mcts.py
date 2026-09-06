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

Each node also keeps the hash of its position, so the threefold rule holds
inside the tree exactly as the game applies it: a leaf whose position has
already occurred twice - in the game's own history or among the ancestors
on its search path - is a terminal draw, and a leaf whose position has
occurred once is evaluated with its repetition plane set. Chaos games are
full of transform cycles (a flip undoes a flip); without this the search
valued a position it could only reach by repeating as if it were fresh,
and a side that could claim the draw by repeating never saw it.

Values are always "for the player to move at this node"; an edge's value
is the negation of the value of the position it leads to.
"""

from __future__ import annotations

import torch

from .gpu_env import ACTIONS, CANVAS, DRAW, NOT_TERMINAL, BoardBatch, hash_keys, step
from .gpu_history import history_counts

C_PUCT = 1.5
# Value assumed for an action the search has not tried yet. The network's
# per-action head already predicts the outcome of every move and the search
# used to discard it, so an untried action looked like a draw: too
# optimistic in a losing position, too pessimistic in a winning one. Using
# the head instead gives every action a real starting value, which is worth
# more than the simulation it would take to find out.
DIRICHLET_ALPHA = 0.4
DIRICHLET_FRACTION = 0.25
MAX_DEPTH = 64            # descent guard; trees are far shallower in practice

_SCALARS = ("rows", "cols", "connect", "chaos", "pieces")


class Forest:
    """One tree per game. Edge statistics are [game, node, action]; each
    node also stores the position it stands for and that position's hash."""

    def __init__(self, games: int, sims: int, device, max_connect: int = 10, any_chaos=True):
        capacity = sims + 2
        shape = (games, capacity, ACTIONS)
        self.games, self.capacity, self.device = games, capacity, device
        self.max_connect = max_connect
        self.any_chaos = any_chaos
        self.child = torch.full(shape, -1, dtype=torch.int64, device=device)
        self.visits = torch.zeros(shape, device=device)
        self.value_sum = torch.zeros(shape, device=device)
        self.prior = torch.zeros(shape, device=device)
        self.legal = torch.zeros(shape, dtype=torch.bool, device=device)
        # Outcome of a terminal edge, for the mover at its parent node.
        self.edge_terminal = torch.full(shape, NOT_TERMINAL, dtype=torch.int64, device=device)
        self.edge_value = torch.zeros(shape, device=device)      # from the per-action head
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
        self.hash = torch.zeros((games, capacity), dtype=torch.int64, device=device)
        # Cached WDL expectation for the node's mover, including the repetition
        # features used at expansion. A depth cutoff is not a terminal draw.
        self.value = torch.zeros((games, capacity), device=device)

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
        board.max_connect = self.max_connect
        board.any_chaos = self.any_chaos
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
        untried = self.edge_value[index]
        q = torch.where(visits > 0, value_sum / visits.clamp(min=1), untried)
        total = visits.sum(dim=1, keepdim=True).clamp(min=1).sqrt()
        u = C_PUCT * self.prior[index] * total / (1.0 + visits)
        return (q + u).masked_fill(~self.legal[index], float("-inf"))

    def install(self, node, logits, legal, q_logits=None):
        """Writes priors, legality and per-action values into a node."""
        prior = torch.softmax(logits.masked_fill(~legal, float("-inf")), dim=1)
        self.prior[self.rows, node] = torch.nan_to_num(prior)
        self.legal[self.rows, node] = legal
        if q_logits is None:
            self.edge_value[self.rows, node] = 0.0
        else:
            distribution = torch.softmax(q_logits.float(), dim=2)
            expected = distribution[:, :, 2] - distribution[:, :, 0]
            self.edge_value[self.rows, node] = torch.nan_to_num(expected)


def pack_history(eras, device):
    """Pads per-game {hash: count} dicts of the positions that could still
    recur into the (hashes, counts) pair search() takes; None when no game
    has any, which is every classic batch."""
    width = max((len(era) for era in eras), default=0)
    if width == 0:
        return None
    hashes = [list(era.keys()) + [0] * (width - len(era)) for era in eras]
    counts = [list(era.values()) + [0] * (width - len(era)) for era in eras]
    return (torch.tensor(hashes, dtype=torch.int64, device=device),
            torch.tensor(counts, dtype=torch.int64, device=device))


@torch.no_grad()
def search_tree(net, forward, board: BoardBatch, rep1, rep2, sims: int,
                add_noise: bool = True, generator=None,
                side=None, history=None, keys=None) -> Forest:
    """Runs `sims` simulations from `board` and returns the forest.

    `forward(net, planes, legal)` evaluates a batch and returns
    (policy logits, wdl logits, q logits), so callers share one autocast
    policy with the rest of the pipeline.

    rep1/rep2 are the root's repetition flags. `side` marks the games in
    which the second player is to move (a bool per game, or one for all):
    it only tells the two empty boards apart, since the planes are
    mover-relative. `history` is pack_history() of the positions each game
    has seen that could recur, hashed with `keys` from hash_keys(); the
    search then counts occurrences the way the actor does. Without them,
    repetition is still tracked along the search path itself.
    """
    games, device = len(board), board.device
    forest = Forest(games, sims, device, getattr(board, "max_connect", 10),
                    getattr(board, "any_chaos", True))
    rows = forest.rows
    root = torch.zeros(games, dtype=torch.int64, device=device)
    if keys is None:
        keys = hash_keys(device)
    if side is None:
        side = torch.zeros(games, dtype=torch.bool, device=device)
    else:
        side = torch.as_tensor(side, dtype=torch.bool, device=device)
        if side.dim() == 0:
            side = side.expand(games)

    root_legal = board.legal()
    logits, root_wdl, q_logits = forward(net, board.planes(rep1, rep2), root_legal)
    forest.install(root, logits, root_legal, q_logits)
    root_distribution = torch.softmax(root_wdl.float(), dim=1)
    forest.value[rows, root] = root_distribution[:, 2] - root_distribution[:, 0]
    forest.store(root, board)
    forest.hash[rows, 0] = board.position_hash(keys, side)
    if add_noise:
        noise = torch.distributions.Dirichlet(
            torch.full((ACTIONS,), DIRICHLET_ALPHA, device=device)).sample((games,))
        noise = (noise * root_legal)
        noise = noise / noise.sum(dim=1, keepdim=True).clamp(min=1e-9)
        forest.prior[rows, 0] = ((1 - DIRICHLET_FRACTION) * forest.prior[rows, 0]
                                 + DIRICHLET_FRACTION * noise)

    playable = root_legal.any(dim=1)
    # These are large for thousands of parallel games. Reuse the allocations
    # across simulations instead of asking the CUDA allocator for two fresh
    # games x depth tensors every time.
    path_nodes = torch.empty((games, MAX_DEPTH), dtype=torch.int64, device=device)
    path_actions = torch.empty_like(path_nodes)
    node = torch.empty_like(root)
    alive = torch.empty_like(playable)
    depth = torch.empty(games, dtype=torch.int64, device=device)
    leaf_value = torch.empty(games, device=device)
    expanding = torch.empty(games, dtype=torch.bool, device=device)
    for _ in range(sims):
        node.copy_(root)
        alive.copy_(playable)
        path_nodes.fill_(-1)
        path_actions.fill_(-1)
        depth.zero_()
        leaf_value.zero_()
        expanding.zero_()

        # --- descent: pure indexing over the tree, no environment steps ----
        for level in range(MAX_DEPTH):
            # Reading a CUDA bool into Python synchronizes the entire stream.
            # Up to three masked iterations are much cheaper than doing that
            # at every tree level.
            if level and level % 4 == 0 and not bool(alive.any()):
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

        # Still-alive games reached MAX_DEPTH through expanded, nonterminal
        # nodes. Bootstrap from the reached node, negated into its parent's
        # perspective; the ordinary alternating backup below does the rest.
        leaf_value = torch.where(alive, -forest.value[rows, node], leaf_value)

        # --- expansion: one environment step and one evaluation ------------
        # Batched actors almost always contain at least one expanding game.
        # Running this mask-first avoids two CUDA->host ``any()`` synchronizes
        # per simulation; inactive rows write only into their unused next slot.
        last = (depth - 1).clamp(min=0)
        parent = path_nodes[rows, last]
        action = path_actions[rows, last]
        leaf_board, outcome = step(forest.load(parent), action.clamp(min=0))
        terminal_now = expanding & (outcome != NOT_TERMINAL)
        index = (rows[terminal_now], parent[terminal_now], action[terminal_now])
        forest.edge_terminal[index] = outcome[terminal_now]
        leaf_value = torch.where(terminal_now, outcome.float(), leaf_value)

        fresh = expanding & (outcome == NOT_TERMINAL)
        # Occurrences of the leaf's position before this one: in the game's
        # history and among ancestors on the path. Computing for inactive rows
        # is cheap and keeps the simulation entirely device-driven.
        leaf_hash = leaf_board.position_hash(keys, side ^ (depth % 2 == 1))
        on_path = forest.hash[rows[:, None], path_nodes.clamp(min=0)]
        before = ((on_path == leaf_hash[:, None]) & (path_nodes >= 0)).sum(dim=1)
        if history is not None:
            before = before + history_counts(history, leaf_hash)
        repeated = fresh & (before >= 2)
        index = (rows[repeated], parent[repeated], action[repeated])
        forest.edge_terminal[index] = DRAW
        leaf_value = torch.where(repeated, torch.zeros_like(leaf_value), leaf_value)
        fresh = fresh & ~repeated

        # A fully-terminal simulation needs no network call. This is the one
        # expansion synchronization retained; the previous outer check and the
        # per-terminal checks are mask-only.
        if bool(fresh.any()):
            leaf_legal = leaf_board.legal()
            logits, wdl, q_logits = forward(net, leaf_board.planes(before >= 1, before >= 2),
                                            leaf_legal)
            distribution = torch.softmax(wdl.float(), dim=1)
            child_value = distribution[:, 2] - distribution[:, 0]
            new_index = forest.size.clamp(max=forest.capacity - 1)
            forest.child[rows[fresh], parent[fresh], action[fresh]] = new_index[fresh]
            forest.install(new_index, logits, leaf_legal, q_logits)
            forest.store(new_index, leaf_board)
            forest.hash[rows, new_index] = leaf_hash
            forest.value[rows[fresh], new_index[fresh]] = child_value[fresh]
            idle = ~fresh
            forest.legal[rows[idle], new_index[idle]] = False
            forest.prior[rows[idle], new_index[idle]] = 0.0
            forest.edge_value[rows[idle], new_index[idle]] = 0.0
            forest.size = torch.where(fresh, forest.size + 1, forest.size)
            leaf_value = torch.where(fresh, -child_value, leaf_value)

        # --- backup: deepest edge takes +leaf_value, alternating upward ----
        for level in range(MAX_DEPTH - 1, -1, -1):
            active = path_actions[:, level] >= 0
            parity = ((depth - 1 - level) & 1).float()
            sign = 1.0 - 2.0 * parity
            index = (rows[active], path_nodes[active, level], path_actions[active, level])
            forest.visits[index] += 1.0
            forest.value_sum[index] += (sign * leaf_value)[active]

    return forest


@torch.no_grad()
def search(net, forward, board: BoardBatch, rep1, rep2, sims: int,
           add_noise: bool = True, generator=None, side=None, history=None, keys=None):
    """Runs `sims` simulations from `board`; returns root visits and values.
    See search_tree() for the arguments."""
    forest = search_tree(net, forward, board, rep1, rep2, sims, add_noise, generator,
                         side, history, keys)
    return forest.visits[forest.rows, 0], forest.value_sum[forest.rows, 0]


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
