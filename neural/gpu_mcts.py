"""Batched PUCT search: one AlphaZero-style tree per game, all in tensors.

Thousands of games search in lockstep. A simulation descends every tree at
once over the visit statistics, expands one leaf per game and backs the
value up its path, so a simulation costs one batched network evaluation
and one environment step regardless of how deep the trees have grown.
Search depth therefore grows with the simulation count, which is what a
fixed-depth lookahead cannot do.

Each node keeps its board, so descending is pure indexing: the position of
a child is read, never replayed. Each node also keeps the hash of its
position, so the threefold rule holds inside the tree exactly as the game
applies it: a leaf whose position has already occurred twice - in the
game's own history or among the ancestors on its search path - is a
terminal draw, and a leaf whose position has occurred once is evaluated
with its repetition plane set.

The simulation step is written for CUDA graphs: every tensor it touches
has a fixed shape, every write is a masked write over the whole batch, and
nothing inside it reads a value back to the host. A batch is padded to one
of a few widths and searched in a static workspace, so a simulation replays
as a single graph launch instead of several hundred kernel launches issued
from Python. Measured on an H100, that launch overhead was a flat 20 ms per
simulation - half of an actor run - whatever the batch width. Without CUDA
the same function runs eagerly, which is what the tests exercise.

Values are always "for the player to move at this node"; an edge's value
is the negation of the value of the position it leads to.
"""

from __future__ import annotations

import math
import os
import time

import torch

from .gpu_env import ACTIONS, CANVAS, DRAW, NOT_TERMINAL, BoardBatch, hash_keys, step
from .gpu_history import DenseHistoryView, history_counts

C_PUCT = 1.5
# Unvisited children start from the Q head's expected value (win minus loss
# probability). MCTS_Q_SEED=0 starts them from zero instead, which measures
# what the head is worth to the search.
Q_SEED = os.environ.get("MCTS_Q_SEED", "1") != "0"
DIRICHLET_ALPHA = 0.4
DIRICHLET_FRACTION = 0.25
MAX_DEPTH = 64            # descent guard; trees are far shallower in practice
# A captured step runs a fixed number of descent and backup levels. The
# search starts at the smallest bound and moves up before the trees can
# outgrow it: depth grows by at most one per simulation, and the depth is
# read back once every CHECK_EVERY simulations.
LEVEL_STEPS = (8, 16, 32, MAX_DEPTH)
CHECK_EVERY = 8
HISTORY_CAPACITY = 224    # minimum workspace size; callers may supply longer eras
USE_GRAPHS = os.environ.get("SELFPLAY_GRAPHS", "1") != "0"

_SCALARS = ("rows", "cols", "connect", "chaos", "pieces")
_WORKSPACES = {}
_GRAPH_POOL = None
STATS = {"workspaces": 0, "captures": 0, "capture_seconds": 0.0}


def search_configuration():
    """Snapshot the Python settings whose values are baked into CUDA kernels."""
    if (isinstance(C_PUCT, bool) or not isinstance(C_PUCT, (int, float))
            or not math.isfinite(C_PUCT) or C_PUCT < 0):
        raise ValueError("C_PUCT must be finite and nonnegative")
    if not isinstance(Q_SEED, bool):
        raise ValueError("Q_SEED must be a boolean")
    return float(C_PUCT), Q_SEED


def bucket(width: int) -> int:
    """Padded batch width: a handful of shapes, so a run captures a handful
    of graphs, with little waste where the batch is wide and compute-bound."""
    if width <= 64:
        return 64
    if width <= 1024:
        return 1 << (width - 1).bit_length()
    return -(-width // 512) * 512


class Forest:
    """One tree per game. Edge statistics are [game, node, action]; each
    node also stores the position it stands for and that position's hash."""

    def __init__(self, games: int, sims: int, device, max_connect: int = 10, any_chaos=True,
                 settings=None):
        self.c_puct, self.q_seed = search_configuration() if settings is None else settings
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

    def reset(self):
        """Empties every tree in place; boards are overwritten as nodes are made."""
        self.child.fill_(-1)
        self.visits.zero_()
        self.value_sum.zero_()
        self.prior.zero_()
        self.legal.zero_()
        self.edge_terminal.fill_(NOT_TERMINAL)
        self.edge_value.zero_()
        self.size.fill_(1)
        self.hash.zero_()
        self.value.zero_()

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
        u = self.c_puct * self.prior[index] * total / (1.0 + visits)
        return (q + u).masked_fill(~self.legal[index], float("-inf"))

    def install(self, node, logits, legal, q_logits=None, keep=None):
        """Writes priors, legality and per-action values into a node. Rows
        outside `keep` are written as empty nodes (the slot stays unused)."""
        prior = torch.nan_to_num(torch.softmax(logits.masked_fill(~legal, float("-inf")), dim=1))
        if q_logits is None or not self.q_seed:
            expected = torch.zeros_like(prior)
        else:
            distribution = torch.softmax(q_logits.float(), dim=2)
            expected = torch.nan_to_num(distribution[:, :, 2] - distribution[:, :, 0])
        if keep is not None:
            mask = keep[:, None]
            prior = prior * mask
            legal = legal & mask
            expected = expected * mask
        self.prior[self.rows, node] = prior
        self.legal[self.rows, node] = legal
        self.edge_value[self.rows, node] = expected


class Workspace:
    """Static buffers for one padded batch width and simulation budget: the
    forest, the search inputs and the per-step scratch that a CUDA graph
    replays against. History capacity is part of the cache key, so growing
    an input never resizes storage underneath a captured graph."""

    def __init__(self, net, forward, games: int, sims: int, device, max_connect: int, any_chaos: bool,
                 history_capacity: int = HISTORY_CAPACITY, *, settings=None):
        if type(history_capacity) is not int or history_capacity < 1:
            raise ValueError("history capacity must be a positive integer")
        # The graphs bake in this network's weights and this forward, so the
        # workspace keeps both alive: a reused id could otherwise replay the
        # wrong network for a newcomer with the same key.
        self.net, self.forward = net, forward
        self.games, self.sims, self.device = games, sims, torch.device(device)
        self.max_depth = MAX_DEPTH
        self.settings = search_configuration() if settings is None else settings
        self.forest = Forest(games, sims, self.device, max_connect, any_chaos, self.settings)
        self.root = torch.zeros(games, dtype=torch.int64, device=self.device)
        self.playable = torch.zeros(games, dtype=torch.bool, device=self.device)
        self.side = torch.zeros(games, dtype=torch.bool, device=self.device)
        self.keys = torch.zeros((2, CANVAS, CANVAS), dtype=torch.int64, device=self.device)
        self.history = DenseHistoryView(
            torch.zeros((games, history_capacity), dtype=torch.int64, device=self.device),
            torch.zeros(games, dtype=torch.int64, device=self.device))
        self.path_nodes = torch.full((games, MAX_DEPTH), -1, dtype=torch.int64, device=self.device)
        self.path_actions = torch.full((games, MAX_DEPTH), -1, dtype=torch.int64, device=self.device)
        self.depth_max = torch.zeros(games, dtype=torch.int64, device=self.device)
        self.graphs = {}

    def reset(self):
        self.forest.reset()
        self.depth_max.zero_()


def _simulate(ws: Workspace, net, forward, levels: int):
    """One simulation for every tree in the workspace: descend `levels`
    levels at most, expand, back up. Fixed shapes, masked writes, no host
    reads - the body a CUDA graph captures."""
    forest, rows = ws.forest, ws.forest.rows
    node = ws.root.clone()
    alive = ws.playable.clone()
    path_nodes, path_actions = ws.path_nodes, ws.path_actions
    path_nodes.fill_(-1)
    path_actions.fill_(-1)
    depth = torch.zeros_like(node)
    leaf_value = torch.zeros(ws.games, device=ws.device)
    expanding = torch.zeros_like(alive)

    # --- descent: pure indexing over the tree, no environment steps --------
    for level in range(levels):
        action = forest.puct(node).argmax(dim=1)
        path_nodes[:, level] = torch.where(alive, node, path_nodes[:, level])
        path_actions[:, level] = torch.where(alive, action, path_actions[:, level])
        depth = depth + alive.long()
        child = forest.child[rows, node, action]
        terminal = forest.edge_terminal[rows, node, action]
        hit_terminal = alive & (terminal != NOT_TERMINAL)
        leaf_value = torch.where(hit_terminal, terminal.float(), leaf_value)
        expanding = expanding | (alive & ~hit_terminal & (child < 0))
        descend = alive & ~hit_terminal & (child >= 0)
        node = torch.where(descend, child, node)
        alive = descend
    # A game still descending after `levels` levels stops at an expanded,
    # nonterminal node: bootstrap from that node's cached value, negated into
    # its parent's perspective. The driver raises `levels` before this binds.
    leaf_value = torch.where(alive, -forest.value[rows, node], leaf_value)
    torch.maximum(ws.depth_max, depth, out=ws.depth_max)

    # --- expansion: one environment step and one evaluation ----------------
    last = (depth - 1).clamp(min=0)
    parent = path_nodes[rows, last].clamp(min=0)
    action = path_actions[rows, last].clamp(min=0)
    leaf_board, outcome = step(forest.load(parent), action)
    terminal_now = expanding & (outcome != NOT_TERMINAL)
    fresh = expanding & (outcome == NOT_TERMINAL)
    # Occurrences of the leaf's position before this one: in the game's
    # history and among the ancestors on the path (root at level 0 up to the
    # parent). Same stones, same shape, same side to move - the actor's count.
    leaf_hash = leaf_board.position_hash(ws.keys, ws.side ^ (depth % 2 == 1))
    on_path = forest.hash[rows[:, None], path_nodes.clamp(min=0)]
    before = ((on_path == leaf_hash[:, None]) & (path_nodes >= 0)).sum(dim=1)
    before = before + history_counts(ws.history, leaf_hash)
    repeated = fresh & (before >= 2)          # third occurrence: a draw by rule
    fresh = fresh & ~repeated

    edge = forest.edge_terminal[rows, parent, action]
    edge = torch.where(terminal_now, outcome, edge)
    edge = torch.where(repeated, torch.full_like(edge, DRAW), edge)
    forest.edge_terminal[rows, parent, action] = edge
    leaf_value = torch.where(terminal_now, outcome.float(), leaf_value)
    leaf_value = torch.where(repeated, torch.zeros_like(leaf_value), leaf_value)

    # Every row is evaluated; a row that did not expand writes an empty node
    # into its next unused slot, which the next expansion overwrites.
    leaf_legal = leaf_board.legal() & fresh[:, None]
    logits, wdl, q_logits = forward(net, leaf_board.planes(before >= 1, before >= 2), leaf_legal)
    distribution = torch.softmax(wdl.float(), dim=1)
    child_value = distribution[:, 2] - distribution[:, 0]
    new_index = forest.size.clamp(max=forest.capacity - 1)
    forest.child[rows, parent, action] = torch.where(fresh, new_index,
                                                     forest.child[rows, parent, action])
    forest.install(new_index, logits, leaf_legal, q_logits, keep=fresh)
    forest.store(new_index, leaf_board)
    forest.hash[rows, new_index] = leaf_hash
    forest.value[rows, new_index] = child_value * fresh
    forest.size.add_(fresh.long())
    # The leaf's value is for its own mover; its parent edge negates it.
    leaf_value = torch.where(fresh, -child_value, leaf_value)

    # --- backup: deepest edge takes +leaf_value, alternating upward --------
    for level in range(levels - 1, -1, -1):
        action_l = path_actions[:, level]
        active = (action_l >= 0).float()
        sign = 1.0 - 2.0 * ((depth - 1 - level) & 1).float()
        index = (rows, path_nodes[:, level].clamp(min=0), action_l.clamp(min=0))
        forest.visits[index] += active
        forest.value_sum[index] += sign * leaf_value * active


def level_schedule(max_depth: int):
    """The descent bounds a search steps through, ending at the depth guard."""
    return tuple(sorted({min(levels, max_depth) for levels in LEVEL_STEPS} | {max_depth}))


def _capture(ws: Workspace, net, forward):
    """Warms the workspace up once, then captures one graph per level bound.
    The forest is scratch during capture; callers reset it afterwards."""
    global _GRAPH_POOL
    started = time.time()
    schedule = level_schedule(ws.max_depth)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _simulate(ws, net, forward, schedule[0])
    torch.cuda.current_stream().wait_stream(stream)
    for levels in schedule:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=_GRAPH_POOL):
            _simulate(ws, net, forward, levels)
        if _GRAPH_POOL is None:
            # Every graph writes its results into the workspace, never into
            # pool memory, so all of them can share one pool of scratch.
            _GRAPH_POOL = graph.pool()
        ws.graphs[levels] = graph
        STATS["captures"] += 1
    torch.cuda.synchronize()
    STATS["capture_seconds"] += time.time() - started


def workspace(net, forward, games: int, sims: int, device, max_connect: int, any_chaos: bool,
              history_capacity: int = HISTORY_CAPACITY):
    """The cached workspace for a network and batch shape, captured on first use."""
    if type(history_capacity) is not int or history_capacity < 1:
        raise ValueError("history capacity must be a positive integer")
    settings = search_configuration()
    # Python scalars and branches are fixed at capture. A changed setting
    # must never reuse an older graph; eager forests freeze the same values.
    key = (id(net), id(forward), games, sims, str(torch.device(device)), max_connect, any_chaos,
           MAX_DEPTH, history_capacity, settings, bool(USE_GRAPHS))
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = Workspace(net, forward, games, sims, device, max_connect, any_chaos, history_capacity,
                       settings=settings)
        STATS["workspaces"] += 1
        if USE_GRAPHS and ws.device.type == "cuda":
            _capture(ws, net, forward)
        _WORKSPACES[key] = ws
    return ws


def _run(ws: Workspace, net, forward, sims: int):
    schedule = level_schedule(ws.max_depth)
    position = 0
    levels = schedule[position]
    graphs = ws.graphs if USE_GRAPHS and ws.device.type == "cuda" else None
    for i in range(sims):
        if i and i % CHECK_EVERY == 0:
            reached = int(ws.depth_max.max())   # the one host read per CHECK_EVERY steps
            while position + 1 < len(schedule) and reached + CHECK_EVERY > levels:
                position += 1
                levels = schedule[position]
        if graphs is not None:
            graphs[levels].replay()
        else:
            _simulate(ws, net, forward, levels)


def _prepare_history(history, width: int, device):
    """Validate inputs before capture; expand legacy counts without dropping entries.

    Dense callers keep their fixed-width buffer on-device. Its shape, not a
    per-ply maximum read back from the GPU, determines the workspace capacity.
    Legacy packed inputs already require host conversion and are expanded once.
    """
    if history is None:
        return None
    if not isinstance(history, DenseHistoryView):
        hashes, counts = history
        if (hashes.ndim != 2 or hashes.shape[0] != width or counts.shape != hashes.shape
                or hashes.dtype != torch.int64 or counts.dtype != torch.int64):
            raise ValueError("packed history must contain matching int64 [game, slot] tensors")
        rows = []
        for row_hashes, row_counts in zip(hashes.tolist(), counts.tolist()):
            if any(count < 0 for count in row_counts):
                raise ValueError("history counts must be nonnegative")
            rows.append([key for key, count in zip(row_hashes, row_counts) for _ in range(count)])
        columns = max((len(row) for row in rows), default=0)
        history = DenseHistoryView(
            torch.tensor([row + [0] * (columns - len(row)) for row in rows],
                         dtype=torch.int64, device=device).reshape(width, columns),
            torch.tensor([len(row) for row in rows], dtype=torch.int64, device=device))
    if (history.hashes.ndim != 2 or history.hashes.shape[0] != width
            or history.lengths.shape != (width,) or history.hashes.dtype != torch.int64
            or history.lengths.dtype != torch.int64):
        raise ValueError("dense history requires int64 hashes [game, slot] and lengths [game]")
    if bool(((history.lengths < 0) | (history.lengths > history.hashes.shape[1])).any()):
        raise ValueError("history length exceeds its supplied buffer")
    return history


def _load_history(ws: Workspace, history, width: int):
    """Copy a prepared dense history in full, keeping captured tensors stable."""
    ws.history.lengths.zero_()
    if history is None:
        return
    columns = history.hashes.shape[1]
    if columns > ws.history.hashes.shape[1]:
        raise ValueError("history exceeds workspace capacity; allocate a larger workspace")
    ws.history.hashes[:width, :columns] = history.hashes
    ws.history.lengths[:width] = history.lengths


def pack_history(eras, device):
    """Pads per-game {hash: count} dicts of the positions that could still
    recur into a (hashes, counts) pair search() accepts; None when no game
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
    mover-relative. `history` holds the positions each game has seen that
    could recur (a DenseHistoryView, or pack_history()'s pair), hashed with
    `keys` from hash_keys(); the search then counts occurrences the way the
    actor does. Without them, repetition is still tracked along the path.

    The returned forest is the padded workspace: rows beyond len(board)
    are dummies, and the next search on the same shape overwrites it.
    """
    width, device = len(board), torch.device(board.device)
    games = bucket(width)
    history = _prepare_history(history, width, device)
    history_capacity = max(HISTORY_CAPACITY, 0 if history is None else history.hashes.shape[1])
    ws = workspace(net, forward, games, sims, device, getattr(board, "max_connect", 10),
                   getattr(board, "any_chaos", True), history_capacity)
    ws.reset()
    forest, rows = ws.forest, ws.forest.rows
    padded = board if games == width else board.padded(games)
    pad = torch.zeros(games - width, dtype=torch.bool, device=device)
    rep1 = torch.cat([rep1.to(device=device, dtype=torch.bool), pad])
    rep2 = torch.cat([rep2.to(device=device, dtype=torch.bool), pad])
    ws.keys.copy_(hash_keys(device) if keys is None else keys)
    ws.side.zero_()
    if side is not None:
        ws.side[:width] = torch.as_tensor(side, dtype=torch.bool, device=device)
    _load_history(ws, history, width)

    root_legal = padded.legal()
    logits, root_wdl, q_logits = forward(net, padded.planes(rep1, rep2), root_legal)
    forest.install(ws.root, logits, root_legal, q_logits)
    root_distribution = torch.softmax(root_wdl.float(), dim=1)
    forest.value[rows, 0] = root_distribution[:, 2] - root_distribution[:, 0]
    # Kept for the Gumbel policy target: the prior before exploration noise
    # and the network's own value of the root position.
    forest.root_prior = forest.prior[rows, 0].clone()
    forest.root_net_value = forest.value[rows, 0].clone()
    forest.store(ws.root, padded)
    forest.hash[rows, 0] = padded.position_hash(ws.keys, ws.side)
    if add_noise:
        noise = torch.distributions.Dirichlet(
            torch.full((ACTIONS,), DIRICHLET_ALPHA, device=device)).sample((games,))
        noise = noise * root_legal
        noise = noise / noise.sum(dim=1, keepdim=True).clamp(min=1e-9)
        forest.prior[rows, 0] = ((1 - DIRICHLET_FRACTION) * forest.prior[rows, 0]
                                 + DIRICHLET_FRACTION * noise)
    ws.playable.copy_(root_legal.any(dim=1))
    _run(ws, net, forward, sims)
    return forest


@torch.no_grad()
def search(net, forward, board: BoardBatch, rep1, rep2, sims: int,
           add_noise: bool = True, generator=None, side=None, history=None, keys=None):
    """Runs `sims` simulations from `board`; returns root visits and values
    for the games in `board`. See search_tree() for the arguments."""
    forest = search_tree(net, forward, board, rep1, rep2, sims, add_noise, generator,
                         side, history, keys)
    width = len(board)
    return forest.visits[:width, 0].clone(), forest.value_sum[:width, 0].clone()


@torch.no_grad()
def search_root(net, forward, board: BoardBatch, rep1, rep2, sims: int,
                add_noise: bool = True, generator=None, side=None, history=None, keys=None):
    """search(), plus the root's prior before exploration noise and the
    network's own value of the root, which improved_policy() needs."""
    forest = search_tree(net, forward, board, rep1, rep2, sims, add_noise, generator,
                         side, history, keys)
    width = len(board)
    return (forest.visits[:width, 0].clone(), forest.value_sum[:width, 0].clone(),
            forest.root_prior[:width].clone(), forest.root_net_value[:width].clone())


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


def improved_policy(prior, visits, value_sum, net_value, legal, c_visit: float = 50.0,
                    c_scale: float = 1.0):
    """Gumbel MuZero's policy improvement from a root search of any size.

    Visit counts are a poor target at small budgets: a handful of visits
    say little about the moves the search never tried. The improved policy
    is softmax(logits + sigma(completed Q)) instead, where every visited
    action contributes its search value, every unvisited one the mixed
    estimate v_mix of the root, and sigma scales values (normalised from
    [-1, 1] to [0, 1]) by (c_visit + max visits) * c_scale. It is an
    improvement on the prior in expectation and gives a usable target even
    from the plies that are searched with a few dozen simulations.
    """
    prior = prior.masked_fill(~legal, 0.0)
    visited = visits > 0
    q = torch.where(visited, value_sum / visits.clamp(min=1), torch.zeros_like(value_sum))
    total = visits.sum(dim=1, keepdim=True)
    weight = (prior * visited).sum(dim=1, keepdim=True)
    q_pi = (prior * q * visited).sum(dim=1, keepdim=True) / weight.clamp(min=1e-9)
    v_mix = torch.where(weight > 0, (net_value[:, None] + total * q_pi) / (1 + total),
                        net_value[:, None])
    completed = torch.where(visited, q, v_mix.expand_as(q))
    normalised = (completed + 1) / 2
    sigma = (c_visit + visits.max(dim=1, keepdim=True).values) * c_scale * normalised
    logits = torch.log(prior.clamp(min=1e-9)) + sigma
    logits = logits.masked_fill(~legal, float("-inf"))
    return torch.softmax(logits, dim=1)


def sample_actions(policy, greedy, generator=None):
    """Samples from the visit distribution, or takes its argmax."""
    picked = torch.multinomial(policy.clamp(min=0) + 1e-12, 1, generator=generator).squeeze(1)
    return torch.where(greedy, policy.argmax(dim=1), picked)
