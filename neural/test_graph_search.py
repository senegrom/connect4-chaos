"""GPU check for the graph-replayed search: it must equal the eager search
exactly, and searching a board inside a wider padded batch must not change
its result. Also times both paths.

Usage: python -m neural.test_graph_search <model.pt>   (skips without CUDA)
Run it on Modal with: modal run neural/modal_app.py --task gpu-test
"""

from __future__ import annotations

import sys
import time

import torch

from . import gpu_mcts
from .gpu_env import BoardBatch, hash_keys
from .gpu_history import DenseHistory, DenseHistoryView
from .gpu_selfplay import MAX_PLIES, _prepare_network, all_shapes, forward
from .test_gpu_mcts import play


def timed_search(net, board, rep1, rep2, sims, view, keys):
    torch.cuda.synchronize()
    started = time.time()
    visits, values = gpu_mcts.search(net, forward, board, rep1, rep2, sims, add_noise=False,
                                     side=False, history=view, keys=keys)
    torch.cuda.synchronize()
    return visits, values, time.time() - started


def main():
    if not torch.cuda.is_available():
        print("no CUDA device; graph search test skipped")
        return
    device = "cuda"
    payload = torch.load(sys.argv[1], map_location=device, weights_only=True)
    net = _prepare_network(payload, device)
    shapes = all_shapes()
    picks = [shapes[i % len(shapes)] for i in range(700)]           # pads to 1024
    board = BoardBatch([p[0] for p in picks], [p[1] for p in picks],
                       [p[2] for p in picks], [p[3] for p in picks], device)
    board = play(board, [0, 1, 0])
    n = len(board)
    keys = hash_keys(device, generator=torch.Generator(device=device).manual_seed(11))
    games = torch.arange(n, device=device)
    history = DenseHistory(n, MAX_PLIES + 1, device)
    # Every game has seen its own root once, so the repetition machinery is live.
    history.append_or_reset(games, board.position_hash(keys, False),
                            torch.zeros(n, dtype=torch.bool, device=device))
    view = history.search_view(games)
    rep1 = torch.ones(n, dtype=torch.bool, device=device)
    rep2 = torch.zeros(n, dtype=torch.bool, device=device)

    graphed = {}
    for sims in (32, 256):
        gpu_mcts.USE_GRAPHS = True
        timed_search(net, board, rep1, rep2, sims, view, keys)      # captures, warms up
        visits_g, values_g, seconds_g = timed_search(net, board, rep1, rep2, sims, view, keys)
        gpu_mcts.USE_GRAPHS = False
        visits_e, values_e, seconds_e = timed_search(net, board, rep1, rep2, sims, view, keys)
        gpu_mcts.USE_GRAPHS = True
        identical = torch.equal(visits_g, visits_e) and torch.allclose(values_g, values_e, atol=1e-4)
        print(f"sims {sims:3d}: graphs {seconds_g:.2f}s, eager {seconds_e:.2f}s "
              f"({seconds_e / max(seconds_g, 1e-9):.1f}x); visits identical: {identical}", flush=True)
        assert identical, "graph replay diverged from the eager search"
        assert bool((visits_g.sum(dim=1) == sims).all()), "every simulation must land on the root"
        graphed[sims] = visits_g

    # The same 100 boards alone (bucket 128) and inside the 700 batch (bucket
    # 1024). bf16 kernels may be chosen per batch size, so a few rows may
    # differ by a visit; the padding itself must not change anything.
    subset = torch.arange(100, device=device)
    small = board.select(subset)
    small_view = DenseHistoryView(view.hashes[:100], view.lengths[:100])
    visits_s, _values, _seconds = timed_search(net, small, rep1[:100], rep2[:100], 32, small_view, keys)
    same = float((visits_s == graphed[32][:100]).all(dim=1).float().mean())
    print(f"padding: {same:.1%} of 100 boards identical between bucket 128 and bucket 1024", flush=True)
    assert same > 0.9, "padding changed the search for too many boards"
    stats = gpu_mcts.STATS
    print(f"{stats['workspaces']} workspaces, {stats['captures']} graphs, "
          f"{stats['capture_seconds']:.1f}s capturing")
    print("GRAPH SEARCH OK")


if __name__ == "__main__":
    main()
