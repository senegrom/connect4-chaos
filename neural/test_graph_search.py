"""GPU check for the graph-replayed search: it must equal the eager search
exactly, and searching a board inside a wider padded batch must not change
its result. Also times both paths. Last, the fused network that the actors
and the arena run must be the network the learner trains, bf16 noise apart:
checked after the search, so that a factor still to be calibrated cannot
hide the graph and padding results.

Usage: python -m neural.test_graph_search <model.pt>   (skips without CUDA)
Run it on Modal with:
  modal run neural/modal_app.py --task gpu-test --module test_graph_search --args models/<name>.pt
"""

from __future__ import annotations

import sys
import time

import torch

from . import gpu_mcts
from .gpu_env import BoardBatch, hash_keys
from .gpu_history import DenseHistory, DenseHistoryView
from .gpu_selfplay import MAX_PLIES, _prepare_network, all_shapes, forward
from .model import FusedInferenceNet, PolicyValueNet, checkpoint_arch, fold_batchnorm
from .test_gpu_mcts import play


def probabilities(outputs):
    """Each head's probabilities: the policy over the legal actions, W/D/L,
    and W/D/L after each action."""
    logits, wdl, q = (output.float() for output in outputs)
    return torch.softmax(logits, dim=1), torch.softmax(wdl, dim=1), torch.softmax(q, dim=2)


@torch.no_grad()
def check_fused(payload, planes, legal):
    """FusedInferenceNet calls private cuDNN ops with positional arguments. A
    dropped residual or bias, or a torch release that changes those ops,
    would have every actor and arena play a different function from the one
    the learner trains and exports. Against the fp32 network, on average it
    may be off by at most twice what the folded network under bf16 autocast
    is off by (SELFPLAY_FUSED=0), and it must pick the same move wherever the
    fp32 choice is clear of that noise. Both factors are first guesses, to be
    calibrated on the figures the first GPU run prints."""
    exact = PolicyValueNet(*checkpoint_arch(payload)).to(planes.device)
    exact.load_state_dict(payload["model"])
    exact.eval()
    # cuDNN convolutions run in TF32 by default, which is not fp32.
    tf32, torch.backends.cudnn.allow_tf32 = torch.backends.cudnn.allow_tf32, False
    try:
        reference = probabilities(exact(planes, legal))
    finally:
        torch.backends.cudnn.allow_tf32 = tf32
    fused = probabilities(forward(FusedInferenceNet(exact), planes, legal))
    folded = fold_batchnorm(exact).to(memory_format=torch.channels_last)
    autocast = probabilities(forward(folded, planes, legal))
    for name, want, got, bf16 in zip(("policy", "value", "q"), reference, fused, autocast):
        error, noise = (got - want).abs(), (bf16 - want).abs()
        print(f"fused {name}: mean |dp| {float(error.mean()):.2e}, max {float(error.max()):.2e}; "
              f"autocast {float(noise.mean()):.2e}, max {float(noise.max()):.2e}", flush=True)
        assert error.mean() <= 2 * noise.mean(), f"the fused {name} head is not the network's"
    # Clear: the fp32 move leads the next by more than four times the worst
    # autocast deviation, so a fused deviation of up to twice the worst
    # autocast one on each of the two moves cannot swap them.
    top = reference[0].topk(2, dim=1).values
    clear = top[:, 0] - top[:, 1] > 4 * (autocast[0] - reference[0]).abs().max()
    same = fused[0].argmax(dim=1) == reference[0].argmax(dim=1)
    print(f"fused policy: the fp32 move on {float(same[clear].float().mean()):.1%} of the "
          f"{int(clear.sum())} boards with a clear one, {float(same.float().mean()):.1%} of all "
          f"{len(same)}", flush=True)
    assert bool(clear.any()), "no board has a move clear of the bf16 noise: recalibrate the margin"
    assert bool(same[clear].all()), "the fused network picks other moves"


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
    check_fused(payload, board.planes(rep1, rep2), board.legal())
    print("GRAPH SEARCH OK")


if __name__ == "__main__":
    main()
