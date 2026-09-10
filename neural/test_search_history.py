"""Full repetition history reaches the real CPU/CUDA search without truncation.

Run: python -m neural.test_search_history
No checkpoint is needed: only network logits are deterministic test outputs.
The board transitions, hashing, history and MCTS are the production code.
"""
from collections import Counter
import json
from pathlib import Path
import unittest

import torch

from . import gpu_mcts
from .gpu_env import ACTIONS, FLIP, BoardBatch, DRAW, NOT_TERMINAL, hash_keys, step
from .gpu_history import DenseHistory, DenseHistoryView, history_counts

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/long-transform-era.json"


def flip_forward(_net, planes, _legal):
    logits = torch.zeros((len(planes), ACTIONS), device=planes.device)
    logits[:, FLIP] = 20
    return (logits, torch.zeros((len(planes), 3), device=planes.device),
            torch.zeros((len(planes), ACTIONS, 3), device=planes.device))


def replay(device="cpu"):
    data = json.loads(FIXTURE.read_text())
    board = BoardBatch([data["rows"]], [data["cols"]], [data["connect"]], [True], device)
    generator = torch.Generator(device=device).manual_seed(173)
    keys = hash_keys(device, generator)
    history = DenseHistory(1, 301, device)
    ids = torch.tensor([0], device=device)
    actions = data["drops"] + [10 + value for value in data["transforms"]]
    side = False
    for ply, value in enumerate(actions):
        action = torch.tensor([value], device=device)
        assert bool(board.legal()[0, value]), f"illegal fixture move at {ply}"
        history.append_or_reset(ids, board.position_hash(keys, side), action < 10)
        board, outcome = step(board, action)
        side = not side
        assert int(outcome[0]) == NOT_TERMINAL, f"fixture ended at {ply}"
        assert int(history.counts(ids, board.position_hash(keys, side))[0]) < 2, f"earlier draw at {ply}"
    absolute = torch.where(board.mover[0], data["mover"],
                           torch.where(board.opponent[0], 3 - data["mover"], 0))
    assert absolute.flip(0).cpu().tolist() == data["board"]
    assert (2 if side else 1) == data["mover"]
    assert int(history.lengths[0]) == 225 and len(actions) == 245
    return board, history.search_view(ids), keys, side


def run_search(board, history, keys, side, sims=1):
    counts = history_counts(history, board.position_hash(keys, side))
    return gpu_mcts.search_tree(None, flip_forward, board, counts >= 1, counts >= 2,
                                sims, add_noise=False, side=side, history=history, keys=keys)


class SearchHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.board, cls.history, cls.keys, cls.side = replay()

    def tearDown(self):
        gpu_mcts._WORKSPACES.clear()

    def assert_flip_draw(self, history):
        child, outcome = step(self.board, torch.tensor([FLIP]))
        self.assertEqual(int(outcome[0]), NOT_TERMINAL)
        self.assertEqual(int(history_counts(history, child.position_hash(self.keys, not self.side))[0]), 2)
        forest = run_search(self.board, history, self.keys, self.side)
        self.assertEqual(int(forest.edge_terminal[0, 0, FLIP]), DRAW)
        self.assertEqual(int(forest.child[0, 0, FLIP]), -1)
        self.assertEqual(float(forest.visits[0, 0].sum()), 1)
        self.assertEqual(float(forest.value_sum[0, 0, FLIP]), 0)

    def test_legal_long_era_is_a_terminal_draw_in_search(self):
        self.assert_flip_draw(self.history)

    def test_legacy_packed_long_era_is_a_terminal_draw(self):
        # Preserve slot order as well as multiplicity to exercise the old cutoff.
        hashes = self.history.hashes[:, :225].clone()
        self.assert_flip_draw((hashes, torch.ones_like(hashes)))
        counted = Counter(hashes[0].tolist())
        self.assert_flip_draw(gpu_mcts.pack_history([counted], "cpu"))

    def test_capacity_boundaries_and_expanded_counts(self):
        for length in (0, 1, 223, 224, 225, 300, 301, 512):
            with self.subTest(length=length):
                history = DenseHistoryView(torch.arange(length).reshape(1, length), torch.tensor([length]))
                prepared = gpu_mcts._prepare_history(history, 1, "cpu")
                ws = gpu_mcts.Workspace(None, flip_forward, 2, 1, "cpu", 5, True,
                                        history_capacity=max(224, length))
                gpu_mcts._load_history(ws, prepared, 1)
                self.assertTrue(torch.equal(ws.history.hashes[0, :length], history.hashes[0]))
                self.assertEqual(ws.history.lengths.tolist(), [length, 0])
                packed = (torch.tensor([[123]]), torch.tensor([[length]]))
                expanded = gpu_mcts._prepare_history(packed, 1, "cpu")
                gpu_mcts._load_history(ws, expanded, 1)
                self.assertEqual(int(history_counts(ws.history, torch.tensor([123, 123]))[0]), length)

    def test_shorter_and_absent_histories_clear_reused_rows(self):
        ws = gpu_mcts.Workspace(None, flip_forward, 3, 1, "cpu", 5, True, 301)
        long = DenseHistoryView(torch.full((3, 301), 17), torch.tensor([301, 301, 301]))
        gpu_mcts._load_history(ws, long, 3)
        pointers = (ws.history.hashes.data_ptr(), ws.history.lengths.data_ptr())
        shorter = DenseHistoryView(torch.tensor([[17, 4]]), torch.tensor([1]))
        gpu_mcts._load_history(ws, shorter, 1)
        self.assertEqual(history_counts(ws.history, torch.tensor([17, 17, 17])).tolist(), [1, 0, 0])
        gpu_mcts._load_history(ws, None, 1)
        self.assertEqual(ws.history.lengths.tolist(), [0, 0, 0])
        self.assertEqual((ws.history.hashes.data_ptr(), ws.history.lengths.data_ptr()), pointers)

    def test_capacity_is_part_of_the_workspace_cache_key(self):
        args = (None, flip_forward, 64, 1, "cpu", 5, True)
        small = gpu_mcts.workspace(*args)
        large = gpu_mcts.workspace(*args, history_capacity=301)
        self.assertIsNot(small, large)
        self.assertIs(small, gpu_mcts.workspace(*args))
        self.assertIs(large, gpu_mcts.workspace(*args, history_capacity=301))
        self.assertEqual(small.history.hashes.shape[1], 224)
        self.assertEqual(large.history.hashes.shape[1], 301)

    def test_undersized_workspace_fails_closed(self):
        ws = gpu_mcts.Workspace(None, flip_forward, 1, 1, "cpu", 5, True)
        with self.assertRaisesRegex(ValueError, "workspace capacity"):
            gpu_mcts._load_history(ws, self.history, 1)

    def test_invalid_lengths_counts_and_shapes_are_rejected(self):
        for length in (-1, 226):
            with self.subTest(length=length), self.assertRaises(ValueError):
                gpu_mcts._prepare_history(DenseHistoryView(torch.zeros((1, 225), dtype=torch.int64),
                                                          torch.tensor([length])), 1, "cpu")
        invalid = [
            (torch.tensor([[1]]), torch.tensor([[-1]])),
            (torch.tensor([[1]]), torch.tensor([[1.5]])),
            (torch.tensor([[1, 2]]), torch.tensor([[1]])),
            DenseHistoryView(torch.zeros((1, 2)), torch.tensor([2])),
            DenseHistoryView(torch.zeros((2, 2), dtype=torch.int64), torch.tensor([2])),
        ]
        for history in invalid:
            with self.subTest(history=history), self.assertRaises(ValueError):
                gpu_mcts._prepare_history(history, 1, "cpu")

    def test_no_history_still_explores_the_flip(self):
        forest = run_search(self.board, None, self.keys, self.side)
        self.assertEqual(int(forest.edge_terminal[0, 0, FLIP]), NOT_TERMINAL)
        self.assertGreaterEqual(int(forest.child[0, 0, FLIP]), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
    def test_cuda_graph_and_eager_search_agree_on_long_history(self):
        board, history, keys, side = replay("cuda")
        original = gpu_mcts.USE_GRAPHS
        results = []
        try:
            for graphs in (False, True):
                gpu_mcts._WORKSPACES.clear()
                gpu_mcts.USE_GRAPHS = graphs
                forest = run_search(board, history, keys, side, sims=4)
                results.append((forest.edge_terminal[0, 0].clone(), forest.visits[0, 0].clone()))
                self.assertEqual(int(forest.edge_terminal[0, 0, FLIP]), DRAW)
            for eager, captured in zip(*results):
                self.assertTrue(torch.equal(eager, captured))
        finally:
            gpu_mcts.USE_GRAPHS = original


if __name__ == "__main__":
    unittest.main()
