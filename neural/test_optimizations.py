"""CPU regressions for the training throughput/compactness optimizations."""
from __future__ import annotations

import inspect
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import build_dataset, distill, gpu_mcts
from .chaos_game import empty_state, successors, NOT_TERMINAL
from .data_split import SPLIT_VERSION
from .gpu_env import BoardBatch
from .gpu_history import DenseHistory, history_counts
from .gpu_selfplay import _finish_shard
from .test_review import shard

ROOT = Path(__file__).resolve().parents[1]


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_dense_history_counts_resets_and_compacts_search_view(self):
        history = DenseHistory(3, 12, 'cpu')
        games = torch.tensor([0, 1, 2])
        hashes = torch.tensor([11, 21, 31])
        history.append_or_reset(games, hashes, torch.tensor([False, False, True]))
        self.assertEqual(history.counts(games, hashes).tolist(), [1, 1, 0])
        history.append_or_reset(games, torch.tensor([12, 22, 32]),
                                torch.tensor([False, True, False]))
        self.assertEqual(history.lengths.tolist(), [2, 0, 1])
        view = history.search_view(games)
        self.assertEqual(tuple(view.hashes.shape), (3, 2))
        self.assertEqual(history_counts(view, torch.tensor([11, 21, 32])).tolist(), [1, 0, 1])

    def test_dense_history_matches_legacy_packed_counts(self):
        history = DenseHistory(2, 8, 'cpu')
        games = torch.tensor([0, 1])
        for hashes in (torch.tensor([5, 7]), torch.tensor([5, 8]), torch.tensor([6, 7])):
            history.append_or_reset(games, hashes, torch.zeros(2, dtype=torch.bool))
        query = torch.tensor([5, 7])
        dense = history_counts(history.search_view(games), query)
        packed_hashes = torch.tensor([[5, 6, 0], [7, 8, 0]])
        packed_counts = torch.tensor([[2, 1, 0], [2, 1, 0]])
        legacy = history_counts((packed_hashes, packed_counts), query)
        self.assertEqual(dense.tolist(), [2, 2])
        self.assertEqual(legacy.tolist(), [2, 2])

    def test_selfplay_shard_is_compact_implicit_q_and_discards_caps(self):
        plies, games = 4, 3
        planes = torch.arange(plies * games * 7 * 10 * 10, dtype=torch.int64).reshape(
            plies, games, 7, 10, 10).remainder(11).to(torch.uint8)
        legal = torch.ones(plies, games, 13, dtype=torch.bool)
        policy = torch.full((plies, games, 13), 1 / 13)
        valid = torch.ones(plies, games, dtype=torch.bool)
        outcomes = torch.tensor([1, -1, 9])
        end_ply = torch.tensor([2, 3, -1])
        result, capped, positions = _finish_shard(planes, legal, policy, valid, outcomes, end_ply)
        self.assertEqual(capped, 1)
        self.assertEqual(positions, 8)  # 4 recorded plies for each completed game in this fixture
        self.assertEqual(result['planes'].dtype, torch.uint8)
        self.assertEqual(result['wdl'].dtype, torch.uint8)
        self.assertEqual(result['q_default'], 3)
        self.assertNotIn('q', result)
        self.assertEqual(len(result['wdl']), positions)

    def test_replay_precomputed_partition_skips_rehash(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,
                DISTILL_HOLDOUT_CONFIGS='', DISTILL_REPLAY_WINDOW='20'):
            root = Path(temp)
            replay = shard([(5, 5, 4, False)] * 20, replay=True, scaled=True)
            replay.pop('q')
            replay['q_default'] = 3
            replay['split_version'] = SPLIT_VERSION
            replay['validation'] = torch.zeros(20, dtype=torch.bool)
            torch.save(replay, root/'gpu-sp-optimized.pt')
            with patch('neural.distill.validation_mask', side_effect=AssertionError('rehash')):
                train, held = distill.load_shards(root)
            self.assertFalse(held)
            self.assertEqual(sum(len(s['wdl']) for s in train), 20)
            self.assertTrue(all('q' not in s and s.get('q_default') == 3 for s in train))

    def test_current_exact_partition_is_trusted_and_compact_builder_output(self):
        states = []
        rng = random.Random(903)
        state = empty_state(4, 5)
        while len(states) < 100:
            edges = [edge for edge in successors(state, 4, True) if edge.terminal == NOT_TERMINAL]
            if not edges:
                state = empty_state(4, 5)
                continue
            state = rng.choice(edges).child
            states.append(state)
        class Table:
            def __init__(self, *_args, **_kwargs): pass
            def sample_state(self, rng): return rng.choice(states), 0
            def edge_value_for_mover(self, _edge): return 0
        with tempfile.TemporaryDirectory() as temp, patch.object(build_dataset, 'PairTable', Table), \
                patch.object(build_dataset, 'SHARD', 16):
            root = Path(temp)
            build_dataset.build(root, 32, 'fixture:4:5:4:chaos', seed=5)
            held = torch.load(root/'4x5c4chaos-0000.pt', weights_only=True)
            train = torch.load(root/'4x5c4chaos-0001.pt', weights_only=True)
            for saved in (held, train):
                self.assertEqual(saved['planes'].dtype, torch.uint8)
                self.assertEqual(saved['planes_scale'], 10)
                self.assertEqual(saved['wdl'].dtype, torch.uint8)
                self.assertEqual(saved['q'].dtype, torch.uint8)
            with patch('neural.distill.validation_mask', side_effect=AssertionError('rehash')):
                loaded, validation = distill.load_shards(root)
            self.assertTrue(loaded and validation)

    def test_compact_learner_staging_and_optimizer_portable_fallback(self):
        n = 9
        tensors = (
            torch.zeros(n, 7, 10, 10, dtype=torch.uint8),
            torch.ones(n, 13, dtype=torch.bool),
            torch.zeros(n, 13),
            torch.ones(n, dtype=torch.uint8),
            torch.full((n, 13), 3, dtype=torch.uint8),
            torch.arange(3), torch.arange(3, n),
        )
        staged, resident = distill.stage_training_tensors(*tensors, 'cpu')
        self.assertFalse(resident)
        self.assertEqual([t.dtype for t in staged[:5]],
                         [torch.uint8, torch.bool, torch.float32, torch.uint8, torch.uint8])
        model = torch.nn.Linear(3, 2)
        optimizer = distill.create_optimizer(model, 1e-3, 'cpu')
        loss = model(torch.ones(2, 3)).sum(); loss.backward(); optimizer.step()
        state = optimizer.state_dict()
        replacement = torch.nn.Linear(3, 2)
        replacement.load_state_dict(model.state_dict())
        restored = distill.create_optimizer(replacement, 2e-3, 'cpu')
        restored.load_state_dict(state)
        self.assertTrue(restored.state)

    def test_mcts_hot_backup_has_no_per_level_cuda_truth_check(self):
        source = inspect.getsource(gpu_mcts.search_tree)
        self.assertNotIn('if not bool(active.any())', source)
        self.assertIn('level % 4', source)  # descent checks are amortized, not per level
        self.assertNotIn('if bool(expanding.any())', source)

    def test_selfplay_and_environment_hot_paths_avoid_host_round_trips(self):
        actor = (ROOT/'neural/gpu_selfplay.py').read_text()
        environment = (ROOT/'neural/gpu_env.py').read_text()
        self.assertNotIn('.cpu().tolist()', actor)
        self.assertNotIn('if is_drop.any()', environment)
        self.assertNotIn('if is_transform.any()', environment)
        self.assertNotIn('if f.any()', environment)
        self.assertNotIn('if cw.any()', environment)
        self.assertNotIn('if ccw.any()', environment)

    def test_modal_pipeline_uses_fast_compression_and_optimizer_sidecar(self):
        source = (ROOT/'neural/modal_app.py').read_text()
        self.assertIn('C4_REPLAY_GZIP_LEVEL', source)
        self.assertIn('"1"', source)
        self.assertIn('DISTILL_INIT_OPT', source)
        self.assertIn('optimizer.pt', source)
        self.assertIn('.opt.partial', source)


if __name__ == '__main__':
    unittest.main()
