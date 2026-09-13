"""CPU regressions for the cutoff bootstrap, disjoint data and strict replay cap."""
from __future__ import annotations

import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import build_dataset, distill, gpu_mcts
from .chaos_game import NOT_TERMINAL, empty_state, successors, to_planes
from .data_split import SPLIT_VERSION, state_is_validation, validation_mask
from .gpu_env import ACTIONS, BoardBatch, hash_keys, step
from .test_review import shard


def positions(count=500):
    rng = random.Random(86420)
    state = empty_state(4, 5)
    found = []
    while len(found) < count:
        edges = [e for e in successors(state, 4, True) if e.terminal == NOT_TERMINAL]
        if not edges or state.pieces > 16:
            state = empty_state(4, 5)
            continue
        state = rng.choice(edges).child
        found.append(state)
    return found


def data(states, replay=False):
    result = shard([(s.rows, s.columns, 4, True) for s in states], replay)
    result['planes'] = torch.tensor([to_planes(s, 4, True) for s in states])
    return result


class RereviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.states = positions()

    def test_scalar_batch_scaled_mirrored_and_repetition_share_partition(self):
        planes = data(self.states)['planes']
        expected = torch.tensor([state_is_validation(s, 4, True) for s in self.states])
        self.assertTrue(bool(expected.any()) and bool((~expected).any()))
        self.assertTrue(torch.equal(validation_mask(planes), expected))
        self.assertTrue(torch.equal(validation_mask((planes * 10).round().to(torch.uint8)), expected))
        zeros = torch.zeros(len(planes), 13)
        mirrored, *_ = distill.mirror_batch(planes, zeros, zeros, zeros)
        mirrored[:, 5:] = 1
        self.assertTrue(torch.equal(validation_mask(mirrored), expected))
        # A known encoding pins the partition across Python processes/platforms.
        self.assertFalse(state_is_validation(empty_state(5, 5), 4, False))

    def test_builder_and_appends_use_disjoint_positions(self):
        pool = self.states
        class Table:
            def __init__(self, *_args, **_kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *_exc): pass
            def validate(self): pass
            def sample_state(self, rng): return rng.choice(pool), 0
            def edge_value_for_mover(self, _edge): return 0
        with tempfile.TemporaryDirectory() as temp, patch.object(build_dataset, 'PairTable', Table), \
                patch.object(build_dataset, 'SHARD', 64):
            root = Path(temp)
            build_dataset.build(root, 128, 'fixture:4:5:4:chaos', seed=17)
            build_dataset.build(root, 64, 'fixture:4:5:4:chaos', seed=99, start_index=2)
            for index in range(3):
                saved = torch.load(root/f'4x5c4chaos-{index:04d}.pt', weights_only=True)
                self.assertEqual(saved['split_version'], SPLIT_VERSION)
                self.assertEqual(saved['split'], 'validation' if index == 0 else 'train')
                self.assertTrue(bool((validation_mask(saved['planes']) == (index == 0)).all()))
            with self.assertRaises(SystemExit):
                build_dataset.build(root, 1, 'fixture:4:5:4:chaos', seed=9)

    def test_legacy_shards_and_replay_cannot_leak_validation(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,
                DISTILL_HOLDOUT_CONFIGS='', DISTILL_REPLAY_WINDOW='10000'):
            root = Path(temp)
            original = data(self.states)
            torch.save(original, root/'4x5c4chaos-0000.pt')
            torch.save(original, root/'4x5c4chaos-0001.pt')
            replay = data(self.states, True)
            zeros = torch.zeros(len(self.states), 13)
            replay['planes'], *_ = distill.mirror_batch(replay['planes'], zeros, zeros, zeros)
            replay['planes'][:, 5:] = 1
            replay['planes'] = (replay['planes'] * 10).round().to(torch.uint8)
            replay['planes_scale'] = 10
            torch.save(replay, root/'gpu-sp-1.pt')
            train, held = distill.load_shards(root)
            self.assertTrue(train and held)
            self.assertTrue(any(s.get('source') == 'selfplay' for s in train))
            for s in train:
                self.assertFalse(bool(validation_mask(s['planes'], s.get('planes_scale')).any()))
            for s in held:
                self.assertTrue(bool(validation_mask(s['planes']).all()))

    def test_search_quality_uses_the_same_legacy_validation_partition(self):
        from .search_quality import load_validation_shard
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, DISTILL_HOLDOUT_CONFIGS=''):
            path = Path(temp)/'4x5c4chaos-0000.pt'
            torch.save(data(self.states), path)
            held = load_validation_shard(path, 13)
            self.assertEqual(len(held['planes']), 13)
            self.assertTrue(bool(validation_mask(held['planes']).all()))

    def test_replay_cap_slices_every_tensor_and_keeps_newest_rows(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,
                DISTILL_HOLDOUT_CONFIGS='', DISTILL_REPLAY_WINDOW='10'):
            root = Path(temp)
            replay = shard([(5, 5, 4, False)] * 100, True)
            replay['wdl'] = torch.arange(100)
            replay['policy'][:, 0] = torch.arange(100)
            replay['q'][:, 0] = torch.arange(100)
            replay['legal'][:, 0] = torch.arange(100) % 2 == 0
            # Repetition flags do not affect the partition.
            replay['planes'][:, 5, 0, 0] = torch.arange(100)
            torch.save(replay, root/'gpu-sp-new.pt')
            # Tiny chunks ensure the cap also cuts inside the final chunk.
            with patch.object(distill, 'SPLIT_CHUNK', 8):
                train, held = distill.load_shards(root)
            self.assertFalse(held)
            self.assertEqual(sum(len(s['planes']) for s in train), 10)
            self.assertEqual(sorted(torch.cat([s['wdl'] for s in train]).tolist()), list(range(90, 100)))
            for s in train:
                self.assertTrue(torch.equal(s['policy'][:, 0].long(), s['wdl']))
                self.assertTrue(torch.equal(s['q'][:, 0], s['wdl']))
                self.assertTrue(torch.equal(s['legal'][:, 0], s['wdl'] % 2 == 0))
                self.assertTrue(torch.equal(s['planes'][:, 5, 0, 0].long(), s['wdl']))

    def test_replay_window_is_applied_after_filtering_and_across_files(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,
                DISTILL_HOLDOUT_CONFIGS='6x6c4classic', DISTILL_REPLAY_WINDOW='7'):
            root = Path(temp)
            for index in range(3):
                file = root/f'gpu-sp-{index}.pt'
                torch.save(shard([(5,5,4,False)] * 5 + [(6,6,4,False)] * 20, True), file)
                os.utime(file, (index + 1, index + 1))
            train, _ = distill.load_shards(root)
            self.assertEqual(sum(len(s['planes']) for s in train), 7)
            self.assertEqual({s['mtime'] for s in train}, {2, 3})
            with patch.dict(os.environ, DISTILL_REPLAY_WINDOW='0'):
                with self.assertRaisesRegex(ValueError, 'No training positions'):
                    distill.load_shards(root)
            with patch.dict(os.environ, DISTILL_REPLAY_WINDOW='-1'):
                with self.assertRaisesRegex(ValueError, 'non-negative'):
                    distill.load_shards(root)

    def test_production_depth_cutoff_bootstraps_without_extra_inference(self):
        self.check_cutoff(gpu_mcts.MAX_DEPTH)

    def test_odd_cutoff_and_other_games_terminal_values_are_preserved(self):
        self.check_cutoff(7, mixed=True)

    def check_cutoff(self, depth, mixed=False):
        rng = random.Random(271828)
        state = empty_state(10, 10)
        actions = []
        for _ in range(depth + 1):
            edge = rng.choice([e for e in successors(state, 5, False) if e.terminal == NOT_TERMINAL])
            actions.append(int(edge.action.removeprefix('drop')))
            state = edge.child
        calls = []
        def evaluator(_net, planes, legal):
            pieces = planes[:, :2].sum((1,2,3)).long()
            selected = torch.tensor([actions[int(n)] if i == 0 else 3 for i, n in enumerate(pieces)])
            calls.append(len(planes))
            logits = torch.full((len(planes), ACTIONS), -30.0)
            logits[torch.arange(len(planes)), selected] = 30.0
            wdl = torch.tensor([[-30.0, -30.0, 30.0]]).expand(len(planes), -1)
            q = torch.full((len(planes), ACTIONS, 3), -30.0); q[:, :, 0] = 30.0
            return logits, wdl, q
        board = BoardBatch([10, 6] if mixed else [10], [10, 7] if mixed else [10],
                           [5, 4] if mixed else [5], [False] * (2 if mixed else 1), 'cpu')
        if mixed:
            tactic = BoardBatch([6], [7], [4], [False], 'cpu')
            for col in (0,6,1,5,2,4):
                tactic, outcome = step(tactic, torch.tensor([col]))
                self.assertEqual(int(outcome[0]), NOT_TERMINAL)
            for field in ('mover','opponent','heights','pieces'):
                getattr(board, field)[1] = getattr(tactic, field)[0]
        zeros = torch.zeros(len(board), dtype=torch.bool)
        keys = hash_keys('cpu', generator=torch.Generator().manual_seed(73))
        with patch.object(gpu_mcts, 'MAX_DEPTH', depth):
            forest = gpu_mcts.search_tree(None, evaluator, board, zeros, zeros, depth + 1, False, keys=keys)
        node = 0
        for level in range(depth - 1):
            node = int(forest.child[0, node, actions[level]])
            self.assertGreaterEqual(node, 0)
        edge = (0, node, actions[depth - 1])
        self.assertEqual(float(forest.visits[edge]), 2)
        self.assertEqual(float(forest.value_sum[edge]), -2)
        # The sync-free step evaluates the network on every simulation and
        # discards the result for rows that did not expand: root, one per
        # expanding simulation, and one masked call for the cut-off simulation.
        self.assertEqual(len(calls), depth + 2)
        self.assertEqual(float(forest.visits[0,0].sum()), depth + 1)
        if mixed:
            self.assertEqual(int(forest.edge_terminal[1,0,3]), 1)
            self.assertEqual(float(forest.value_sum[1,0,3]), float(forest.visits[1,0,3]))


if __name__ == '__main__': unittest.main()
