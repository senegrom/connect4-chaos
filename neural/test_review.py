"""Review regressions using the real CPU tensor and helper implementations."""
import ast
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import torch
from .training_config import DEFAULT_SIMS, validate_selfplay
from .distill import load_shards, without_heldout_positions
from .search_quality import blunder_rate

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def shard(shapes, replay=False, scaled=False):
    n = len(shapes)
    planes = torch.zeros(n, 7, 10, 10)
    legal = torch.zeros(n, 13, dtype=torch.bool)
    for i, (r, c, k, chaos) in enumerate(shapes):
        planes[i, 2, :r, :c] = 1
        planes[i, 3] = k / 10
        planes[i, 4] = float(chaos)
        legal[i, :c] = True
        legal[i, 10:] = chaos
    policy = torch.zeros(n, 13)
    policy[:, 0] = 1
    result = dict(planes=planes, legal=legal, policy=policy, wdl=torch.ones(n, dtype=torch.long),
                  q=torch.full((n, 13), 3, dtype=torch.long), config=shapes[0][:3])
    if scaled:
        result.update(planes=(planes * 10).round().to(torch.uint8), planes_scale=10)
    if replay:
        result['source'] = 'selfplay'
    return result


class Net:
    def __call__(self, planes, legal):
        n = len(planes)
        return (torch.zeros(n, 13), torch.zeros(n, 3), torch.zeros(n, 13, 3))


class ReviewTests(unittest.TestCase):
    def test_defaults_and_invalid_configuration(self):
        validate_selfplay(10, DEFAULT_SIMS)
        for name in ['neural/modal_loop.py', 'neural/modal_app.py', 'neural/gpu_selfplay.py']:
            source = (ROOT / name).read_text()
            self.assertIn('DEFAULT_SIMS', source)
            self.assertIn('validate_selfplay(', source)
        ps = (ROOT / 'scripts/launch-modal-loop.ps1').read_text()
        self.assertEqual(int(re.search(r'\$Sims = (\d+)', ps)[1]), DEFAULT_SIMS)
        for games, sims, shapes, targets, share in [(0, 128, 'all', 0, .25), (1, 0, 'all', 0, .25),
                (1, 128, '12x4c4chaos', 0, .25), (1, 128, 'all', -1, .25),
                (1, 128, 'all', 0, float('nan'))]:
            with self.assertRaises(ValueError): validate_selfplay(games, sims, shapes, targets, share)

    def test_invalid_driver_options_do_not_spawn(self):
        tree = ast.parse((ROOT / 'neural/modal_loop.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        env = dict(validate_selfplay=validate_selfplay, GAMES=1, SIMS=0, SHAPES='all', TARGET_SIMS=0, TARGET_SHARE=.25)
        exec(compile(ast.Module(body=[main], type_ignores=[]), 'modal_loop.py', 'exec'), env)
        with self.assertRaises(ValueError): env['main']()  # fails before any remote or filesystem access

    def test_invalid_remote_options_do_not_allocate_or_read(self):
        tree = ast.parse((ROOT / 'neural/modal_app.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'selfplay_gpu')
        fn.decorator_list = []
        env = dict(validate_selfplay=validate_selfplay, DEFAULT_SIMS=DEFAULT_SIMS)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'modal_app.py', 'exec'), env)
        with self.assertRaises(ValueError): env['selfplay_gpu']('model.pt', 1, 'all', 1, sims=0)

    def test_late_checkpoint_read_never_changes_new_pointer(self):
        tree = ast.parse((ROOT / 'neural/modal_loop.py').read_text())
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['with_timeout', 'read_model', 'mirror_model']]
        old_started, release, finished = threading.Event(), threading.Event(), threading.Event()
        class Volume:
            def read_file(self, name):
                if name.endswith('old.pt'):
                    old_started.set()
                    release.wait(2)
                    finished.set()
                return [name.encode()]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = dict(Path=Path, ROOT=root, MODELS=root/'models', vol=Volume(), threading=threading)
            exec(compile(ast.Module(body=funcs, type_ignores=[]), 'modal_loop.py', 'exec'), env)
            deadline = env['with_timeout']
            env['with_timeout'] = lambda _seconds, work, *args: deadline(.02, work, *args)
            try:
                with self.assertRaises(TimeoutError): env['mirror_model']('old.pt')
                self.assertTrue(old_started.is_set())
                env['mirror_model']('new.pt')
                release.set()
                self.assertTrue(finished.wait(2))
                self.assertEqual(Path((root/'current-model.txt').read_text().strip()).name, 'new.pt')
                self.assertFalse((root/'models'/'old.pt').exists())
            finally: release.set()

    def test_mixed_replay_excludes_exact_and_rotated_heldout_boards(self):
        for scaled in (False, True):
            with self.subTest(scaled=scaled), tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,
                    DISTILL_HOLDOUT_CONFIGS='6x6c4classic,4x6c4chaos'):
                root = Path(temp)
                torch.save(shard([(6,6,4,False)]), root/'6x6c4classic-0000.pt')
                torch.save(shard([(6,6,4,False)]), root/'6x6c4classic-0001.pt')
                torch.save(shard([(6,6,4,False),(6,4,4,True),(4,6,4,True),(5,5,4,False)], True, scaled), root/'gpu-sp-1.pt')
                train, held = load_shards(root)
                self.assertEqual(sum(len(s['wdl']) for s in train), 1)
                self.assertEqual(len(held), 1)
                self.assertEqual(int((train[0]['planes'][0,2,:,0] > 0).sum()), 5)

    def test_replay_window_filters_only_needed_newest_shards(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,
                DISTILL_HOLDOUT_CONFIGS='6x6c4classic', DISTILL_REPLAY_WINDOW='1'):
            root = Path(temp)
            for index in range(3):
                path = root/f'gpu-sp-{index}.pt'
                torch.save(shard([(6,6,4,False),(5,5,4,False)], True), path)
                os.utime(path, (index + 1, index + 1))
            with patch('neural.distill.without_heldout_positions', wraps=without_heldout_positions) as filter_shard:
                train, held = load_shards(root)
            self.assertEqual(filter_shard.call_count, 1)
            self.assertEqual(sum(len(s['wdl']) for s in train), 1)
            self.assertEqual(train[0]['mtime'], 3)
            self.assertFalse(held)

    def test_empty_training_split_never_promotes_holdout(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, DISTILL_HOLDOUT_CONFIGS='6x6c4classic'):
            root = Path(temp)
            torch.save(shard([(6,6,4,False)]), root/'6x6c4classic-0000.pt')
            with self.assertRaisesRegex(ValueError, 'No training positions'): load_shards(root)

    def test_measurement_bounds_every_tensor(self):
        data = shard([(4,4,4,False)] * 600)
        for limit in (1, 127, 511, 512, 513, 599, 600, 1000):
            with self.subTest(limit=limit):
                rate, count = blunder_rate(Net(), data, 0, limit, 'cpu')
                self.assertEqual(count, min(limit, 600))
                self.assertEqual(rate, 0)
        for limit in (0, -1, 1.2, True):
            with self.assertRaises(ValueError): blunder_rate(Net(), data, 0, limit, 'cpu')
        rate, count = blunder_rate(Net(), data, 2, 3, 'cpu')
        self.assertEqual(count, 3)


if __name__ == '__main__': unittest.main()
