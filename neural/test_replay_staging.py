"""Modal replay staging through real filtering, CPU training and checkpoint IO."""
from __future__ import annotations

from contextlib import redirect_stdout
import gzip
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from . import distill
from .data_split import SPLIT_VERSION, validation_mask
from .model import PolicyValueNet
from .test_modal_arena import function
from .test_review import shard

ROOT = Path(__file__).resolve().parents[1]


def replay(shapes, *, current=True, marker=0.25, scaled=False):
    data = shard(shapes, replay=True, scaled=scaled)
    data['root_value'] = torch.arange(len(shapes), dtype=torch.float32) / 100 + marker
    if current:
        data.update(split_version=SPLIT_VERSION, validation=validation_mask(data['planes']))
    return data


class ReplayStagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        shapes = [(r, c, 4, False) for r in range(4, 8) for c in range(4, 8)]
        cls.eligible = [s for s in shapes if not validation_mask(shard([s])['planes'])[0]]
        cls.reserved = next(s for s in shapes if validation_mask(shard([s])['planes'])[0])

    def invoke(self, newest, *, holdout='', window=8, older=None, corrupt=False, same_time=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tables = root / 'tables'
            models = tables / 'models'; models.mkdir(parents=True)
            exact = tables / 'exact'; exact.mkdir()
            archive = tables / 'replay'; archive.mkdir()
            exact_data = shard([self.eligible[0]] * 8)
            exact_data.update(split_version=SPLIT_VERSION, split='train')
            torch.save(exact_data, exact / 'exact-0001.pt')
            net = PolicyValueNet(4, 1, 4)
            initial = net.stem[0].weight.detach().clone()
            torch.save({'model': net.state_dict(), 'arch': (4, 1, 4)}, models / 'init.pt')
            older = older if older is not None else replay([self.eligible[0]] * 8, marker=0.5)
            # Insert z before a: equal mtimes must still match load_shards' name order.
            records = [('z-old', older, 1000), ('a-new', newest, 1000 if same_time else 2000)]
            for name, payload, mtime in records:
                buffer = io.BytesIO(); torch.save(payload, buffer)
                path = archive / f'gpu-sp-{name}.pt.gz'
                path.write_bytes(gzip.compress(buffer.getvalue()))
                os.utime(path, (mtime, mtime))
            if corrupt:
                path = archive / 'gpu-sp-corrupt.pt.gz'
                path.write_bytes(b'truncated gzip')
                os.utime(path, (3000, 3000))
            volume = SimpleNamespace(reload=Mock(), commit=Mock())
            seen = {}
            def local_path(path):
                path = str(path)
                return root / path.lstrip('/') if path.startswith(('/tmp/replay-', '/tmp/learn-')) else Path(path)
            load = distill.load_shards
            def observed_load(paths):
                train, held = load(paths)
                seen['roots'] = [float(v) for s in train if s.get('source') == 'selfplay'
                                 for v in s['root_value']]
                seen['files'] = sorted(p.name for p in (root / 'tmp/replay-7').glob('*.pt'))
                return train, held
            def run(command, **kwargs):
                output = io.StringIO()
                with patch.object(sys, 'argv', command[2:]), patch.dict(os.environ, kwargs['env'], clear=True), \
                        patch.object(distill, 'load_shards', side_effect=observed_load), \
                        patch.object(torch.cuda, 'is_available', return_value=False), redirect_stdout(output):
                    distill.main()
                return subprocess.CompletedProcess(command, 0, output.getvalue(), '')
            learn = function(ROOT / 'neural/modal_app.py', 'learn', dict(
                Path=local_path, os=os, time=time, TABLES=str(tables), tables=volume,
                LEARNER_GPU='cpu', subprocess=SimpleNamespace(run=run)))
            env = {'DISTILL_HOLDOUT_CONFIGS': holdout, 'DISTILL_PERSIST_OPTIMIZER': '0',
                   'DISTILL_INIT_OPT': '', 'DISTILL_PROFILE_STEPS': '0'}
            with patch.dict(os.environ, env, clear=True):
                result = learn(7, 'init.pt', steps=2, batch=4, replay_window=window,
                               exact_subdir='exact', replay_subdir='replay')
            self.assertEqual(result['exit'], 0)
            self.assertIsNotNone(result['model'])
            saved = torch.load(models / result['model'], weights_only=True)
            self.assertFalse(torch.equal(initial, saved['model']['stem.0.weight']))
            self.assertTrue(all(torch.isfinite(t).all() for t in saved['model'].values()))
            volume.commit.assert_called_once()
            # Direct-loader reference includes the older files, not just the
            # wrapper's selected subset. Compare exactly which rows trained.
            baseline = root / 'all-replay'; baseline.mkdir()
            for name, payload, mtime in records:
                path = baseline / f'gpu-sp-{name}.pt'
                torch.save(payload, path); os.utime(path, (mtime, mtime))
            with patch.dict(os.environ, dict(env, DISTILL_REPLAY_WINDOW=str(window)), clear=True), \
                    redirect_stdout(io.StringIO()):
                train, _ = load(f'{exact};{baseline}')
            expected = [float(v) for s in train if s.get('source') == 'selfplay' for v in s['root_value']]
            self.assertEqual(seen['roots'], expected)
            self.assertEqual(result['replay_positions'], len(expected))
            self.assertTrue(any(f'keeping {len(expected)} newest eligible positions' in s
                                for s in result['lines']) or not expected)
            self.assertFalse((root / 'tmp/replay-7').exists())
            return result, seen

    def test_whole_board_holdout_does_not_starve_older_replay(self):
        excluded = self.eligible[1]
        r, c, k, _ = excluded
        result, seen = self.invoke(replay([excluded] * 8), holdout=f'{r}x{c}c{k}classic')
        self.assertEqual(result['replay_positions'], 8)
        self.assertEqual(result['excluded_shards'], 1)
        self.assertEqual(result['skipped_shards'], 0)
        self.assertEqual(seen['files'], ['gpu-sp-z-old.pt'])
        self.assertTrue(any('replay fraction 0.75' in s for s in result['lines']))

    def test_legacy_validation_rows_do_not_starve_older_replay(self):
        for scaled in (False, True):
            with self.subTest(scaled=scaled):
                result, _ = self.invoke(replay([self.reserved] * 8, current=False, scaled=scaled))
                self.assertEqual(result['replay_positions'], 8)
                self.assertEqual(result['excluded_shards'], 1)

    def test_rotated_chaos_holdout_uses_the_training_predicate(self):
        shapes = [(r, c, 4, True) for r in range(4, 7) for c in range(4, 7) if r != c]
        excluded = next(s for s in shapes if not validation_mask(shard([s])['planes'])[0])
        r, c, k, _ = excluded
        result, _ = self.invoke(replay([excluded] * 8), holdout=f'{c}x{r}c{k}chaos')
        self.assertEqual(result['replay_positions'], 8)
        self.assertEqual(result['excluded_shards'], 1)

    def test_partial_window_counts_eligible_rows_and_keeps_newest_tail(self):
        result, seen = self.invoke(replay([self.eligible[0], self.reserved] * 4, current=False), window=6)
        self.assertEqual(result['replay_positions'], 6)
        self.assertEqual(result['replay_shards'], 2)
        self.assertEqual(len(seen['roots']), 6)
        self.assertEqual(seen['roots'][-2:], replay([self.eligible[0]] * 8, marker=0.5)['root_value'][-2:].tolist())

    def test_eligible_newest_shard_alone_fills_the_window(self):
        result, seen = self.invoke(replay([self.eligible[0]] * 12), window=8)
        self.assertEqual(result['replay_shards'], 1)
        self.assertEqual(result['excluded_shards'], 0)
        self.assertEqual(seen['files'], ['gpu-sp-a-new.pt'])
        self.assertEqual(seen['roots'], replay([self.eligible[0]] * 12)['root_value'][-8:].tolist())

    def test_corrupt_shards_do_not_prevent_filling_the_eligible_window(self):
        result, _ = self.invoke(replay([self.reserved] * 8, current=False), corrupt=True)
        self.assertEqual(result['skipped_shards'], 1)
        self.assertEqual(result['excluded_shards'], 1)
        self.assertEqual(result['replay_positions'], 8)

    def test_archive_exhaustion_and_zero_window_report_actual_eligible_counts(self):
        result, _ = self.invoke(replay([self.reserved] * 8), older=replay([self.eligible[0]] * 3))
        self.assertEqual(result['replay_positions'], 3)
        result, seen = self.invoke(replay([self.eligible[0]] * 8), window=0, corrupt=True)
        self.assertEqual(result['replay_positions'], 0)
        self.assertEqual(result['replay_shards'], 0)
        self.assertEqual(result['skipped_shards'], 0)
        self.assertEqual(seen['files'], [])

    def test_equal_mtimes_use_the_same_tie_order_as_the_loader(self):
        result, seen = self.invoke(replay([self.eligible[0]] * 8), same_time=True)
        self.assertEqual(result['replay_positions'], 8)
        self.assertEqual(seen['files'], ['gpu-sp-a-new.pt'])

    def test_bad_configuration_fails_before_volume_or_shard_work(self):
        volume = SimpleNamespace(reload=Mock())
        learn = function(ROOT / 'neural/modal_app.py', 'learn', dict(
            Path=Path, os=os, time=time, TABLES='/unused', tables=volume,
            LEARNER_GPU='cpu', subprocess=Mock()))
        for window, holdout in [(-1, ''), (True, ''), (1, 'all'), (1, 'not-a-shape')]:
            with self.subTest(window=window, holdout=holdout), \
                    patch.dict(os.environ, DISTILL_HOLDOUT_CONFIGS=holdout):
                with self.assertRaises(ValueError):
                    learn(7, 'init.pt', replay_window=window)
        volume.reload.assert_not_called()


if __name__ == '__main__':
    unittest.main()
