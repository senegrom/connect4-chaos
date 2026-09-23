"""CPU regressions for shared replay staging, soup calibration and lineage IO."""
from contextlib import redirect_stdout
import ast
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from .data_split import SPLIT_VERSION, validation_mask
from .replay_staging import stage_replay
from .soup import calibration_data
from .training_config import validate_selfplay

ROOT = Path(__file__).resolve().parents[1]


def data(count=4, *, shape=(8, 8, 4, False), reserved=False, exact=False):
    r, c, k, chaos = shape
    planes = torch.zeros(count, 7, 10, 10, dtype=torch.uint8)
    planes[:, 2, :r, :c] = 10
    planes[:, 3] = k
    planes[:, 4] = 10 if chaos else 0
    legal = torch.zeros(count, 13, dtype=torch.bool)
    legal[:, :c] = True
    legal[:, 10:] = chaos
    policy = legal.float() / legal.sum(dim=1, keepdim=True)
    result = dict(planes=planes, planes_scale=10, legal=legal, policy=policy,
                  wdl=torch.ones(count, dtype=torch.uint8), q_default=3,
                  source='exact' if exact else 'selfplay', split_version=SPLIT_VERSION,
                  validation=torch.full((count,), reserved, dtype=torch.bool))
    if exact:
        result['q'] = torch.ones(count, 13, dtype=torch.uint8)
    return result


def save(path, payload, mtime):
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(buffer.getvalue()) if path.name.endswith('.gz') else buffer.getvalue())
    os.utime(path, (mtime, mtime))


def remote_function(name, namespace):
    tree = ast.parse((ROOT / 'neural/modal_app.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'modal_app.py', 'exec'), namespace)
    return namespace[name]


class SoupReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_validation_and_rotated_holdouts_do_not_starve_older_replay(self):
        for newest, holdouts in [(data(reserved=True), ()),
                (data(shape=(6, 4, 4, True)), ((4, 6, 4, True),))]:
            with self.subTest(holdouts=holdouts), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                save(root / 'archive/gpu-sp-a-new.pt.gz', newest, 200)
                save(root / 'archive/gpu-sp-z-old.pt.gz', data(), 100)
                stats = stage_replay(root / 'archive', root / 'stage', 4, holdouts)
                self.assertEqual((stats['positions'], stats['shards'], stats['excluded']), (4, 1, 1))
                planes, _ = calibration_data(root / 'stage', pool=8, exact_share=0,
                                             replay_window=4, holdout_shapes=holdouts, require_replay=True)
                self.assertEqual(len(planes), 4)
                self.assertTrue(((planes[:, 2, :, 0] > 0).sum(dim=1) == 8).all())

    def test_legacy_validation_uses_the_real_partition(self):
        shape = next((r, c, 4, False) for r in range(4, 9) for c in range(4, 9)
                     if validation_mask(data(shape=(r, c, 4, False))['planes'])[0])
        legacy = data(shape=shape)
        del legacy['validation']; del legacy['split_version']
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save(root / 'archive/gpu-sp-a-new.pt.gz', legacy, 200)
            save(root / 'archive/gpu-sp-z-old.pt.gz', data(), 100)
            stats = stage_replay(root / 'archive', root / 'stage', 4)
            self.assertEqual((stats['positions'], stats['excluded']), (4, 1))

    def test_window_samples_the_newest_eligible_rows_after_filtering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            newest = data(8)
            newest['planes'][:, 0, 0, 0] = torch.arange(1, 9, dtype=torch.uint8)
            newest['validation'] = torch.tensor([False, True] * 4)
            save(root / 'gpu-sp-z-new.pt', newest, 200)
            save(root / 'gpu-sp-a-old.pt', data(), 100)
            with patch.dict(os.environ, DISTILL_REPLAY_WINDOW='2'):
                planes, _ = calibration_data(root, pool=20, exact_share=0)
                again, _ = calibration_data(root, pool=20, exact_share=0)
                subsets = {tuple(sorted(calibration_data(root, pool=20, exact_share=0, seed=seed)[0]
                                        [:, 0, 0, 0].tolist())) for seed in range(8)}
            kept = planes[:, 0, 0, 0].tolist()
            # Two of the newest shard's four eligible rows, the same two each
            # time; the seed, not the shard's order, decides which.
            self.assertEqual(len(set(kept)), 2)
            self.assertLessEqual(set(kept), {1, 3, 5, 7})
            self.assertTrue(torch.equal(planes, again))
            self.assertGreater(len(subsets), 1)
            self.assertTrue(all(set(subset) <= {1, 3, 5, 7} for subset in subsets))

    def test_equal_mtimes_have_lexical_tie_break_across_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save(root / 'first/gpu-sp-z.pt', data(shape=(7, 7, 4, False)), 100)
            save(root / 'second/gpu-sp-a.pt', data(), 100)
            planes, _ = calibration_data(f'{root}/first;{root}/second', pool=4, exact_share=0)
            self.assertTrue(((planes[:, 2, :, 0] > 0).sum(dim=1) == 8).all())

    def test_corrupt_and_wrong_source_shards_are_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save(root / 'archive/gpu-sp-old.pt.gz', data(), 100)
            save(root / 'archive/gpu-sp-wrong.pt.gz', data(exact=True), 200)
            bad = root / 'archive/gpu-sp-corrupt.pt.gz'
            bad.write_bytes(b'bad gzip'); os.utime(bad, (300, 300))
            stats = stage_replay(root / 'archive', root / 'stage', 4)
            self.assertEqual((stats['positions'], stats['skipped']), (4, 2))
            self.assertEqual([p.name for p in (root / 'stage').iterdir()], ['gpu-sp-old.pt'])

    def test_zero_window_and_invalid_bounds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save(root / 'archive/gpu-sp.pt.gz', data(), 100)
            self.assertEqual(stage_replay(root / 'archive', root / 'stage', 0)['positions'], 0)
            for invalid in (-1, True, 1.5):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    stage_replay(root / 'archive', root / 'stage', invalid)
            with self.assertRaises(ValueError):
                calibration_data(root, pool=0)

    def test_exact_data_cannot_hide_missing_required_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save(root / '4x4c4classic-0001.pt', data(shape=(4, 4, 4, False), exact=True), 100)
            save(root / 'gpu-sp-reserved.pt', data(reserved=True), 200)
            with self.assertRaisesRegex(SystemExit, 'no eligible replay'):
                calibration_data(root, pool=8, require_replay=True)

    def invoke_soup(self, *, fail=False, all_reserved=False, holdout=''):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'models').mkdir()
            for name in ('a.pt', 'b.pt'):
                torch.save(dict(holdout_configs=holdout, data_split_version=SPLIT_VERSION), root / 'models' / name)
            save(root / 'archive/gpu-sp-a-new.pt.gz', data(reserved=True), 200)
            save(root / 'archive/gpu-sp-z-old.pt.gz', data(reserved=all_reserved), 100)
            # An abandoned directory must not become the next invocation's input.
            save(root / 'soup-replay/gpu-sp-stale.pt', data(shape=(7, 7, 4, False)), 999)
            seen = []
            def run(command, **kwargs):
                stage = Path(command[4].split(';')[1])
                seen.append(stage)
                self.assertNotEqual(stage, root / 'soup-replay')
                self.assertEqual(kwargs['env']['SOUP_REQUIRE_REPLAY'], '1')
                self.assertEqual(kwargs['env']['DISTILL_REPLAY_WINDOW'], '4')
                if fail:
                    raise OSError('trainer failed to start')
                with patch.dict(os.environ, kwargs['env'], clear=True):
                    planes, _ = calibration_data(command[4], pool=8, exact_share=0, require_replay=True)
                self.assertEqual(len(planes), 4)
                self.assertTrue(((planes[:, 2, :, 0] > 0).sum(dim=1) == 8).all())
                return subprocess.CompletedProcess(command, 0, 'calibrated', '')
            volume = SimpleNamespace(reload=Mock(), commit=Mock())
            fn = remote_function('soup', dict(Path=Path, TABLES=str(root), tables=volume,
                                             os=os, time=time, subprocess=SimpleNamespace(run=run)))
            if all_reserved or holdout == '8x8c4classic':
                with self.assertRaisesRegex(RuntimeError, 'no eligible replay'):
                    fn('a.pt,b.pt', 'out.pt', replay_window=4, replay_subdir='archive')
                self.assertEqual(seen, [])
                volume.commit.assert_not_called()
            elif fail:
                with self.assertRaisesRegex(OSError, 'trainer failed'):
                    fn('a.pt,b.pt', 'out.pt', replay_window=4, replay_subdir='archive')
                self.assertFalse(seen[0].exists())
                volume.commit.assert_not_called()
            else:
                for _ in range(2):
                    result = fn('a.pt,b.pt', 'out.pt', replay_window=4, replay_subdir='archive')
                    self.assertEqual((result['replay_positions'], result['excluded_shards']), (4, 1))
                self.assertNotEqual(seen[0], seen[1])
                self.assertTrue(all(not p.exists() for p in seen))
                self.assertEqual(volume.commit.call_count, 2)

    def test_remote_soup_uses_fresh_directory_and_eligible_counts(self):
        self.invoke_soup()

    def test_remote_soup_cleans_up_after_subprocess_exception(self):
        self.invoke_soup(fail=True)

    def test_remote_soup_rejects_excluded_replay_before_training(self):
        self.invoke_soup(all_reserved=True)

    def test_remote_staging_uses_checkpoint_holdouts_not_current_environment(self):
        with patch.dict(os.environ, DISTILL_HOLDOUT_CONFIGS=''):
            self.invoke_soup(holdout='8x8c4classic')

    def test_actor_wrapper_compresses_at_the_requested_level_and_names_its_gpu(self):
        # gzip records its level in the header's XFL byte: 4 fastest, 2 best, 0 other.
        for level, xfl in ((None, 4), (9, 2), (5, 0)):
            with self.subTest(level=level), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)

                def local_path(path):
                    path = str(path)
                    return root / path.lstrip('/') if path.startswith('/tmp/selfplay-') else Path(path)

                def run(command, **kwargs):
                    buffer = io.BytesIO()
                    torch.save(data(), buffer)
                    (Path(command[4]) / 'gpu-sp-3-1.pt').write_bytes(buffer.getvalue())
                    return subprocess.CompletedProcess(
                        command, 0, 'gpu: Fake GPU 80GB\nself-play [x]: 1 games, 4 positions\n', '')

                volume = SimpleNamespace(reload=Mock(), commit=Mock())
                fn = remote_function('selfplay_gpu', dict(
                    Path=local_path, TABLES=str(root), tables=volume, os=os, time=time,
                    subprocess=SimpleNamespace(run=run), validate_selfplay=validate_selfplay,
                    DEFAULT_SIMS=128, ACTOR_GPU='H100'))
                # Never read in the container: the level is an argument.
                with patch.dict(os.environ, C4_REPLAY_GZIP_LEVEL='9'):
                    result = fn('model.pt', 1, 'all', 3, **({} if level is None else {'gzip_level': level}))
                self.assertEqual(result['gpu'], 'Fake GPU 80GB')
                archive = (root / 'replay-gpu' / result['shard']).read_bytes()
                self.assertEqual(archive[8], xfl)
                self.assertEqual(result['shard_bytes'], len(archive))
                restored = torch.load(io.BytesIO(gzip.decompress(archive)), weights_only=True)
                self.assertTrue(torch.equal(restored['planes'], data()['planes']))
        volume = SimpleNamespace(reload=Mock())
        fn = remote_function('selfplay_gpu', dict(Path=Path, TABLES='/unused', tables=volume, os=os,
                                                  time=time, validate_selfplay=validate_selfplay,
                                                  DEFAULT_SIMS=128))
        for bad in (-1, 10, 1.5, True):
            with self.subTest(gzip_level=bad), self.assertRaisesRegex(ValueError, 'gzip_level'):
                fn('model.pt', 1, 'all', 3, gzip_level=bad)
        volume.reload.assert_not_called()

    def test_success_only_lineage_publication_in_real_learner_wrapper(self):
        for exit_code in (0, 1):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                def local_path(path):
                    path = str(path)
                    return root / path.lstrip('/') if path.startswith(('/tmp/replay-', '/tmp/learn-')) else Path(path)
                def run(command, **kwargs):
                    out = Path(command[4]); out.mkdir(parents=True)
                    (out / 'distilled.pt').write_bytes(b'completed checkpoint')
                    return subprocess.CompletedProcess(command, exit_code, '', 'evaluation failed' if exit_code else '')
                volume = SimpleNamespace(reload=Mock(), commit=Mock())
                (root / 'datasets-v3').mkdir()      # the learner refuses a missing exact corpus
                fn = remote_function('learn', dict(Path=local_path, TABLES=str(root), tables=volume,
                    os=os, time=time, subprocess=SimpleNamespace(run=run), LEARNER_GPU='cpu'))
                with patch.dict(os.environ, DISTILL_HOLDOUT_CONFIGS=''):
                    result = fn(7, 'seed.pt', steps=1, replay_window=0)
                model = result['model']
                self.assertTrue((root / 'models' / model).exists())
                sidecar = root / 'models' / f'{model}.lineage.json'
                self.assertEqual(sidecar.exists(), exit_code == 0)
                if sidecar.exists():
                    record = json.loads(sidecar.read_text())
                    self.assertEqual((record['model'], record['parent'], record['generation']), (model, 'seed.pt', 7))
                volume.commit.assert_called_once()


if __name__ == '__main__':
    unittest.main()
