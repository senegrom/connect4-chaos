"""Exercise real driver control flow, lineage restoration and both mirror modes."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from .checkpoint_lineage import lineage_record

ROOT = Path(__file__).resolve().parents[1]


class DriverTests(unittest.TestCase):
    def run_driver(self, mirror, *, broken_history=False, exact=None, extra_env=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'new-root'
            calls = {'actor': [], 'learner': [], 'arena': []}
            options = {'actor': [], 'learner': [], 'arena': []}
            names = [f'big{n}-abcdef.pt' for n in range(13)]
            records = {f'models/{names[n]}.lineage.json': lineage_record(names[n], names[n - 1], n)
                       for n in range(1, 7)}
            # A retained failure and an unrelated successful branch cannot become ancestors.
            records['models/big6-deadbeef.pt.lineage.json'] = lineage_record('big6-deadbeef.pt', 'other.pt', 6)

            class Call:
                def __init__(self, kind, payload, index):
                    self.kind, self.payload = kind, payload
                    self.object_id = f'fc-{kind}-{index}'
                    self.polls = 0
                def get(self, timeout=None):
                    self.polls += 1
                    if self.polls < (4 if self.kind == 'actor' else 2):
                        raise TimeoutError
                    if self.polls == 2 and self.kind == 'learner':
                        raise ConnectionError('getaddrinfo failed')
                    return self.payload

            class Stub:
                def __init__(self, kind):
                    self.kind = kind
                def spawn(self, *args, **kwargs):
                    calls[self.kind].append(args)
                    options[self.kind].append(kwargs)
                    index = len(calls[self.kind])
                    if self.kind == 'actor':
                        payload = dict(exit=0, shard=f'gpu-sp-{index}.pt.gz', seconds=1,
                                       shard_bytes=1000, out='self-play: 8192 games, 260000 positions')
                    elif self.kind == 'learner':
                        payload = dict(exit=0, model=names[args[0]], seconds=1, lines=[])
                    else:
                        payload = dict(exit=0, out='arena complete')
                    return Call(self.kind, payload, index)

            reads = []

            def read_file(path):
                reads.append(path)
                if broken_history:
                    raise ConnectionError('network unavailable')
                if path not in records:
                    raise FileNotFoundError(path)
                return [json.dumps(records[path]).encode()]
            volume = types.SimpleNamespace(read_file=read_file,
                listdir=Mock(side_effect=AssertionError('must not scan checkpoint filenames')))
            modal = types.ModuleType('modal')
            modal.Function = types.SimpleNamespace(from_name=lambda app, name:
                Stub({'selfplay_gpu': 'actor', 'learn': 'learner', 'arena': 'arena'}[name]))
            modal.Volume = types.SimpleNamespace(from_name=lambda name: volume)
            argv = ['modal_loop.py', names[6], '7', '2', '8192', '10', '64', '4e-4',
                    '4000000', '1000000', '64', '2', '5']
            if exact is not None:
                argv += ['all', '0', '0.25', '0', '1', '0.75', 'visits', '0', exact]
            env = dict(C4_NEURAL_ROOT=str(root), **(extra_env or {}))
            if mirror is not None:
                env['C4_MIRROR'] = '1' if mirror else '0'
            with patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', argv), \
                    patch.dict(sys.modules, modal=modal), redirect_stdout(io.StringIO()):
                loaded = runpy.run_path(str(ROOT / 'neural/modal_loop.py'), run_name='driver_test')
                state = loaded['main'].__globals__
                self.assertEqual(state['MIRROR'], bool(mirror))
                logs, publications = [], []
                def log(message):
                    logs.append(message)
                    if message.startswith('learner gen ') and ' done ' in message:
                        publications.append(message)
                        if len(publications) == 6:
                            state['STOP'].write_text('stop')
                def forbidden(*args):
                    raise AssertionError('mirroring is disabled')
                fetch = Mock(side_effect=(lambda name: (root / name, 1000)) if mirror else forbidden)
                model_mirror = Mock(side_effect=(lambda name: root / name) if mirror else forbidden)
                ticks = []
                def sleep(seconds):
                    ticks.append(seconds)
                    if len(ticks) > 200:
                        raise AssertionError('driver did not terminate')
                state.update(log=log, fetch_shard=fetch, mirror_model=model_mirror,
                             time=types.SimpleNamespace(time=lambda: 1234, sleep=sleep))
                state['main']()
            self.assertEqual(len(publications), 6)
            self.assertEqual([a[0] for a in calls['learner']], list(range(7, 13)))
            self.assertEqual([a[1] for a in calls['learner']], names[6:12])
            self.assertEqual({o['exact_subdir'] for o in options['learner']}, {exact or 'datasets-v3'})
            extra_env = extra_env or {}
            self.assertEqual({o['holdout_configs'] for o in options['learner']},
                             {extra_env.get('DISTILL_HOLDOUT_CONFIGS', '')})
            self.assertEqual({o['gzip_level'] for o in options['actor']},
                             {int(extra_env.get('C4_REPLAY_GZIP_LEVEL', '1'))})
            expected = [(names[n], names[n - 5]) for n in ((12,) if broken_history else (8, 10, 12))]
            self.assertEqual([(a[0], a[1]) for a in calls['arena']], expected)
            # ARENA_LAG + 1 = 6 checkpoints need five sidecars, not the whole ancestry.
            self.assertEqual(len([path for path in reads if path.endswith('.lineage.json')]),
                             1 if broken_history else 5)
            self.assertTrue(any('while polling; still tracked' in s for s in logs))
            self.assertTrue(any('learner pacing:' in s for s in logs))
            self.assertTrue(logs[-1].startswith('loop end:'))
            self.assertIn('next gen 13', logs[-1])
            volume.listdir.assert_not_called()
            if mirror:
                self.assertEqual(model_mirror.call_count, 6)
                self.assertGreater(fetch.call_count, 0)
            else:
                fetch.assert_not_called()
                model_mirror.assert_not_called()
                self.assertFalse(state['REPLAY'].exists())
                self.assertFalse(state['MODELS'].exists())
                self.assertFalse((root / 'current-model.txt').exists())
                self.assertTrue(any('Volume replay-gpu/' in s for s in logs))

    def test_mirror_opt_in(self):
        self.run_driver(True)

    def test_explicit_mirror_off(self):
        self.run_driver(False)

    def test_default_mirror_off(self):
        self.run_driver(None)

    def test_history_outage_fails_closed_and_training_continues(self):
        self.run_driver(False, broken_history=True)

    def test_exact_corpus_reaches_every_learner(self):
        self.run_driver(False, exact='exact/v4')

    def test_local_holdouts_and_gzip_level_reach_the_containers(self):
        # A container does not inherit the driver's environment; these used to
        # be read only inside it, where they always took their defaults.
        self.run_driver(False, extra_env={'DISTILL_HOLDOUT_CONFIGS': '4x4c3classic',
                                          'C4_REPLAY_GZIP_LEVEL': '6'})

    def test_bad_local_settings_fail_before_any_spawn(self):
        for name, value in (('C4_REPLAY_GZIP_LEVEL', '10'), ('DISTILL_HOLDOUT_CONFIGS', 'all'),
                            ('DISTILL_HOLDOUT_CONFIGS', '12x4c4chaos')):
            with self.subTest(name=name, value=value), tempfile.TemporaryDirectory() as temp:
                modal = types.ModuleType('modal')
                spawn = Mock(side_effect=AssertionError('spawned before validating'))
                modal.Function = types.SimpleNamespace(
                    from_name=lambda app, function: types.SimpleNamespace(spawn=spawn))
                modal.Volume = types.SimpleNamespace(from_name=lambda volume: object())
                with patch.dict(os.environ, {'C4_NEURAL_ROOT': temp, name: value}, clear=True), \
                        patch.object(sys, 'argv', ['modal_loop.py', 'big1-abc.pt', '2']), \
                        patch.dict(sys.modules, modal=modal):
                    loaded = runpy.run_path(str(ROOT / 'neural/modal_loop.py'), run_name='driver_test')
                    # A driver that got past validation would retry its failed
                    # spawns every 60 s forever; fail fast instead.
                    loaded['main'].__globals__.update(log=Mock(), time=types.SimpleNamespace(
                        time=lambda: 0, sleep=Mock(side_effect=AssertionError('reached the loop'))))
                    with self.assertRaises(ValueError):
                        loaded['main']()
                spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
