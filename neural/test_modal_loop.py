"""Exercise real driver control flow, lineage restoration and both mirror modes."""
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock

from .checkpoint_lineage import lineage_record
from .test_support import scripted_driver


class DriverTests(unittest.TestCase):
    def run_driver(self, mirror, *, broken_history=False, exact=None, extra_env=None, until=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'new-root'
            names = [f'big{n}-abcdef.pt' for n in range(13)]
            records = {f'models/{names[n]}.lineage.json': lineage_record(names[n], names[n - 1], n)
                       for n in range(1, 7)}
            # A retained failure and an unrelated successful branch cannot become ancestors.
            records['models/big6-deadbeef.pt.lineage.json'] = lineage_record('big6-deadbeef.pt', 'other.pt', 6)

            def learner(call):
                if call.polls == 2:
                    return ConnectionError('getaddrinfo failed')
                return dict(exit=0, model=names[call.args[0]], seconds=1, lines=[])

            def actor(call):
                if call.polls < 4:
                    return TimeoutError()
                return dict(exit=0, shard=f'gpu-sp-{call.index + 1}.pt.gz', seconds=1,
                            shard_bytes=1000, out='self-play: 8192 games, 260000 positions')

            def listing(path, _exceptions):
                # The initial checkpoint's own entry and the data the learner
                # reads, for the preflight; the lineage must never come from
                # scanning models/.
                shards = {f'models/{names[6]}': '', exact or 'datasets-v3': '/classic-4x4-c4-0001.pt',
                          'replay-gpu': '/gpu-sp-1-1.pt.gz'}
                if path not in shards:
                    raise AssertionError('must not scan checkpoint filenames')
                return [types.SimpleNamespace(path=path + shards[path], size=1, mtime=0)]

            argv = [names[6], '7', '2', '8192', '10', '64', '4e-4', '4000000', '1000000', '64', '2', '5']
            if exact is not None or until is not None:
                argv += ['all', '0', '0.25', '0', '1', '0.75', 'visits', '0', exact or 'datasets-v3']
            if until is not None:
                argv.append(str(until))
            # The last generation trained: the stop file after six, or until_gen.
            last = 12 if until is None else until
            logs, publications = [], []

            def log(message):
                # The stop lands as the sixth checkpoint is logged, before the
                # driver can submit a seventh learner in the same iteration.
                logs.append(message)
                if message.startswith('learner gen ') and ' done ' in message:
                    publications.append(message)
                    if len(publications) == 6 and until is None:
                        (root / 'modal-loop.stop').write_text('stop')

            def forbidden(*args):
                raise AssertionError('mirroring is disabled')

            fetch = Mock(side_effect=(lambda name: (root / name, 1000)) if mirror else forbidden)
            model_mirror = Mock(side_effect=(lambda name: root / name) if mirror else forbidden)
            state = scripted_driver(
                root, argv=argv, script={'learner': learner, 'actor': actor}, listing=listing,
                lineage=(lambda path: ConnectionError('network unavailable')) if broken_history else records,
                real_history=True, max_ticks=200,
                env=dict(extra_env or {}, C4_MIRROR=None if mirror is None else '1' if mirror else '0'),
                overrides=dict(log=log, fetch_shard=fetch, mirror_model=model_mirror))
            self.assertIsNone(state.error)
            self.assertEqual(state.module['MIRROR'], bool(mirror))
            self.assertEqual(len(publications), last - 6)
            learners = state.calls['learner']
            self.assertEqual([call.args[0] for call in learners], list(range(7, last + 1)))
            self.assertEqual([call.args[1] for call in learners], names[6:last])
            self.assertEqual({call.kwargs['exact_subdir'] for call in learners}, {exact or 'datasets-v3'})
            extra_env = extra_env or {}
            self.assertEqual({call.kwargs['holdout_configs'] for call in learners},
                             {extra_env.get('DISTILL_HOLDOUT_CONFIGS', '')})
            self.assertEqual({call.kwargs['gzip_level'] for call in state.calls['actor']},
                             {int(extra_env.get('C4_REPLAY_GZIP_LEVEL', '1'))})
            expected = [(names[n], names[n - 5]) for n in ((12,) if broken_history else (8, 10, 12))
                        if n <= last]
            self.assertEqual([call.args[:2] for call in state.calls['arena']], expected)
            # ARENA_LAG + 1 = 6 checkpoints need five sidecars, not the whole
            # ancestry; one more read checks the first generation against it.
            self.assertEqual(len([path for path in state.reads if path.endswith('.lineage.json')]),
                             2 if broken_history else 6)
            self.assertTrue(any('while polling; still tracked' in s for s in logs))
            self.assertTrue(any('learner pacing:' in s for s in logs))
            self.assertTrue(logs[-1].startswith('loop end:'))
            self.assertIn(f'next gen {last + 1}', logs[-1])
            if until is None:
                self.assertEqual(state.cancelled, [])
            else:
                # Self-play still running when the last generation lands is
                # cancelled rather than paid for; the arena that is due plays.
                self.assertTrue(state.cancelled)
                self.assertTrue(all(cid.startswith('fc-actor-') for cid in state.cancelled))
                self.assertTrue(any(f'generation {until} published: training done' in s for s in logs))
            self.assertEqual([call.args[0] for call in state.volume.listdir.call_args_list],
                             [f'models/{names[6]}', exact or 'datasets-v3', 'replay-gpu'])
            if mirror:
                self.assertEqual(model_mirror.call_count, last - 6)
                self.assertGreater(fetch.call_count, 0)
            else:
                fetch.assert_not_called()
                model_mirror.assert_not_called()
                self.assertFalse(state.module['REPLAY'].exists())
                self.assertFalse(state.module['MODELS'].exists())
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

    def test_until_gen_stops_after_that_generation_and_its_arena(self):
        self.run_driver(False, until=10)

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
                # A driver that got past validation would retry its failed
                # spawns every 60 s forever; the first sleep ends this one.
                state = scripted_driver(Path(temp), argv=['big1-abc.pt', '2'], env={name: value}, crash_after=1)
                self.assertIsInstance(state.error, ValueError)
                self.assertEqual(state.spawns, [])

if __name__ == '__main__':
    unittest.main()
