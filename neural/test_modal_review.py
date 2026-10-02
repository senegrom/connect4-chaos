"""Post-merge driver/CLI regressions; no Modal account, GPU or network needed.

Load both complete production modules. Only the Modal boundary, clock and
lineage reader are mocked. The timeout hierarchy mirrors the public SDK:
modal.exception.TimeoutError is distinct from Python's polling TimeoutError.
"""
from contextlib import redirect_stderr, redirect_stdout
import inspect
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .test_support import (FULL, OUTCOMES, ROLES, ROOT, SMALL, scripted_driver, sdk_exceptions,
                           volume_entries)

# What the SDK raises when this client cannot reach Modal's API: no word
# about the call, which keeps running.
CLIENT_ERRORS = ('ServiceError', 'InternalError', 'ResourceExhaustedError', 'ConnectionError')


class DriverPollingTests(unittest.TestCase):
    def run_driver(self, role, error_name, *, stopping, message='connection lost: deadline timed out; service unavailable'):
        exceptions = sdk_exceptions()
        error_type = getattr(exceptions, error_name, None)
        if error_type is None:
            error_type = {'pending': TimeoutError, 'connection': ConnectionError}[error_name]
        terminal = error_name not in ('pending', 'connection') + CLIENT_ERRORS
        failed = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stop = root / 'modal-loop.stop'

            def outcome(call):
                # A scripted call is pending on its first poll, so the error
                # comes on the second. Terminal results stay failed forever;
                # transient failures and empty polls must be collected from
                # this SAME call.
                if call.index == 0:
                    if terminal or call.polls == 2:
                        failed.append(call.object_id)
                        if stopping:
                            stop.touch()
                        return error_type(message)
                    if not stopping:
                        stop.touch()
                elif terminal and not stopping:
                    stop.touch()  # a replacement was safely submitted
                return OUTCOMES[role](call)

            state = scripted_driver(root, script={role: outcome}, exceptions=exceptions, max_ticks=20,
                                    overrides={'ARENA_EVERY': 1 if role == 'arena' else 0})
            self.assertIsNone(state.error)
            self.assertTrue(stop.exists())
        self.assertTrue(state.logs[-1].startswith('loop end:'))
        self.assertEqual(len(failed), 1, 'polled the same completed failure again')
        first = state.calls[role][0]
        if terminal:
            self.assertEqual(first.polls, 2)
            self.assertEqual(len(state.calls[role]), 1 if stopping else 2)
            self.assertTrue(any(f'{role} ' in line and 'failed:' in line for line in state.logs))
            if role == 'learner':
                self.assertTrue(all(call.args[1] == 'initial.pt' for call in state.calls['learner']))
                self.assertTrue(all(call.args[0] == 1 for call in state.calls['learner']))
                if stopping:
                    self.assertIn('next gen 1, model initial.pt', state.logs[-1])
        else:
            self.assertEqual(len(state.calls[role]), 1, 'duplicated a pending or uncertain job')
            self.assertEqual(first.polls, 3)
            if error_name != 'pending':
                self.assertTrue(any('still tracked' in line for line in state.logs))

    def test_terminal_timeouts_drain_all_roles(self):
        for role in ROLES:
            for name in ('FunctionTimeoutError', 'OutputExpiredError'):
                with self.subTest(role=role, error=name):
                    self.run_driver(role, name, stopping=True)

    def test_other_completed_failures_ignore_misleading_transport_words(self):
        for role in ROLES:
            for name in ('RemoteError', 'ExecutionError', 'InternalFailure', 'DeserializationError'):
                with self.subTest(role=role, error=name):
                    self.run_driver(role, name, stopping=True)

    def test_terminal_failure_below_the_cap_releases_slot_for_replacement(self):
        # One failure is replaced; C4_MAX_FAILURES in a row stop submission
        # instead (FailureCapTests), where this used to replace forever.
        for role in ROLES:
            for name in ('FunctionTimeoutError', 'InternalFailure'):
                with self.subTest(role=role, error=name):
                    self.run_driver(role, name, stopping=False)

    def test_pending_and_transient_calls_are_not_replaced(self):
        for role in ROLES:
            for name in ('pending', 'connection') + CLIENT_ERRORS:
                for stopping in (False, True):
                    with self.subTest(role=role, error=name, stopping=stopping):
                        self.run_driver(role, name, stopping=stopping)

    def test_api_errors_without_transport_words_keep_the_call(self):
        # Modal raises ServiceError('') for an UNAVAILABLE or CANCELLED
        # gRPC status after its own retries, and InternalError('Internal
        # server error') for INTERNAL. Read as completed failures, three in
        # one poll stopped the loop and released calls that kept running.
        for role in ROLES:
            for name in CLIENT_ERRORS:
                for message in ('', 'Internal server error', 'rate limit exceeded'):
                    with self.subTest(role=role, error=name, message=message):
                        self.run_driver(role, name, stopping=False, message=message)

    def test_exception_subclasses_are_classified_by_type_not_exact_name(self):
        exceptions = sdk_exceptions()
        class CustomTimeout(exceptions.FunctionTimeoutError):
            pass
        modal = ModuleType('modal')
        modal.exception = exceptions
        modal.Function = SimpleNamespace(from_name=lambda *args: object())
        modal.Volume = SimpleNamespace(from_name=lambda *args: object())
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(sys.modules, {'modal': modal, 'modal.exception': exceptions}), \
                patch.dict(os.environ, {'C4_NEURAL_ROOT': temporary}), \
                patch.object(sys, 'argv', ['modal_loop.py', 'initial.pt', '1']):
            loaded = runpy.run_path(str(ROOT / 'neural/modal_loop.py'), run_name='classification_test')
            for message in ('', 'timed out', 'deadline', 'connectionerror', 'unavailable'):
                with self.subTest(message=message):
                    self.assertFalse(loaded['is_transient'](CustomTimeout(message)))


class FailureCapTests(unittest.TestCase):
    def test_consecutive_failures_in_one_role_stop_submission_and_drain(self):
        for role in ROLES:
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temp:
                state = scripted_driver(Path(temp), script={role: lambda call: dict(exit=1, err='boom')})
                self.assertIsNone(state.error)
                self.assertEqual(len(state.calls[role]), 3, 'one call per allowed failure, then no more')
                self.assertTrue(any(f'{role} failed 3 times in a row' in line for line in state.logs))
                self.assertTrue(state.logs[-1].startswith('loop end:'))
                self.assertEqual(state.journal['calls'], [])

    def test_the_cap_is_configurable_and_a_success_resets_the_count(self):
        codes = [1, 1, 0, 1, 1, 0]
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), script={'actor': lambda call: dict(
                exit=codes[call.index], shard='s.pt.gz', seconds=1, out='1 games, 1 positions')},
                stop_when=lambda state: len(state.calls['actor']) >= len(codes))
        self.assertIsNone(state.error)
        self.assertEqual(len(state.calls['actor']), len(codes))
        self.assertFalse(any('times in a row' in line for line in state.logs))
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), env={'C4_MAX_FAILURES': '1'},
                                    script={'actor': lambda call: dict(exit=1, err='boom')})
        self.assertEqual(len(state.calls['actor']), 1)
        self.assertTrue(any('actor failed 1 times in a row' in line for line in state.logs))

    def test_spawn_failures_count_unless_transient(self):
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), spawn_errors={
                'actor': lambda attempt: RuntimeError("App 'connect4-chaos' not found")})
        self.assertEqual((state.attempts['actor'], len(state.calls['actor'])), (3, 0))
        self.assertTrue(any('actor failed 3 times in a row' in line for line in state.logs))
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), spawn_errors={
                'actor': lambda attempt: ConnectionError('connection lost') if attempt <= 5 else None},
                stop_when=lambda state: len(state.calls['actor']) >= 1)
        self.assertEqual((state.attempts['actor'], len(state.calls['actor'])), (6, 1))
        self.assertFalse(any('times in a row' in line for line in state.logs))

    def test_failed_learner_checkpoint_is_removed_unless_only_evaluation_failed(self):
        results = [dict(exit=1, model='big1-half.pt', seconds=1, lines=[], err='optimizer save failed'),
                   dict(exit=1, model='big1-good.pt', adopted=True, seconds=1, lines=[], err='evaluation failed')]
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), script={'learner': lambda call: results[call.index]
                                                        if call.index < 2 else dict(exit=0, model='big2-ok.pt',
                                                                                    seconds=1, lines=[])},
                                    stop_when=lambda state: len(state.calls['learner']) >= 3)
        self.assertIsNone(state.error)
        # The half-saved run is deleted and the same generation retried; the
        # run that failed only its evaluation is the next generation's parent.
        self.assertEqual(state.removed, ['models/big1-half.pt', 'models/big1-half.pt.opt'])
        self.assertEqual([call.args[:2] for call in state.calls['learner']],
                         [(1, 'initial.pt'), (1, 'initial.pt'), (2, 'big1-good.pt')])
        self.assertTrue(any('adopted models/big1-good.pt' in line for line in state.logs))
        self.assertTrue(any('removed the retained models/big1-half.pt' in line for line in state.logs))


class DriverStartTests(unittest.TestCase):
    def test_missing_initial_checkpoint_fails_before_any_spawn(self):
        def not_found(path, exceptions):
            raise exceptions.NotFoundError(path)
        def absent(path, exceptions):
            raise FileNotFoundError(path)
        for listing in (lambda path, exceptions: [], not_found, absent):
            with self.subTest(listing=listing), tempfile.TemporaryDirectory() as temp:
                state = scripted_driver(Path(temp), listing=listing)
                self.assertIsInstance(state.error, FileNotFoundError)
                self.assertIn('models/initial.pt is not on the Volume', str(state.error))
                self.assertEqual(state.spawned, 0)
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), stop_when=lambda state: True)
        self.assertEqual([call.args[0] for call in state.volume.listdir.call_args_list],
                         ['models/initial.pt', 'datasets-v3', 'replay-gpu'])
        self.assertIsNone(state.error)

    def test_bad_arguments_fail_before_the_preflight_or_any_spawn(self):
        bad = {1: ('initial', 'models/initial.pt'), 2: ('-1',), 10: ('0',), 16: ('-0.5', 'inf'), 17: ('2', 'yes'),
               18: ('1.5', '-0.1', 'nan'), 19: ('softmax',), 20: ('-1', 'nan'), 21: ('', '../elsewhere')}
        for index, values in bad.items():
            for value in values:
                argv = list(FULL)
                argv[index - 1] = value
                with self.subTest(argument=index, value=value), tempfile.TemporaryDirectory() as temp:
                    state = scripted_driver(Path(temp), argv=argv)
                    self.assertIsInstance(state.error, ValueError)
                    state.volume.listdir.assert_not_called()
                    self.assertEqual(state.spawned, 0)
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), argv=FULL, env={'C4_MAX_FAILURES': '0'})
            self.assertIsInstance(state.error, ValueError)

    def test_a_killed_driver_leaves_a_journal_of_the_calls_in_flight(self):
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), argv=['initial.pt', '1', '2'], crash_after=1,
                                    script={'actor': lambda call: TimeoutError()})
        self.assertIsInstance(state.error, KeyboardInterrupt)
        journaled = {(entry['role'], entry['id']) for entry in state.journal['calls']}
        self.assertEqual(journaled, {('learner', 'fc-learner-0'), ('actor', 'fc-actor-0'), ('actor', 'fc-actor-1')})
        learner = next(entry for entry in state.journal['calls'] if entry['role'] == 'learner')
        self.assertEqual((learner['gen'], learner['init']), (1, 'initial.pt'))
        self.assertTrue(any('driver exiting with 3 calls in flight' in line for line in state.logs))

    def test_restart_reattaches_journaled_calls_and_cancels_an_earlier_learner(self):
        journal = {'version': 1, 'calls': [
            dict(id='fc-actor-old0', role='actor', seed=5, model='big0-a.pt', spawned=1),
            dict(id='fc-learner-old', role='learner', gen=1, init='initial.pt', spawned=1),
            dict(id='fc-learner-stale', role='learner', gen=0, init='seed.pt', spawned=1),
            dict(id='fc-arena-old', role='arena', newer='big0-a.pt', older='seed.pt', spawned=1)]}
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), journal=journal,
                                    restored_args={'fc-learner-old': (1, 'initial.pt')},
                                    stop_when=lambda state: len(state.calls['learner']) >= 1)
        self.assertIsNone(state.error)
        self.assertEqual(set(state.restored), {entry['id'] for entry in journal['calls']})
        self.assertEqual(state.cancelled, ['fc-learner-stale'])
        # The reattached learner is this run's generation 1, so the first new
        # learner trains generation 2 from its checkpoint; the reattached
        # actor holds the only actor slot until it finishes.
        self.assertEqual(state.calls['learner'][0].args[:2], (2, 'big1-ok.pt'))
        done = next(i for i, line in enumerate(state.logs) if line.startswith('actor fc-actor-old0 done'))
        first_spawn = next(i for i, line in enumerate(state.logs) if line.startswith('actor spawned'))
        self.assertLess(done, first_spawn)
        self.assertTrue(any('arena completed' in line for line in state.logs))
        self.assertEqual(state.journal['calls'], [])

    def test_a_learner_for_this_or_a_later_generation_stops_the_start(self):
        # Rerunning the documented -Init big504 -Gen 505 after a crash at gen
        # 521 used to cancel the gen-521 learner and restart from 504.
        for gen, init in ((4, 'big3-b.pt'), (1, 'other.pt')):
            journal = {'version': 1, 'calls': [
                dict(id='fc-actor-old0', role='actor', seed=5, model='big0-a.pt', spawned=1),
                dict(id='fc-learner-later', role='learner', gen=gen, init=init, spawned=1)]}
            with self.subTest(gen=gen, init=init), tempfile.TemporaryDirectory() as temp:
                text = json.dumps(journal)
                state = scripted_driver(Path(temp), journal=text)
                self.assertIsInstance(state.error, ValueError)
                self.assertIn(f'rerun with -Init {init} -Gen {gen}', str(state.error))
                self.assertEqual((state.spawned, state.cancelled, state.restored), (0, [], {}))
                self.assertEqual(state.journal_text, text, 'the journal is left for a person')

    def test_a_restart_that_cannot_cancel_an_earlier_learner_refuses_to_start(self):
        # Dropped from the journal uncancelled, the earlier learner ran on
        # untracked and published beside the new run.
        text = json.dumps({'version': 1, 'calls': [
            dict(id='fc-learner-stale', role='learner', gen=0, init='seed.pt', spawned=1)]})
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), journal=text,
                                    cancel_errors={'learner': ConnectionError('UNAVAILABLE')})
        self.assertIsInstance(state.error, ValueError)
        self.assertIn('could not cancel learner fc-learner-stale', str(state.error))
        self.assertEqual((state.spawned, state.cancelled), (0, []))
        self.assertIn('fc-learner-stale', state.restored)
        self.assertEqual(state.journal_text, text, 'the journal is kept for the next start')

    def test_failures_of_reattached_calls_do_not_count(self):
        journal = {'version': 1, 'calls': [
            dict(id=f'fc-actor-old{n}', role='actor', seed=n, model='big0-a.pt', spawned=1) for n in range(3)]}
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), argv=['initial.pt', '1', '3'], journal=journal,
                                    script={'actor': lambda call: dict(exit=1, err='boom') if call.index < 0
                                            else dict(exit=0, shard='s.pt.gz', seconds=1, out='1 games, 1 positions')},
                                    stop_when=lambda state: len(state.calls['actor']) >= 3)
        self.assertIsNone(state.error)
        self.assertFalse(any('times in a row' in line for line in state.logs))
        self.assertEqual(len(state.calls['actor']), 3)

    def test_unreadable_journal_refuses_to_start(self):
        # The last is an arena journaled before arenas recorded their spawn.
        for text in ('not json', '[]', '{"version": 2, "calls": []}', '{"version": 1, "calls": [{"role": "actor"}]}',
                     '{"version": 1, "calls": [{"id": "fc-arena-0", "role": "arena", "newer": "a.pt", "older": "b.pt"}]}'):
            with self.subTest(journal=text), tempfile.TemporaryDirectory() as temp:
                state = scripted_driver(Path(temp), journal=text)
                self.assertIsInstance(state.error, ValueError)
                self.assertIn('call journal', str(state.error))
                self.assertEqual(state.spawned, 0)
                self.assertEqual(state.journal_text, text, 'an unreadable journal is left for a person')


class DriverRecoveryTests(unittest.TestCase):
    CEILINGS = {'actor': 100, 'learner': 100, 'arena': 100}
    RESULT = dict(exit=0, seconds=1, lines=[])

    def test_a_call_that_never_reports_is_cancelled_at_its_ceiling(self):
        # A remote TimeoutError or connection error is re-raised on every
        # poll, where it reads as "still running" or as an outage; a call the
        # SDK lost looks the same. Each used to hold its slot for ever.
        for role in ROLES:
            for error in (TimeoutError('an asyncio timeout in a Volume reload'), ConnectionError('UNAVAILABLE')):
                with self.subTest(role=role, error=type(error).__name__), tempfile.TemporaryDirectory() as temp:
                    state = scripted_driver(Path(temp), script={role: lambda call, error=error: error},
                                            overrides={'CEILING_SECONDS': self.CEILINGS}, clock=True)
                    self.assertIsNone(state.error)
                    self.assertEqual(len(state.calls[role]), 3, 'one call per allowed failure, then no more')
                    self.assertEqual(state.cancelled, [call.object_id for call in state.calls[role]])
                    self.assertTrue(any(line.startswith(f'{role} fc-{role}-0: no result') for line in state.logs))
                    self.assertTrue(any(f'{role} failed 3 times in a row' in line for line in state.logs))
                    self.assertTrue(state.logs[-1].startswith('loop end:'))
                    self.assertEqual(state.journal['calls'], [])

    def test_an_outage_past_the_ceiling_keeps_calls_it_cannot_cancel_and_reads_them_after(self):
        # Polls and cancels fail alike while the network is down, and the
        # calls finish meanwhile (the 2026-09-10 outage lasted 33 minutes). At
        # their ceiling they were released uncancelled: they ran on untracked,
        # their results were lost, and three actors' releases stopped the loop.
        def outage(ready):
            return lambda call: ConnectionError('UNAVAILABLE') if call.polls < 40 else ready(call)
        actor = outage(lambda call: dict(exit=0, shard=f'{call.object_id}.pt.gz', seconds=1,
                                         out='self-play: 1 games, 100 positions', shard_bytes=1))
        learner = outage(lambda call: dict(exit=0, model=f'big{call.args[0]}-ok.pt', seconds=1, lines=[]))
        argv = ['initial.pt', '1', '3', '1', '10', '64', '4e-4', '4000000', '0', '64', '0', '1']
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), argv=argv, script={'actor': actor, 'learner': learner},
                                    cancel_errors={'actor': ConnectionError('UNAVAILABLE'),
                                                   'learner': ConnectionError('UNAVAILABLE')},
                                    overrides={'CEILING_SECONDS': self.CEILINGS}, clock=True,
                                    stop_when=lambda state: any(line.startswith('learner gen 1 done')
                                                                for line in state.logs))
        self.assertIsNone(state.error)
        self.assertEqual(state.cancelled, [])
        self.assertFalse(any('times in a row' in line or 'and released' in line for line in state.logs))
        for cid in ('fc-actor-0', 'fc-actor-1', 'fc-actor-2', 'fc-learner-0'):
            said = [line for line in state.logs if line.startswith(f'{cid.split("-")[1]} {cid}:') and 'not cancelled' in line]
            self.assertEqual(len(said), 1, f'{cid}: said once, however long the outage')
        for cid in ('fc-actor-0', 'fc-actor-1', 'fc-actor-2'):
            self.assertTrue(any(line.startswith(f'actor {cid} done') for line in state.logs), cid)
        self.assertTrue(any('big1-ok.pt' in line for line in state.logs if line.startswith('learner gen 1 done')))
        self.assertEqual([call.object_id for call in state.calls['learner']], ['fc-learner-0'],
                         'no second learner beside the first')

    # K=1, never an arena: the learner's result decides each case.
    ONE_ACTOR = ['initial.pt', '1', '1', '1', '10', '64', '4e-4', '4000000', '0', '64', '0', '1']

    def run_learner_past_its_ceiling(self, learner, **options):
        with tempfile.TemporaryDirectory() as temp:
            return scripted_driver(Path(temp), argv=self.ONE_ACTOR, script={'learner': learner},
                                   overrides={'CEILING_SECONDS': self.CEILINGS}, clock=True,
                                   stop_when=lambda state: any(line.startswith('learner gen 1 done')
                                                               for line in state.logs), **options)

    def assert_one_learner_read(self, state):
        self.assertIsNone(state.error)
        self.assertTrue(any('big1-ok.pt' in line for line in state.logs if line.startswith('learner gen 1 done')))
        self.assertEqual([call.object_id for call in state.calls['learner']], ['fc-learner-0'],
                         'the generation was trained once')
        released = ('times in a row', 'learner failed', 'after its cancel; released', 'cancelled and released')
        self.assertFalse([line for line in state.logs if any(text in line for text in released)])

    def test_a_result_that_turns_up_while_a_cancel_goes_through_is_read(self):
        # When an outage ends, the cancel can be the first request to reach
        # Modal, for a call that finished meanwhile. It used to release the
        # call unread, and the generation was trained again from the old model.
        network = {'up': False}

        def learner(call):
            if call.age <= self.CEILINGS['learner']:
                return TimeoutError()
            if not network['up']:
                return ConnectionError('UNAVAILABLE')
            return dict(exit=0, model=f'big{call.args[0]}-ok.pt', seconds=1, lines=[])

        def cancel(call):
            if call.cancels < 3:
                return ConnectionError('UNAVAILABLE')
            network['up'] = True             # the network returns during this cancel
            return None

        state = self.run_learner_past_its_ceiling(learner, cancel_errors={'learner': cancel})
        self.assert_one_learner_read(state)
        self.assertEqual(state.cancelled, ['fc-learner-0'])
        self.assertTrue(any(line.startswith('learner fc-learner-0:') and 'polled once more' in line
                            for line in state.logs))

    def test_a_failed_poll_past_the_ceiling_gets_one_more_before_any_cancel(self):
        # A driver waking from sleep fails its first poll while the network
        # reconnects; cancelling then would have dropped a finished result.
        failed = set()

        def learner(call):
            if call.age <= self.CEILINGS['learner']:
                return TimeoutError()
            if call.object_id not in failed:
                failed.add(call.object_id)
                return ConnectionError('UNAVAILABLE')
            return dict(exit=0, model=f'big{call.args[0]}-ok.pt', seconds=1, lines=[])

        state = self.run_learner_past_its_ceiling(learner)
        self.assert_one_learner_read(state)
        self.assertEqual(state.cancelled, [], 'cancelled before the second poll')
        self.assertTrue(any(line.startswith('learner fc-learner-0:') and 'polled again' in line
                            for line in state.logs))

    def test_calls_within_their_ceiling_are_not_cancelled(self):
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), overrides={'CEILING_SECONDS': self.CEILINGS}, clock=True,
                                    stop_when=lambda state: len(state.calls['learner']) >= 4)
        self.assertIsNone(state.error)
        self.assertEqual(state.cancelled, [])
        self.assertFalse(any('no result' in line or 'times in a row' in line for line in state.logs))

    def test_a_corpus_without_training_shards_fails_before_any_spawn(self):
        def corpus(names):
            def listing(path, exceptions):
                if path != 'datasets-v3':
                    return volume_entries(path)
                if names is None:
                    raise exceptions.NotFoundError(path)
                return [SimpleNamespace(path=f'{path}/{name}') for name in names]
            return listing
        # Missing, empty, held-out shards only, and files that are not shards.
        for names in (None, (), ('classic-4x4-c4-0000.pt',), ('notes.txt', 'classic-4x4-c4.pt')):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as temp:
                state = scripted_driver(Path(temp), listing=corpus(names))
                self.assertIsInstance(state.error, FileNotFoundError)
                self.assertIn('datasets-v3/ on the Volume holds no exact training shards', str(state.error))
                self.assertEqual(state.spawned, 0)

    def test_an_empty_replay_holds_the_first_learner_until_a_window_is_written(self):
        # A learner that finds no replay trains on the exact rows alone. Two
        # actors, a 250-position window and no pacing after the first
        # generation; every actor reports 100 positions.
        argv = ['initial.pt', '1', '2', '1', '10', '64', '4e-4', '250', '0', '64', '0', '1']
        empty = lambda path, exceptions: [] if path == 'replay-gpu' else volume_entries(path)
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), argv=argv, listing=empty,
                                    stop_when=lambda state: len(state.calls['learner']) >= 1)
        self.assertIsNone(state.error)
        self.assertTrue(any('replay-gpu/ is empty' in line for line in state.logs))
        first = next(i for i, line in enumerate(state.logs) if line.startswith('learner spawned'))
        self.assertGreaterEqual(sum(line.startswith('actor ') and ' done ' in line for line in state.logs[:first]), 3)
        # Exact-only training (replay fraction 0) needs no window.
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), argv=argv + ['all', '0', '0.25', '0', '1', '0'], listing=empty,
                                    stop_when=lambda state: len(state.calls['learner']) >= 1)
        self.assertIsNone(state.error)
        self.assertFalse(any('is empty' in line for line in state.logs))
        first = next(i for i, line in enumerate(state.logs) if line.startswith('learner spawned'))
        self.assertFalse(any(line.startswith('actor ') and ' done ' in line for line in state.logs[:first]))

    def test_a_restart_resumes_the_replay_prefill_from_the_journal(self):
        # A window of 250 rows takes 278 positions (one in ten is kept for
        # validation); every actor reports 100. The shards written before the
        # restart used to end the wait at once.
        argv = ['initial.pt', '1', '2', '1', '10', '64', '4e-4', '250', '0', '64', '0', '1']
        empty = lambda path, exceptions: [] if path == 'replay-gpu' else volume_entries(path)
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), argv=argv, listing=empty, crash_after=3)
        self.assertIsInstance(state.error, KeyboardInterrupt)
        self.assertEqual(state.calls['learner'], [])
        written = state.journal['prefill']
        self.assertEqual(written % 100, 0)
        self.assertTrue(0 < written < 278, written)
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), argv=argv, journal={'version': 1, 'calls': [], 'prefill': 200},
                                    stop_when=lambda state: len(state.calls['learner']) >= 1)
        self.assertIsNone(state.error)
        self.assertTrue(any('replay-gpu/ holds 200 of the 278 positions' in line for line in state.logs))
        # Both actors finish in one poll: 200 + 200 reaches 278. Counting from
        # zero would take a third; without the journal the learner starts at once.
        first = next(i for i, line in enumerate(state.logs) if line.startswith('learner spawned'))
        self.assertEqual(sum(line.startswith('actor ') and ' done ' in line for line in state.logs[:first]), 2)
        self.assertNotIn('prefill', state.journal)
        for bad in ('-1', '"many"', '1.5'):
            with self.subTest(prefill=bad), tempfile.TemporaryDirectory() as temp:
                state = scripted_driver(Path(temp), argv=argv,
                                        journal='{"version": 1, "calls": [], "prefill": ' + bad + '}')
                self.assertIsInstance(state.error, ValueError)
                self.assertEqual(state.spawned, 0)

    def test_actors_whose_cancel_fails_at_the_last_generation_stay_tracked(self):
        # UNTIL_GEN 1: once generation 1 is published the actors are
        # cancelled. One whose cancel raised used to be dropped from the
        # journal while its H100 kept running.
        with tempfile.TemporaryDirectory() as temp:
            # The actor is still running when generation 1 is published.
            slow = lambda call: TimeoutError() if call.polls < 4 else dict(
                exit=0, shard=f'{call.object_id}.pt.gz', seconds=1, out='self-play: 1 games, 100 positions',
                shard_bytes=1)
            state = scripted_driver(Path(temp), argv=FULL + ['1'], script={'actor': slow},
                                    cancel_errors={'actor': RuntimeError('cancel refused')})
        self.assertIsNone(state.error)
        self.assertTrue(any('cancel failed (RuntimeError: cancel refused); still tracked' in line
                            for line in state.logs))
        tracked = next(i for i, line in enumerate(state.logs) if 'still tracked' in line)
        self.assertTrue(any(line.startswith('actor ') and ' done ' in line for line in state.logs[tracked:]))
        self.assertEqual(state.journal['calls'], [])
        self.assertTrue(state.logs[-1].startswith('loop end:'))

    def test_replay_staging_losses_reach_the_log(self):
        # Corrupt archives were skipped every generation, older shards filled
        # the window, and the log said only how many positions staged.
        staged = dict(self.RESULT, skipped_shards=2, excluded_shards=1,
                      replay_errors=['gpu-sp-9-9.pt.gz: EOFError: Compressed file ended'])
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), script={'learner': lambda call: dict(
                staged, model=f'big{call.args[0]}-ok.pt')}, stop_when=lambda state: len(state.calls['learner']) >= 1)
        self.assertIsNone(state.error)
        self.assertTrue(any('replay staging skipped 2 unreadable and excluded 1 ineligible shards: '
                            'gpu-sp-9-9.pt.gz: EOFError: Compressed file ended' in line for line in state.logs))

    def test_a_failed_learner_is_retried_without_waiting_for_fresh_positions(self):
        # Its spawn used up the pacing it waited for; the retry used to wait
        # for another MIN_NEW positions while the log said "retry in 120 s".
        argv = ['initial.pt', '1', '1', '1', '10', '64', '4e-4', '4000000', '250', '64', '0', '1']
        def learner(call):
            if call.index == 1:
                return dict(self.RESULT, exit=1, err='boom')
            return dict(self.RESULT, model=f'big{call.args[0]}-ok.pt')
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), argv=argv, script={'learner': learner},
                                    stop_when=lambda state: len(state.calls['learner']) >= 3)
        self.assertIsNone(state.error)
        self.assertEqual([call.args[0] for call in state.calls['learner']], [1, 2, 2])
        failure = next(i for i, line in enumerate(state.logs) if line.startswith('learner gen 2 exit=1'))
        retry = next(i for i, line in enumerate(state.logs) if line.startswith('learner spawned fc-learner-2'))
        self.assertFalse(any(line.startswith('actor ') and ' done ' in line for line in state.logs[failure:retry]))
        self.assertFalse(any('learner pacing' in line for line in state.logs[failure:retry]))

    def test_the_first_generation_follows_the_initial_models_lineage(self):
        # A wrong one publishes lineage whose generations do not increase,
        # and read_history then refuses the whole chain.
        lineage = {'models/initial.pt.lineage.json': dict(version=1, model='initial.pt', parent='seed.pt',
                                                           generation=4)}
        for gen in ('1', '4', '6'):
            with self.subTest(gen=gen), tempfile.TemporaryDirectory() as temp:
                state = scripted_driver(Path(temp), argv=['initial.pt', gen, *SMALL[2:]], lineage=lineage)
                self.assertIsInstance(state.error, ValueError)
                self.assertIn('generation 4 by its lineage, so the first generation is 5', str(state.error))
                self.assertEqual(state.spawned, 0)
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), argv=['initial.pt', '5', *SMALL[2:]], lineage=lineage,
                                    stop_when=lambda state: True)
        self.assertIsNone(state.error)
        # A record that cannot be read leaves the generation unchecked.
        with tempfile.TemporaryDirectory() as temp:
            state = scripted_driver(Path(temp), argv=['initial.pt', '9', *SMALL[2:]],
                                    lineage={'models/initial.pt.lineage.json': ConnectionError('unavailable')},
                                    stop_when=lambda state: True)
        self.assertIsNone(state.error)
        self.assertTrue(any('could not read the lineage of initial.pt' in line for line in state.logs))


class CliDirectoryTests(unittest.TestCase):
    def setUp(self):
        class Image:
            def __getattr__(self, name):
                return lambda *args, **kwargs: self
        self.functions = {}
        owner = self
        class App:
            def function(self, **options):
                def decorate(fn):
                    owner.functions[fn.__name__] = fn
                    fn.remote = Mock(return_value=dict(exit=0, out='', stdout='', lines=[], err=''))
                    fn.spawn = Mock(return_value=SimpleNamespace(object_id='fc-submitted'))
                    return fn
                return decorate
            def local_entrypoint(self):
                return lambda fn: fn
        modal = ModuleType('modal')
        modal.App = lambda name: App()
        modal.Volume = SimpleNamespace(from_name=lambda *args, **kwargs: object())
        modal.Image = SimpleNamespace(debian_slim=lambda **kwargs: Image())
        with patch.dict(sys.modules, {'modal': modal}):
            loaded = runpy.run_path(str(ROOT / 'neural/modal_app.py'), run_name='cli_test')
        self.main = loaded['main']

    def invoke(self, task, **options):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.main(task, **options)

    def output_directory(self, task, *, spawn=False):
        name = 'selfplay_gpu' if task == 'selfplay-gpu' else task
        call = getattr(self.functions[name], 'spawn' if spawn else 'remote').call_args
        return call.args[4 if task == 'selfplay-gpu' else 6]

    def test_manual_selfplay_default_matches_remote_and_consumers(self):
        self.invoke('selfplay-gpu')
        self.assertEqual(self.output_directory('selfplay-gpu'), 'replay-gpu')
        default = inspect.signature(self.functions['selfplay_gpu']).parameters['out_subdir'].default
        self.assertEqual(self.output_directory('selfplay-gpu'), default)
        self.invoke('learn')
        self.assertEqual(self.functions['learn'].remote.call_args.kwargs['replay_subdir'], default)

    def test_exact_data_commands_retain_existing_defaults_in_both_modes(self):
        for task in ('dataset', 'prepare'):
            for spawn in (False, True):
                with self.subTest(task=task, spawn=spawn):
                    self.invoke(task, spawn=spawn)
                    self.assertEqual(self.output_directory(task, spawn=spawn), 'datasets')

    def test_explicit_output_directories_are_never_rewritten(self):
        for task in ('selfplay-gpu', 'dataset', 'prepare'):
            for directory in ('datasets', 'replay-gpu', 'experiment/replay', ''):
                for spawn in ((False,) if task == 'selfplay-gpu' else (False, True)):
                    with self.subTest(task=task, directory=directory, spawn=spawn):
                        self.invoke(task, out_subdir=directory, spawn=spawn)
                        self.assertEqual(self.output_directory(task, spawn=spawn), directory)

    def test_explicit_consumer_replay_directory_is_preserved(self):
        self.invoke('learn', replay_subdir='experiment/replay')
        self.assertEqual(self.functions['learn'].remote.call_args.kwargs['replay_subdir'], 'experiment/replay')

    def test_the_polled_functions_return_their_exceptions_as_failed_results(self):
        # Raised, a remote exception reached every poll of the driver as
        # itself: a TimeoutError read as "still running" and a connection
        # error as an outage, so a failed call held its slot for ever.
        cases = {'selfplay_gpu': lambda run: run('big1.pt', 0, 'all', 1),
                 'learn': lambda run: run(1, 'big0.pt', replay_window=-1),
                 'arena': lambda run: run('a.pt,b.pt', 'c.pt')}
        for name, invoke in cases.items():
            with self.subTest(function=name):
                result = invoke(self.functions[name])
                self.assertEqual(result['exit'], -1)
                self.assertIn('ValueError', result['err'])
                self.assertEqual((result['out'], result['lines']), ('', []))

    def test_arena_needs_two_checkpoints_before_a_container_starts(self):
        # Model B is --subdir, whose default names a table directory.
        for options in ({}, {'model': 'a.pt'}, {'subdir': 'b.pt'}, {'model': 'a.pt', 'subdir': 'chaos-4x4-c4'}):
            with self.subTest(options=options), self.assertRaisesRegex(SystemExit, '--subdir <b>.pt'):
                self.invoke('arena', **options)
        self.functions['arena'].remote.assert_not_called()
        self.invoke('arena', model='a.pt', subdir='b.pt')
        self.assertEqual(self.functions['arena'].remote.call_args.args[:2], ('a.pt', 'b.pt'))


if __name__ == '__main__':
    unittest.main()
