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

ROOT = Path(__file__).resolve().parents[1]
ROLES = ('actor', 'learner', 'arena')


def sdk_exceptions():
    exceptions = ModuleType('modal.exception')
    exceptions.Error = type('Error', (Exception,), {})
    exceptions.TimeoutError = type('TimeoutError', (exceptions.Error,), {})
    for name in ('FunctionTimeoutError', 'OutputExpiredError'):
        setattr(exceptions, name, type(name, (exceptions.TimeoutError,), {}))
    for name in ('RemoteError', 'ExecutionError', 'InternalFailure', 'DeserializationError', 'ServiceError'):
        setattr(exceptions, name, type(name, (exceptions.Error,), {}))
    return exceptions


def volume_entries(path):
    """A Volume with an exact corpus and a replay window, which every driver
    lists before its first spawn; any other path lists as itself."""
    shards = {'datasets-v3': ('classic-4x4-c4-0000.pt', 'classic-4x4-c4-0001.pt'),
              'replay-gpu': ('gpu-sp-1-1.pt.gz',)}
    if path not in shards:
        return [SimpleNamespace(path=path, size=1, mtime=0)]
    return [SimpleNamespace(path=f'{path}/{name}', size=1, mtime=0) for name in shards[path]]


def no_lineage(path):
    raise FileNotFoundError(path)


class DriverPollingTests(unittest.TestCase):
    def run_driver(self, role, error_name, *, stopping):
        exceptions = sdk_exceptions()
        error_type = getattr(exceptions, error_name, None)
        if error_type is None:
            error_type = {'pending': TimeoutError, 'connection': ConnectionError}[error_name]
        terminal = error_name not in ('pending', 'connection', 'ServiceError')
        error = error_type('connection lost: deadline timed out; service unavailable')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stop = root / 'modal-loop.stop'
            calls = {kind: [] for kind in ROLES}
            logs, ticks = [], []
            failed = []

            class Call:
                def __init__(self, kind, args):
                    self.kind, self.args, self.polls = kind, args, 0
                    self.object_id = f'fc-{kind}-{len(calls[kind])}'

                def get(self, timeout=None):
                    if timeout != 0:
                        raise AssertionError('driver stopped using nonblocking polling')
                    self.polls += 1
                    if self.kind == role and self is calls[role][0]:
                        # Terminal results stay failed forever; transient failures
                        # and empty polls must be collected from this SAME call.
                        if terminal or self.polls == 1:
                            failed.append(self.object_id)
                            if stopping:
                                stop.touch()
                            raise error
                        if not stopping:
                            stop.touch()
                    elif self.kind == role and terminal and not stopping:
                        stop.touch()  # a replacement was safely submitted
                    if self.kind == 'learner':
                        return dict(exit=0, model=f'big{self.args[0]}-ok.pt', seconds=1, lines=[])
                    if self.kind == 'actor':
                        return dict(exit=0, shard=f'{self.object_id}.pt.gz', seconds=1,
                                    out='self-play: 1 games, 100 positions', shard_bytes=1)
                    return dict(exit=0, out='arena completed')

            def remote(kind):
                def spawn(*args, **kwargs):
                    if stop.exists():
                        raise AssertionError('submitted work after stop was requested')
                    call = Call(kind, args)
                    calls[kind].append(call)
                    return call
                return SimpleNamespace(spawn=spawn)

            modal = ModuleType('modal')
            modal.exception = exceptions
            functions = {name: remote(kind) for name, kind in
                         (('selfplay_gpu', 'actor'), ('learn', 'learner'), ('arena', 'arena'))}
            modal.Function = SimpleNamespace(from_name=lambda app, name: functions[name])
            modal.Volume = SimpleNamespace(from_name=lambda name: SimpleNamespace(
                listdir=volume_entries, read_file=no_lineage))
            argv = ['modal_loop.py', 'initial.pt', '1', '1', '1']
            with patch.dict(sys.modules, {'modal': modal, 'modal.exception': exceptions}), \
                    patch.dict(os.environ, {'C4_NEURAL_ROOT': str(root), 'C4_MIRROR': '0'}), \
                    patch.object(sys, 'argv', argv):
                loaded = runpy.run_path(str(ROOT / 'neural/modal_loop.py'), run_name='driver_test')
                state = loaded['main'].__globals__
                def sleep(seconds):
                    ticks.append(seconds)
                    if len(ticks) > 20:
                        raise AssertionError('driver repeatedly polled a completed failure')
                state.update(log=logs.append, published_history=lambda: ['initial.pt'],
                             ARENA_EVERY=1 if role == 'arena' else 0, ARENA_LAG=1, MIN_NEW=0,
                             time=SimpleNamespace(time=lambda: 1234, sleep=sleep))
                state['main']()
            self.assertTrue(stop.exists())
            self.assertTrue(logs[-1].startswith('loop end:'))
            self.assertEqual(len(failed), 1, 'polled the same completed failure again')
            if terminal:
                self.assertEqual(calls[role][0].polls, 1)
                self.assertEqual(len(calls[role]), 1 if stopping else 2)
                self.assertTrue(any(f'{role} ' in line and 'failed:' in line for line in logs))
                if role == 'learner':
                    self.assertTrue(all(call.args[1] == 'initial.pt' for call in calls['learner']))
                    self.assertTrue(all(call.args[0] == 1 for call in calls['learner']))
                    if stopping:
                        self.assertIn('next gen 1, model initial.pt', logs[-1])
            else:
                self.assertEqual(len(calls[role]), 1, 'duplicated a pending or uncertain job')
                self.assertEqual(calls[role][0].polls, 2)
                if error_name != 'pending':
                    self.assertTrue(any('still tracked' in line for line in logs))

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
            for name in ('pending', 'connection', 'ServiceError'):
                for stopping in (False, True):
                    with self.subTest(role=role, error=name, stopping=stopping):
                        self.run_driver(role, name, stopping=stopping)

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


SMALL = ['initial.pt', '1', '1', '1', '10', '64', '4e-4', '4000000', '0', '64', '1', '1']   # K=1, arena every gen, lag 1
FULL = SMALL + ['all', '0', '0.25', '0', '1', '0.75', 'visits', '0', 'datasets-v3']


def scripted_driver(root, *, argv=SMALL, script=None, spawn_errors=None, env=None, journal=None,
                    restored_args=None, listing=None, lineage=None, stop_when=None, crash_after=None,
                    max_ticks=300, overrides=None, clock=False):
    """Run the real driver module with Modal replaced by scripted calls.

    script[kind](call) is a call's outcome, returned on its second and every
    later poll: a result dict, or an exception to raise. spawn_errors[kind](attempt) may
    return an exception for that spawn attempt. `journal` is written first;
    `listing(path, exceptions)` answers the preflight (volume_entries by
    default), and `lineage` maps sidecar paths to records or to exceptions
    (none by default: the initial model is a root). stop_when(state) asks
    for a stop from inside the fake sleep; crash_after=n makes the nth sleep
    raise KeyboardInterrupt, a driver killed mid-loop. `overrides` replace
    module globals, and with `clock` time advances by every sleep.
    """
    exceptions = sdk_exceptions()
    exceptions.NotFoundError = type('NotFoundError', (exceptions.Error,), {})
    stop = root / 'modal-loop.stop'
    state = SimpleNamespace(calls={kind: [] for kind in ROLES}, attempts=dict.fromkeys(ROLES, 0),
                            logs=[], ticks=[], removed=[], cancelled=[], restored={}, error=None)
    outcomes = {
        'actor': lambda call: dict(exit=0, shard=f'{call.object_id}.pt.gz', seconds=1,
                                   out='self-play: 1 games, 100 positions', shard_bytes=1),
        'learner': lambda call: dict(exit=0, model=f'big{call.args[0]}-ok.pt', seconds=1, lines=[]),
        'arena': lambda call: dict(exit=0, out='arena completed'),
    }
    outcomes.update(script or {})

    class Call:
        def __init__(self, kind, args, kwargs, object_id, index):
            self.kind, self.args, self.kwargs, self.object_id, self.index = kind, args, kwargs, object_id, index
            self.polls = 0

        def get(self, timeout=None):
            self.polls += 1
            if self.polls < 2:
                raise TimeoutError
            outcome = outcomes[self.kind](self)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        def cancel(self, terminate_containers=False):
            state.cancelled.append(self.object_id)

    def remote(kind):
        def spawn(*args, **kwargs):
            if stop.exists():
                raise AssertionError('submitted work after stop was requested')
            state.attempts[kind] += 1
            error = (spawn_errors or {}).get(kind, lambda attempt: None)(state.attempts[kind])
            if error is not None:
                raise error
            call = Call(kind, args, kwargs, f'fc-{kind}-{len(state.calls[kind])}', len(state.calls[kind]))
            state.calls[kind].append(call)
            return call
        return SimpleNamespace(spawn=spawn)

    def from_id(cid):
        call = Call(cid.split('-')[1], (restored_args or {}).get(cid, ()), {}, cid, -1)
        state.restored[cid] = call
        return call

    def listdir(path):
        return listing(path, exceptions) if listing else volume_entries(path)

    def read_file(path):
        record = (lineage or {}).get(path)
        if record is None:
            raise FileNotFoundError(path)
        if isinstance(record, BaseException):
            raise record
        return [json.dumps(record).encode()]

    volume = SimpleNamespace(listdir=Mock(side_effect=listdir), read_file=read_file,
                             remove_file=Mock(side_effect=state.removed.append))
    modal = ModuleType('modal')
    modal.exception = exceptions
    functions = {name: remote(kind) for name, kind in
                 (('selfplay_gpu', 'actor'), ('learn', 'learner'), ('arena', 'arena'))}
    modal.Function = SimpleNamespace(from_name=lambda app, name: functions[name])
    modal.Volume = SimpleNamespace(from_name=lambda name: volume)
    modal.FunctionCall = SimpleNamespace(from_id=from_id)
    journal_path = root / 'modal-loop.calls.json'
    if journal is not None:
        journal_path.write_text(journal if isinstance(journal, str) else json.dumps(journal), encoding='utf-8')

    def sleep(seconds):
        state.ticks.append(seconds)
        if crash_after is not None and len(state.ticks) >= crash_after:
            raise KeyboardInterrupt
        if len(state.ticks) > max_ticks:
            raise AssertionError('the driver did not stop')
        if stop_when is not None and stop_when(state):
            stop.touch()

    with patch.dict(sys.modules, {'modal': modal, 'modal.exception': exceptions}), \
            patch.dict(os.environ, {'C4_NEURAL_ROOT': str(root), 'C4_MIRROR': '0', **(env or {})}, clear=True), \
            patch.object(sys, 'argv', ['modal_loop.py', *argv]):
        loaded = runpy.run_path(str(ROOT / 'neural/modal_loop.py'), run_name='driver_test')
        module = loaded['main'].__globals__
        now = (lambda: 1234 + sum(state.ticks)) if clock else (lambda: 1234)
        module.update(log=state.logs.append, published_history=lambda: [module['INIT_MODEL']],
                      time=SimpleNamespace(time=now, sleep=sleep), **(overrides or {}))
        try:
            module['main']()
        except BaseException as exc:            # the tests inspect how the driver ended
            state.error = exc
    state.volume = volume
    state.journal_text = journal_path.read_text(encoding='utf-8') if journal_path.exists() else None
    try:
        state.journal = json.loads(state.journal_text or 'null')
    except ValueError:
        state.journal = None
    state.spawned = sum(len(calls) for calls in state.calls.values())
    return state


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
        bad = {1: ('initial', 'models/initial.pt'), 2: ('-1',), 16: ('-0.5', 'inf'), 17: ('2', 'yes'),
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

    def test_restart_reattaches_journaled_calls_and_cancels_a_stale_learner(self):
        journal = {'version': 1, 'calls': [
            dict(id='fc-actor-old0', role='actor', seed=5, model='big0-a.pt', spawned=1),
            dict(id='fc-learner-old', role='learner', gen=1, init='initial.pt', spawned=1),
            dict(id='fc-learner-stale', role='learner', gen=4, init='big3-b.pt', spawned=1),
            dict(id='fc-arena-old', role='arena', newer='big0-a.pt', older='seed.pt')]}
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
        for text in ('not json', '[]', '{"version": 2, "calls": []}', '{"version": 1, "calls": [{"role": "actor"}]}'):
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

    def test_cli_failure_status_is_unchanged(self):
        for task, name in (('selfplay-gpu', 'selfplay_gpu'), ('dataset', 'dataset'), ('prepare', 'prepare')):
            with self.subTest(task=task):
                self.functions[name].remote.return_value['exit'] = 7
                with self.assertRaises(SystemExit) as caught:
                    self.invoke(task)
                self.assertEqual(caught.exception.code, 7)


if __name__ == '__main__':
    unittest.main()
