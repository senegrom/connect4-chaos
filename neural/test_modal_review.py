"""Post-merge driver/CLI regressions; no Modal account, GPU or network needed.

Load both complete production modules. Only the Modal boundary, clock and
lineage reader are mocked. The timeout hierarchy mirrors the public SDK:
modal.exception.TimeoutError is distinct from Python's polling TimeoutError.
"""
from contextlib import redirect_stderr, redirect_stdout
import inspect
import io
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
            modal.Volume = SimpleNamespace(from_name=lambda name: object())
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

    def test_terminal_failure_releases_slot_for_replacement_before_shutdown(self):
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
        for task in ('learn', 'soup'):
            self.invoke(task)
            self.assertEqual(self.functions[task].remote.call_args.kwargs['replay_subdir'], default)

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
        for task in ('learn', 'soup'):
            self.invoke(task, replay_subdir='experiment/replay')
            self.assertEqual(self.functions[task].remote.call_args.kwargs['replay_subdir'], 'experiment/replay')

    def test_cli_failure_status_is_unchanged(self):
        for task, name in (('selfplay-gpu', 'selfplay_gpu'), ('dataset', 'dataset'), ('prepare', 'prepare')):
            with self.subTest(task=task):
                self.functions[name].remote.return_value['exit'] = 7
                with self.assertRaises(SystemExit) as caught:
                    self.invoke(task)
                self.assertEqual(caught.exception.code, 7)


if __name__ == '__main__':
    unittest.main()
