"""What the CPU tests share: a Modal function's body run locally, and the
loop driver run against a scripted Modal."""
# function() compiles with this module's future flags: without this one,
# Python 3.12 evaluates an extracted function's annotations when it is
# defined, and modal_app.main's Optional[...] meets a namespace without it.
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("actor", "learner", "arena")
SMALL = ["initial.pt", "1", "1", "1", "10", "64", "4e-4", "4000000", "0", "64", "1", "1"]   # K=1, arena every gen, lag 1
FULL = SMALL + ["all", "0", "0.25", "0", "1", "0.75", "visits", "0", "datasets-v3"]


def function(path, name, namespace):
    """The function `name` of the module at `path`, without its decorators,
    defined in `namespace`, which supplies the globals it reads."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def container_paths(root, prefixes=("/tmp/replay-", "/tmp/learn-")):
    """A stand-in for Path that keeps a container's scratch under `root`."""
    def local_path(path):
        path = str(path)
        return root / path.lstrip("/") if path.startswith(prefixes) else Path(path)
    return local_path


def sdk_exceptions():
    """modal.exception's hierarchy: its TimeoutError is not Python's."""
    exceptions = ModuleType("modal.exception")
    exceptions.Error = type("Error", (Exception,), {})
    exceptions.TimeoutError = type("TimeoutError", (exceptions.Error,), {})
    for name in ("FunctionTimeoutError", "OutputExpiredError"):
        setattr(exceptions, name, type(name, (exceptions.TimeoutError,), {}))
    for name in ("RemoteError", "ExecutionError", "InternalFailure", "DeserializationError", "ServiceError",
                 "InternalError", "ResourceExhaustedError", "ConnectionError", "NotFoundError"):
        setattr(exceptions, name, type(name, (exceptions.Error,), {}))
    return exceptions


def volume_entries(path):
    """A Volume with an exact corpus and a replay window, which every driver
    lists before its first spawn; any other path lists as itself."""
    shards = {"datasets-v3": ("classic-4x4-c4-0000.pt", "classic-4x4-c4-0001.pt"),
              "replay-gpu": ("gpu-sp-1-1.pt.gz",)}
    if path not in shards:
        return [SimpleNamespace(path=path, size=1, mtime=0)]
    return [SimpleNamespace(path=f"{path}/{name}", size=1, mtime=0) for name in shards[path]]


# What each role's call returns unless a script says otherwise.
OUTCOMES = {
    "actor": lambda call: dict(exit=0, shard=f"{call.object_id}.pt.gz", seconds=1,
                               out="self-play: 1 games, 100 positions", shard_bytes=1),
    "learner": lambda call: dict(exit=0, model=f"big{call.args[0]}-ok.pt", seconds=1, lines=[]),
    "arena": lambda call: dict(exit=0, out="arena completed"),
}


def scripted_driver(root, *, argv=SMALL, script=None, spawn_errors=None, env=None, journal=None,
                    restored_args=None, listing=None, lineage=None, real_history=False, stop_when=None,
                    crash_after=None, max_ticks=300, overrides=None, clock=False, cancel_errors=None,
                    exceptions=None):
    """Run the real driver module with Modal replaced by scripted calls.

    script[kind](call) is a call's outcome, asked on its second and every
    later poll: a result dict, or an exception to raise. spawn_errors[kind](attempt)
    may return an exception for that spawn attempt. `journal` is written
    first; `listing(path, exceptions)` answers the preflight (volume_entries
    by default). `lineage` maps sidecar paths - or is a function of one - to
    records or exceptions; there are none by default, so the initial model
    is a root, and the driver's history is that model alone unless
    `real_history` lets it read its own. stop_when(state) asks for a stop
    from inside the fake sleep; crash_after=n makes the nth sleep raise
    KeyboardInterrupt, a driver killed mid-loop. `overrides` replace module
    globals, the harness's own among them, and with `clock` time advances by
    every sleep. cancel_errors[kind] is an exception every cancel of that
    kind raises. An `env` value of None unsets that variable; `exceptions` is
    the modal.exception the driver sees.

    A spawn after the stop file exists, or a blocking poll, fails the run
    even when the driver catches it. state.spawns lists every spawn attempt
    in order, state.collected the calls whose results were read, and
    state.reads every sidecar read.
    """
    exceptions = exceptions or sdk_exceptions()
    stop = root / "modal-loop.stop"
    state = SimpleNamespace(calls={kind: [] for kind in ROLES}, attempts=dict.fromkeys(ROLES, 0),
                            spawns=[], logs=[], ticks=[], removed=[], cancelled=[], restored={},
                            error=None, reads=[], collected=[], violations=[])
    outcomes = {**OUTCOMES, **(script or {})}

    def violation(message):
        state.violations.append(message)
        return AssertionError(message)

    class Call:
        def __init__(self, kind, args, kwargs, object_id, index):
            self.kind, self.args, self.kwargs, self.object_id, self.index = kind, args, kwargs, object_id, index
            self.polls = 0

        def get(self, timeout=None):
            if timeout != 0:
                raise violation(f"{self.object_id} polled with timeout={timeout!r}, not 0")
            self.polls += 1
            if self.polls < 2:
                raise TimeoutError
            outcome = outcomes[self.kind](self)
            if isinstance(outcome, BaseException):
                raise outcome
            state.collected.append(self.object_id)
            return outcome

        def cancel(self, terminate_containers=False):
            error = (cancel_errors or {}).get(self.kind)
            if error is not None:
                raise error
            state.cancelled.append(self.object_id)

    def remote(kind):
        def spawn(*args, **kwargs):
            if stop.exists():
                raise violation(f"{kind} submitted after the stop was requested")
            state.attempts[kind] += 1
            state.spawns.append(kind)
            error = (spawn_errors or {}).get(kind, lambda attempt: None)(state.attempts[kind])
            if error is not None:
                raise error
            call = Call(kind, args, kwargs, f"fc-{kind}-{len(state.calls[kind])}", len(state.calls[kind]))
            state.calls[kind].append(call)
            return call
        return SimpleNamespace(spawn=spawn)

    def from_id(cid):
        call = Call(cid.split("-")[1], (restored_args or {}).get(cid, ()), {}, cid, -1)
        state.restored[cid] = call
        return call

    def listdir(path):
        return listing(path, exceptions) if listing else volume_entries(path)

    def read_file(path):
        state.reads.append(path)
        record = lineage(path) if callable(lineage) else (lineage or {}).get(path)
        if record is None:
            raise FileNotFoundError(path)
        if isinstance(record, BaseException):
            raise record
        return [json.dumps(record).encode()]

    volume = SimpleNamespace(listdir=Mock(side_effect=listdir), read_file=read_file,
                             remove_file=Mock(side_effect=state.removed.append))
    modal = ModuleType("modal")
    modal.exception = exceptions
    functions = {name: remote(kind) for name, kind in
                 (("selfplay_gpu", "actor"), ("learn", "learner"), ("arena", "arena"))}
    modal.Function = SimpleNamespace(from_name=lambda app, name: functions[name])
    modal.Volume = SimpleNamespace(from_name=lambda name: volume)
    modal.FunctionCall = SimpleNamespace(from_id=from_id)
    journal_path = root / "modal-loop.calls.json"
    if journal is not None:
        journal_path.write_text(journal if isinstance(journal, str) else json.dumps(journal), encoding="utf-8")

    def sleep(seconds):
        state.ticks.append(seconds)
        if crash_after is not None and len(state.ticks) >= crash_after:
            raise KeyboardInterrupt
        if len(state.ticks) > max_ticks:
            raise AssertionError("the driver did not stop")
        if stop_when is not None and stop_when(state):
            stop.touch()

    environment = {"C4_NEURAL_ROOT": str(root), "C4_MIRROR": "0", **(env or {})}
    with patch.dict(sys.modules, {"modal": modal, "modal.exception": exceptions}), \
            patch.dict(os.environ, {k: v for k, v in environment.items() if v is not None}, clear=True), \
            patch.object(sys, "argv", ["modal_loop.py", *argv]):
        loaded = runpy.run_path(str(ROOT / "neural/modal_loop.py"), run_name="driver_test")
        module = loaded["main"].__globals__
        now = (lambda: 1234 + sum(state.ticks)) if clock else (lambda: 1234)
        harness = dict(log=state.logs.append, time=SimpleNamespace(time=now, sleep=sleep))
        if not real_history:
            harness["published_history"] = lambda: [module["INIT_MODEL"]]
        module.update(harness, **(overrides or {}))
        state.module = module
        try:
            module["main"]()
        except BaseException as exc:            # the tests inspect how the driver ended
            state.error = exc
    if state.violations:
        raise AssertionError("; ".join(state.violations))
    state.volume = volume
    state.journal_text = journal_path.read_text(encoding="utf-8") if journal_path.exists() else None
    try:
        state.journal = json.loads(state.journal_text or "null")
    except ValueError:
        state.journal = None
    state.spawned = sum(len(calls) for calls in state.calls.values())
    return state
