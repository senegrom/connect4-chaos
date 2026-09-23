"""CLI status/options and stop-file submission boundaries, without remote compute.

Execute the production entrypoint and driver bodies; only remote calls, the
clock and mirrors are replaced. No Modal deployment or GPU is needed.
"""
from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
import inspect
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .training_config import parse_shape_spec

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


TASKS = {
    "solve": "solve_8", "sidecars": "sidecars", "prepare": "prepare",
    "dataset": "dataset", "selfplay-gpu": "selfplay_gpu", "learn": "learn",
    "arena": "arena", "measure": "measure", "soup": "soup", "gpu-test": "gpu_test",
}
# Options a task cannot run without.
REQUIRED = {"gpu-test": {"args": ""}}


class EntrypointTests(unittest.TestCase):
    def entrypoint(self, code=0):
        self.payload = dict(exit=code, model="retained.pt", lines=["training completed"],
                            out="remote report", stdout="soup report", profile="profile report",
                            err="remote process failed" if code else "")
        self.remotes = {name: SimpleNamespace(remote=Mock(return_value=self.payload),
                          spawn=Mock(return_value=SimpleNamespace(object_id="fc-submitted")))
                        for name in set(TASKS.values()) | {"solve_32"}}
        namespace = dict(json=json, os=os, sys=sys, DEFAULT_SIMS=128, validate_selfplay=Mock(), **self.remotes)
        return function(ROOT / "neural/modal_app.py", "main", namespace)

    def test_every_synchronous_task_returns_normally_on_success(self):
        for task, name in TASKS.items():
            with self.subTest(task=task), redirect_stdout(io.StringIO()):
                self.assertIsNone(self.entrypoint()(task, **REQUIRED.get(task, {})))
                self.remotes[name].remote.assert_called_once()
                self.remotes[name].spawn.assert_not_called()

    def test_every_synchronous_task_propagates_nonzero_exit(self):
        for task, name in TASKS.items():
            for code in (1, 7, -9):
                with self.subTest(task=task, code=code), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        self.entrypoint(code)(task, **REQUIRED.get(task, {}))
                    self.assertEqual(caught.exception.code, code)
                    self.remotes[name].remote.assert_called_once()

    def test_failure_diagnostics_are_not_hidden_by_nonempty_stdout(self):
        for task in TASKS:
            stderr = io.StringIO()
            with self.subTest(task=task), redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                with self.assertRaises(SystemExit):
                    self.entrypoint(7)(task, **REQUIRED.get(task, {}))
                self.assertIn("remote process failed", stderr.getvalue())

    def test_gpu_test_requires_its_arguments_and_passes_an_empty_list_through(self):
        # The old default named a checkpoint that went with the Volume.
        entrypoint = self.entrypoint()
        with self.assertRaisesRegex(SystemExit, "--args"):
            entrypoint("gpu-test", module="test_search_history")
        self.remotes["gpu_test"].remote.assert_not_called()
        commands = []
        runner = function(ROOT / "neural/modal_app.py", "gpu_test", dict(
            TABLES="/tables", tables=SimpleNamespace(reload=Mock()), os=os,
            subprocess=SimpleNamespace(run=lambda command, **kwargs: commands.append(command) or
                                       SimpleNamespace(returncode=0, stdout="OK", stderr=""))))
        self.assertEqual(list(inspect.signature(runner).parameters.values())[1].default, inspect.Parameter.empty)
        for args, expected in (("", []), ("  ", []),
                               ("models/big504-808970a6d2.pt cuda 32", ["/tables/models/big504-808970a6d2.pt", "cuda", "32"])):
            with self.subTest(args=args), redirect_stdout(io.StringIO()):
                self.entrypoint()("gpu-test", module="test_search_history", args=args)
                self.assertEqual(self.remotes["gpu_test"].remote.call_args.args, ("test_search_history", args))
                runner("test_search_history", args)
                self.assertEqual(commands[-1], ["python", "-m", "neural.test_search_history", *expected])

    def test_retained_checkpoint_does_not_turn_failed_learner_into_success(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            self.entrypoint(7)("learn")
        self.assertEqual(caught.exception.code, 7)
        self.assertIn('"model": "retained.pt"', stdout.getvalue())
        self.assertIn("training completed", stdout.getvalue())
        self.assertIn("profile report", stdout.getvalue())
        self.assertEqual(self.payload["model"], "retained.pt")

    def test_large_solver_uses_same_exit_contract(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            self.entrypoint(7)("solve", threads=32)
        self.assertEqual(caught.exception.code, 7)
        self.remotes["solve_32"].remote.assert_called_once()
        self.remotes["solve_8"].remote.assert_not_called()

    def test_spawn_reports_submission_without_requiring_a_completion_result(self):
        for task in ("solve", "prepare", "dataset"):
            output = io.StringIO()
            with self.subTest(task=task), redirect_stdout(output):
                self.assertIsNone(self.entrypoint()(task, spawn=True))
            self.assertEqual(json.loads(output.getvalue())["spawned"], "fc-submitted")
            remote = self.remotes[TASKS[task]]
            remote.spawn.assert_called_once()
            remote.remote.assert_not_called()

    def test_submission_and_transport_exceptions_are_not_suppressed(self):
        for task in ("solve", "prepare", "dataset"):
            for method in ("spawn", "remote"):
                with self.subTest(task=task, method=method), self.assertRaisesRegex(OSError, "unavailable"):
                    entrypoint = self.entrypoint()
                    getattr(self.remotes[TASKS[task]], method).side_effect = OSError("unavailable")
                    entrypoint(task, spawn=method == "spawn")

    def test_soup_forwards_custom_replay_arguments(self):
        with redirect_stdout(io.StringIO()):
            self.entrypoint()("soup", models="a.pt,b.pt", out_name="mix.pt", batches=3,
                              replay_window=17, replay_subdir="experiment-only")
        self.remotes["soup"].remote.assert_called_once_with(
            "a.pt,b.pt", "mix.pt", 3, replay_window=17, exact_subdir="datasets-v3",
            replay_subdir="experiment-only")

    def test_local_holdouts_and_gzip_level_are_passed_as_arguments(self):
        # A Modal container does not inherit this environment: the entrypoint
        # reads these settings where they are set and passes them on.
        settings = dict(C4_REPLAY_GZIP_LEVEL="6", DISTILL_HOLDOUT_CONFIGS="4x4c3classic")
        expected = {"selfplay-gpu": ("selfplay_gpu", "gzip_level", 6),
                    "learn": ("learn", "holdout_configs", "4x4c3classic"),
                    "measure": ("measure", "holdout_configs", "4x4c3classic")}
        for task, (name, option, value) in expected.items():
            with self.subTest(task=task), patch.dict(os.environ, settings), redirect_stdout(io.StringIO()):
                self.entrypoint()(task)
                self.assertEqual(self.remotes[name].remote.call_args.kwargs[option], value)
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            self.entrypoint()("selfplay-gpu")
            self.assertEqual(self.remotes["selfplay_gpu"].remote.call_args.kwargs["gzip_level"], 1)
            self.entrypoint()("learn")
            self.assertEqual(self.remotes["learn"].remote.call_args.kwargs["holdout_configs"], "")

    def test_exact_corpus_option_reaches_every_task_that_reads_it(self):
        for task in ("learn", "measure", "soup"):
            with self.subTest(task=task), redirect_stdout(io.StringIO()):
                self.entrypoint()(task, exact_subdir="exact/v4", allow_no_exact=True)
                call = self.remotes[task].remote.call_args
                self.assertEqual(call.kwargs["exact_subdir"], "exact/v4")
                if task == "learn":
                    self.assertIs(call.kwargs["allow_no_exact"], True)
        with redirect_stdout(io.StringIO()):
            self.entrypoint()("learn")
        call = self.remotes["learn"].remote.call_args
        self.assertEqual((call.kwargs["exact_subdir"], call.kwargs["allow_no_exact"]), ("datasets-v3", False))

    def test_omitted_windows_preserve_the_distinct_remote_defaults(self):
        for task, expected in (("learn", 4_000_000), ("soup", 400_000)):
            with self.subTest(task=task), redirect_stdout(io.StringIO()):
                self.entrypoint()(task)
                call = self.remotes[task].remote.call_args
                actual = call.args[6] if task == "learn" else call.kwargs["replay_window"]
                self.assertEqual(actual, expected)
                self.assertEqual(call.kwargs["replay_subdir"], "replay-gpu")
                # The CLI default must stay aligned with the actual remote signature.
                wrapper = function(ROOT / "neural/modal_app.py", task, {})
                self.assertEqual(inspect.signature(wrapper).parameters["replay_window"].default, expected)

    def test_explicit_learner_zero_is_not_replaced_by_default(self):
        with redirect_stdout(io.StringIO()):
            self.entrypoint()("learn", replay_window=0, replay_subdir="custom")
        call = self.remotes["learn"].remote.call_args
        self.assertEqual(call.args[6], 0)
        self.assertEqual(call.kwargs["replay_subdir"], "custom")

    def test_invalid_replay_windows_fail_before_remote_submission(self):
        for task in ("learn", "soup"):
            for window in (-1, True, 1.5, "17") + ((0,) if task == "soup" else ()):
                with self.subTest(task=task, window=window), redirect_stdout(io.StringIO()):
                    with self.assertRaises(ValueError):
                        self.entrypoint()(task, replay_window=window)
                    self.remotes[task].remote.assert_not_called()

    def test_unknown_task_still_fails_without_remote_work(self):
        # "closure" measured a winning-strategy closure whose inputs went with
        # the Volume; the task was removed with neural/winning_closure.py.
        for task in ("unknown", "closure"):
            with self.subTest(task=task), self.assertRaisesRegex(SystemExit, "unknown task"):
                self.entrypoint()(task)
            for remote in self.remotes.values():
                remote.remote.assert_not_called()
                remote.spawn.assert_not_called()


class ShutdownTests(unittest.TestCase):
    def run_driver(self, phase, *, mirror=False, remove_stop=False, transient=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stop = root / "stop"
            logs, spawns, jobs, completed, sleeps = [], [], [], [], []
            requested = False

            def request_stop():
                nonlocal requested
                requested = True
                stop.write_text("stop", encoding="utf-8")

            def event(name):
                if name == phase and not requested:
                    request_stop()

            class Call:
                def __init__(self, kind, args):
                    self.kind, self.args, self.polls = kind, args, 0
                    self.object_id = f"fc-{kind}-{len(jobs)}"
                def get(self, timeout=0):
                    self.polls += 1
                    if self.polls == 1:
                        raise TimeoutError
                    event(f"poll:{self.kind}")
                    if transient and self.kind == "learner" and self.polls == 2:
                        raise ConnectionError("transient")
                    # Keep one job alive after shutdown is observed to test the latch.
                    if remove_stop and self.kind == "actor" and self.polls < 4:
                        raise TimeoutError
                    completed.append(self.object_id)
                    if self.kind == "learner":
                        return dict(exit=0, model=f"big{self.args[0]}-abc.pt", seconds=1, lines=[])
                    if self.kind == "actor":
                        return dict(exit=0, shard=f"{self.object_id}.pt.gz", seconds=1,
                                    out="self-play: 1 games, 20 positions", shard_bytes=1000)
                    return dict(exit=0, out="arena completed")

            def remote(kind):
                def spawn(*args, **kwargs):
                    spawns.append((kind, requested))
                    event(f"spawn:{kind}")
                    if phase == "backoff" and kind == "learner":
                        raise RuntimeError("failed to submit learner")
                    call = Call(kind, args)
                    jobs.append(call)
                    return call
                return SimpleNamespace(spawn=spawn)

            def sleep(seconds):
                sleeps.append(seconds)
                if len(sleeps) > 30:
                    raise AssertionError("Driver failed to drain and exit")
                event("backoff" if seconds == 60 else "sleep")
                if remove_stop and requested:
                    stop.unlink(missing_ok=True)

            def mirror_model(name):
                event("mirror:model")
                return root / name

            def fetch_shard(name):
                event("mirror:shard")
                return root / name, 1000

            env = dict(GAMES=1, SIMS=1, SHAPES="all", TARGET_SIMS=0, TARGET_SHARE=.25,
                K=2, STEPS=1, BATCH=1, WINDOW=20, MIN_NEW=1000, ARENA_EVERY=5, ARENA_LAG=5,
                LR=.0004, MIRROR=mirror, ROOT=root, REPLAY=root / "replay", STOP=stop,
                INIT_MODEL="big4-abc.pt", GEN=5, ENTROPY_BONUS=0, Q_SEED=True,
                REPLAY_FRACTION=.75, POLICY_TARGET="visits", ROOT_VALUE_WEIGHT=0,
                EXACT_SUBDIR="datasets-v3", GZIP_LEVEL=1, HOLDOUT_CONFIGS="",
                parse_shape_spec=parse_shape_spec,
                OUT_SUBDIR="replay-gpu", ARENA_GAMES=6, ARENA_SIMS=32, re=re,
                validate_selfplay=Mock(), log=logs.append,
                published_history=lambda: [f"big{n}-abc.pt" for n in range(5)],
                learn_fn=remote("learner"), actor_fn=remote("actor"), arena_fn=remote("arena"),
                is_transient=lambda exc: isinstance(exc, ConnectionError),
                mirror_model=mirror_model, fetch_shard=fetch_shard,
                with_timeout=lambda seconds, work, *args: work(*args),
                time=SimpleNamespace(time=lambda: 1234, sleep=sleep))
            event("startup")
            function(ROOT / "neural/modal_loop.py", "main", env)()
            self.assertTrue(requested, "scenario never requested shutdown")
            self.assertFalse([kind for kind, after_stop in spawns if after_stop],
                             "submitted new work after shutdown was requested")
            self.assertCountEqual(completed, [job.object_id for job in jobs],
                                  "in-flight work was not fully collected")
            self.assertTrue(logs[-1].startswith("loop end:"))
            if any(job.kind == "learner" for job in jobs):
                self.assertIn("next gen 6", logs[-1])
                self.assertIn("model big5-abc.pt", logs[-1])
            return [kind for kind, _ in spawns], logs

    def test_already_stopped_driver_submits_nothing(self):
        self.assertEqual(self.run_driver("startup")[0], [])

    def test_stop_between_iterations_drains_without_new_arena(self):
        kinds, _ = self.run_driver("sleep")
        self.assertNotIn("arena", kinds)

    def test_stop_during_learner_poll_is_checked_in_the_same_iteration(self):
        self.assertNotIn("arena", self.run_driver("poll:learner")[0])

    def test_stop_during_actor_poll_is_checked_before_arena(self):
        self.assertNotIn("arena", self.run_driver("poll:actor")[0])

    def test_stop_during_each_mirror_is_checked_before_arena(self):
        for phase in ("mirror:model", "mirror:shard"):
            with self.subTest(phase=phase):
                self.assertNotIn("arena", self.run_driver(phase, mirror=True)[0])

    def test_stop_during_learner_submission_prevents_actor_submissions(self):
        self.assertEqual(self.run_driver("spawn:learner")[0], ["learner"])

    def test_stop_during_actor_submission_prevents_filling_remaining_slots(self):
        self.assertEqual(self.run_driver("spawn:actor")[0], ["learner", "actor"])

    def test_stop_during_failed_submission_backoff_prevents_retry_or_actors(self):
        self.assertEqual(self.run_driver("backoff")[0], ["learner"])

    def test_preexisting_arena_is_collected_not_cancelled(self):
        kinds, logs = self.run_driver("spawn:arena")
        self.assertEqual(kinds.count("arena"), 1)
        self.assertTrue(any("arena completed" in line for line in logs))

    def test_observed_shutdown_stays_latched_if_file_is_removed(self):
        self.assertNotIn("arena", self.run_driver("poll:learner", remove_stop=True)[0])

    def test_transient_poll_failure_during_shutdown_keeps_job_tracked(self):
        kinds, logs = self.run_driver("poll:learner", transient=True)
        self.assertEqual(kinds.count("learner"), 1)
        self.assertNotIn("arena", kinds)
        self.assertTrue(any("still tracked" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
