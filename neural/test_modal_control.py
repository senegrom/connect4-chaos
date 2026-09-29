"""CLI status/options and stop-file submission boundaries, without remote compute.

Execute the production entrypoint and driver bodies; only remote calls, the
clock and mirrors are replaced. No Modal deployment or GPU is needed.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .test_support import OUTCOMES, ROLES, ROOT, function, scripted_driver


TASKS = {
    "solve": "solve_8", "prepare": "prepare",
    "dataset": "dataset", "selfplay-gpu": "selfplay_gpu", "learn": "learn",
    "arena": "arena", "measure": "measure", "gpu-test": "gpu_test",
}
# Options a task cannot run without.
REQUIRED = {"gpu-test": {"args": ""}, "arena": {"model": "a.pt", "subdir": "b.pt"}}


class EntrypointTests(unittest.TestCase):
    def entrypoint(self, code=0):
        self.payload = dict(exit=code, model="retained.pt", lines=["training completed"],
                            out="remote report", profile="profile report",
                            err="remote process failed" if code else "")
        self.remotes = {name: SimpleNamespace(remote=Mock(return_value=self.payload),
                          spawn=Mock(return_value=SimpleNamespace(object_id="fc-submitted")))
                        for name in set(TASKS.values()) | {"solve_32"}}
        namespace = dict(json=json, os=os, sys=sys, DEFAULT_SIMS=128, ARENA_GAMES=6, ARENA_SHAPES="all",
                         ARENA_SIMS=32, validate_selfplay=Mock(), **self.remotes)
        return function(ROOT / "neural/modal_app.py", "main", namespace)

    def test_every_synchronous_task_returns_normally_on_success(self):
        for task, name in TASKS.items():
            with self.subTest(task=task), redirect_stdout(io.StringIO()):
                self.assertIsNone(self.entrypoint()(task, **REQUIRED.get(task, {})))
                self.remotes[name].remote.assert_called_once()
                self.remotes[name].spawn.assert_not_called()

    def test_a_manual_arena_plays_the_loops_arena_unless_told_otherwise(self):
        # It took self-play's defaults - two 6x7 boards, 256 games at 128
        # simulations - where the loop's arena plays every board.
        with redirect_stdout(io.StringIO()):
            self.entrypoint()("arena", model="a.pt", subdir="b.pt")
            self.remotes["arena"].remote.assert_called_once_with("a.pt", "b.pt", 6, 32, "all", 1, -1)
            self.entrypoint()("arena", model="a.pt", subdir="b.pt", games=2, sims=8, shapes="6x7c4chaos")
            self.remotes["arena"].remote.assert_called_once_with("a.pt", "b.pt", 2, 8, "6x7c4chaos", 1, -1)
            self.entrypoint()("selfplay-gpu")
            self.assertEqual(self.remotes["selfplay_gpu"].remote.call_args.args[1:3],
                             (256, "6x7c4chaos,6x7c4classic"))

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
        for task in ("learn", "measure"):
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

    def test_an_omitted_window_keeps_the_remote_default(self):
        with redirect_stdout(io.StringIO()):
            self.entrypoint()("learn")
        call = self.remotes["learn"].remote.call_args
        self.assertEqual(call.args[6], 4_000_000)
        self.assertEqual(call.kwargs["replay_subdir"], "replay-gpu")
        # The CLI default must stay aligned with the actual remote signature.
        wrapper = function(ROOT / "neural/modal_app.py", "learn", {})
        self.assertEqual(inspect.signature(wrapper).parameters["replay_window"].default, 4_000_000)

    def test_explicit_learner_zero_is_not_replaced_by_default(self):
        with redirect_stdout(io.StringIO()):
            self.entrypoint()("learn", replay_window=0, replay_subdir="custom")
        call = self.remotes["learn"].remote.call_args
        self.assertEqual(call.args[6], 0)
        self.assertEqual(call.kwargs["replay_subdir"], "custom")

    def test_invalid_replay_windows_fail_before_remote_submission(self):
        for window in (-1, True, 1.5, "17"):
            with self.subTest(window=window), redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    self.entrypoint()("learn", replay_window=window)
                self.remotes["learn"].remote.assert_not_called()

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
        """Request a stop at `phase` - startup, a spawn, poll or mirror of a
        role, the next sleep or a submission's backoff - and check that the
        driver drains: nothing submitted after it, everything collected."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stop = root / "modal-loop.stop"
            requested = False

            def event(name):
                nonlocal requested
                if name == phase and not requested:
                    requested = True
                    stop.write_text("stop", encoding="utf-8")

            def poll(kind):
                def outcome(call):
                    event(f"poll:{kind}")
                    if transient and kind == "learner" and call.polls == 2:
                        return ConnectionError("transient")
                    # Keep one job alive after shutdown is observed to test the latch.
                    if remove_stop and kind == "actor" and call.polls < 4:
                        return TimeoutError()
                    return OUTCOMES[kind](call)
                return outcome

            def spawn(kind):
                def attempt(_number):
                    event(f"spawn:{kind}")
                    return RuntimeError("failed to submit learner") if (phase, kind) == ("backoff", "learner") else None
                return attempt

            def sleep(state):
                event("backoff" if state.ticks[-1] == 60 else "sleep")
                if remove_stop and requested:
                    stop.unlink(missing_ok=True)
                return False

            def mirrored(kind, copy):
                def mirror(name):
                    event(f"mirror:{kind}")
                    return copy(name)
                return mirror

            event("startup")
            # Gen 5 from big4 with two actors and an arena every five
            # generations: the first learner's checkpoint makes one due.
            state = scripted_driver(
                root, argv=["big4-abc.pt", "5", "2", "1", "1", "1", "4e-4", "20", "1000", "64", "5", "5"],
                script={kind: poll(kind) for kind in ROLES}, spawn_errors={kind: spawn(kind) for kind in ROLES},
                stop_when=sleep, env={"C4_MIRROR": "1" if mirror else "0"}, max_ticks=30,
                overrides=dict(published_history=lambda: [f"big{n}-abc.pt" for n in range(5)],
                               mirror_model=mirrored("model", lambda name: root / name),
                               fetch_shard=mirrored("shard", lambda name: (root / name, 1000))))
            self.assertIsNone(state.error)
            self.assertEqual(state.journal["calls"], [], "a drained driver leaves an empty journal")
        self.assertTrue(requested, "scenario never requested shutdown")
        self.assertCountEqual(state.collected, [call.object_id for calls in state.calls.values() for call in calls],
                              "in-flight work was not fully collected")
        self.assertEqual(state.cancelled, [])
        self.assertTrue(state.logs[-1].startswith("loop end:"))
        if state.calls["learner"]:
            self.assertIn("next gen 6", state.logs[-1])
            self.assertIn("model big5-ok.pt", state.logs[-1])
        return state.spawns, state.logs

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
