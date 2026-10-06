"""Exercise Modal's arena wrapper through the real CLI argument parser, without GPUs.

Only deployment, subprocess and match execution are replaced. The functions
under test are read from the production modules, not copied into the fixture.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .test_support import function
from .training_config import ARENA_GAMES, ARENA_SEED, ARENA_SHAPES, ARENA_SIMS

ROOT = Path(__file__).resolve().parents[1]
# The loop's arena, which both the wrapper and the CLI take for what they are not given.
ARENA = dict(ARENA_GAMES=ARENA_GAMES, ARENA_SEED=ARENA_SEED, ARENA_SHAPES=ARENA_SHAPES, ARENA_SIMS=ARENA_SIMS)


class ModalArenaTests(unittest.TestCase):
    def cli(self):
        """arena.main with only its match and its checkpoints replaced:
        returns it and the play() and report() it calls."""
        played = Mock(return_value=({}, 0, 0.0, {}, {}))
        load = Mock(side_effect=lambda path, device: path)
        parse = Mock(side_effect=lambda spec: spec)
        report = Mock(return_value=(0.5, "arena regression report"))
        main = function(ROOT / "neural/arena.py", "main", dict(
            sys=sys, Path=Path, parse_shapes=parse, load=load, play=played,
            torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)), report=report, **ARENA))
        return main, played, report

    def invoke(self, **options):
        """The wrapper through the real CLI; games and sims are 2 and 32
        unless `options` says otherwise, and None leaves one out."""
        options = {key: value for key, value in {"games": 2, "sims": 32, **options}.items() if value is not None}
        cli, played, report = self.cli()
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            self.assertEqual(command[:3], ["python", "-m", "neural.arena"])
            self.assertEqual(kwargs["cwd"], "/repo")
            self.assertEqual(kwargs["env"]["PYTHONPATH"], "/repo")
            output = io.StringIO()
            with patch.object(sys, "argv", command[2:]), redirect_stdout(output):
                cli()
            return subprocess.CompletedProcess(command, 0, output.getvalue(), "")

        volume = SimpleNamespace(reload=Mock())
        wrapper = function(ROOT / "neural/modal_app.py", "arena", dict(
            os=os, time=time, TABLES="/tables", tables=volume,
            subprocess=SimpleNamespace(run=run), **ARENA))
        result = wrapper(" a.pt ", "c.pt", **options)
        volume.reload.assert_called_once_with()
        played.assert_called_once()
        # Labelled by checkpoint name: on Modal the paths are /tables/models/...
        self.assertEqual(report.call_args.args[3:5], ("a.pt", "c.pt"))
        self.assertEqual(result["exit"], 0)
        self.assertIn("arena regression report", result["out"])
        self.assertEqual(played.call_args.args[:2], ("/tables/models/a.pt", "/tables/models/c.pt"))
        self.assertEqual(played.call_args.args[3:5], (options.get("games", ARENA_GAMES),
                                                      options.get("sims", ARENA_SIMS)))
        self.assertEqual(played.call_args.args[6], "cpu")
        return played.call_args.args, commands[0]

    def test_omitted_values_are_the_loops_arena(self):
        # Both took 32 games a board where the loop plays 6: a direct
        # `modal run neural/modal_app.py::arena`, or a local match, was five
        # times the loop's arena, and on the same seed drew other openings.
        args, command = self.invoke(games=None, sims=None)
        self.assertEqual(command[5:], [str(ARENA_GAMES), str(ARENA_SIMS), ARENA_SHAPES, str(ARENA_SEED)])
        self.assertEqual(args[2:6], (ARENA_SHAPES, ARENA_GAMES, ARENA_SIMS, ARENA_SEED))
        self.assertEqual(args[7], ARENA_SIMS)
        cli, played, _report = self.cli()
        with patch.object(sys, "argv", ["arena.py", "a.pt", "b.pt"]), redirect_stdout(io.StringIO()):
            cli()
        self.assertEqual(played.call_args.args[2:], (ARENA_SHAPES, ARENA_GAMES, ARENA_SIMS, ARENA_SEED,
                                                     "cpu", ARENA_SIMS))

    def test_omitted_shapes_preserve_seed_and_b_budget(self):
        args, command = self.invoke(seed=42, sims_b=256)
        self.assertEqual(args[2], "all")
        self.assertEqual(args[5], 42)
        self.assertEqual(args[7], 256)
        self.assertEqual(command[-3:], ["all", "42", "256"])

    def test_empty_whitespace_and_explicit_shapes(self):
        for spec in ("", "  ", "all", "6x7c4chaos,8x8c5classic"):
            with self.subTest(shapes=spec):
                args, _ = self.invoke(shapes=spec, seed=19, sims_b=128)
                self.assertEqual(args[2], spec.strip() or "all")
                self.assertEqual(args[5], 19)
                self.assertEqual(args[7], 128)

    def test_zero_seed_and_policy_only_b_are_not_treated_as_omitted(self):
        args, _ = self.invoke(seed=0, sims_b=0)
        self.assertEqual(args[5], 0)
        self.assertEqual(args[7], 0)

    def test_checkpoint_lists_are_refused_before_any_volume_work(self):
        # Ensembles were removed: a comma list is not a file name either.
        volume = SimpleNamespace(reload=Mock())
        run = Mock()
        namespace = dict(os=os, time=time, TABLES="/tables", tables=volume,
                         subprocess=SimpleNamespace(run=run), **ARENA)
        arena = function(ROOT / "neural/modal_app.py", "arena", dict(namespace))
        measure = function(ROOT / "neural/modal_app.py", "measure", dict(namespace))
        for a, b in (("a.pt,b.pt", "c.pt"), ("a.pt", " b.pt, c.pt"), ("a.pt", "  ")):
            with self.subTest(a=a, b=b), self.assertRaisesRegex(ValueError, "one checkpoint"):
                arena(a, b)
        for name in ("a.pt,b.pt", " "):
            with self.subTest(model=name), self.assertRaisesRegex(ValueError, "one checkpoint"):
                measure(name)
        volume.reload.assert_not_called()
        run.assert_not_called()

    def test_default_b_budget_matches_a_without_discarding_seed(self):
        for spec in ("", "all", "6x7c4classic"):
            with self.subTest(shapes=spec):
                args, command = self.invoke(shapes=spec, seed=23)
                self.assertEqual(args[5], 23)
                self.assertEqual(args[7], 32)
                self.assertEqual(len(command), 9)  # B's default remains the CLI's default.
        args, _ = self.invoke()
        self.assertEqual(args[5], 7)
        self.assertEqual(args[7], 32)


if __name__ == "__main__":
    unittest.main()
