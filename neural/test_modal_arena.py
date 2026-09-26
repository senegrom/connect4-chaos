"""Exercise Modal's arena wrapper through the real CLI argument parser, without GPUs.

Only deployment, subprocess and match execution are replaced. The functions
under test are read from the production modules, not copied into the fixture.
"""
from __future__ import annotations

import ast
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

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class ModalArenaTests(unittest.TestCase):
    def invoke(self, **options):
        played = Mock(return_value=({}, 0, 0.0, {}, {}))
        load = Mock(side_effect=lambda path, device: path)
        parse = Mock(side_effect=lambda spec: spec)
        report = Mock(return_value=(0.5, "arena regression report"))
        cli = function(ROOT / "neural/arena.py", "main", dict(
            sys=sys, Path=Path, DEFAULT_SHAPES="all", parse_shapes=parse, load=load, play=played,
            torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)), report=report))
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
            subprocess=SimpleNamespace(run=run)))
        result = wrapper(" a.pt ", "c.pt", games=2, sims=32, **options)
        volume.reload.assert_called_once_with()
        played.assert_called_once()
        # Labelled by checkpoint name: on Modal the paths are /tables/models/...
        self.assertEqual(report.call_args.args[3:5], ("a.pt", "c.pt"))
        self.assertEqual(result["exit"], 0)
        self.assertIn("arena regression report", result["out"])
        self.assertEqual(played.call_args.args[:2], ("/tables/models/a.pt", "/tables/models/c.pt"))
        self.assertEqual(played.call_args.args[3:5], (2, 32))
        self.assertEqual(played.call_args.args[6], "cpu")
        return played.call_args.args, commands[0]

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
                         subprocess=SimpleNamespace(run=run))
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

    def test_measure_passes_its_holdouts_to_the_scorer(self):
        environments = []
        volume = SimpleNamespace(reload=Mock())

        def run(command, **kwargs):
            environments.append(kwargs["env"])
            return subprocess.CompletedProcess(command, 0, "", "")

        measure = function(ROOT / "neural/modal_app.py", "measure", dict(
            os=os, time=time, TABLES="/tables", tables=volume, subprocess=SimpleNamespace(run=run)))
        with patch.dict(os.environ, DISTILL_HOLDOUT_CONFIGS="6x6c4chaos"):   # the container's, not the caller's
            measure("a.pt", holdout_configs="4x4c3classic")
            measure("a.pt")
        self.assertEqual([env["DISTILL_HOLDOUT_CONFIGS"] for env in environments], ["4x4c3classic", ""])

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
