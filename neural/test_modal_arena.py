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
        played = Mock(return_value=({}, 0, 0.0, {}))
        load = Mock(side_effect=lambda path, device: path)
        parse = Mock(side_effect=lambda spec: spec)
        cli = function(ROOT / "neural/arena.py", "main", dict(
            sys=sys, DEFAULT_SHAPES="all", parse_shapes=parse, load=load, play=played,
            torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            report=lambda *_: (0.5, "arena regression report")))
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
        result = wrapper(" a.pt, b.pt ", "c.pt", games=2, sims=32, **options)
        volume.reload.assert_called_once_with()
        played.assert_called_once()
        self.assertEqual(result["exit"], 0)
        self.assertIn("arena regression report", result["out"])
        self.assertEqual(played.call_args.args[:2], ("/tables/models/a.pt,/tables/models/b.pt", "/tables/models/c.pt"))
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
