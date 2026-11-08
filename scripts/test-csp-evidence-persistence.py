#!/usr/bin/env python3
"""Unit tests for CSP artifact retention; no browser or network required."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "browser_csp", Path(__file__).with_name("test-browser-csp.py"))
assert SPEC is not None and SPEC.loader is not None
CSP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CSP)
PASS = "CSP control assertions passed\n"


class EvidencePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="csp-artifact-unit-")
        self.addCleanup(self.temporary.cleanup)
        self.previous = Path.cwd()
        os.chdir(self.temporary.name)
        self.addCleanup(os.chdir, self.previous)
        self.case = CSP.CspTests("test_select_does_not_violate_csp")
        self.calls = []

    def runner(self, returncode=0, stdout=PASS, stderr="", timeout=False):
        def execute(command, **kwargs):
            evidence = Path(command[command.index("--output") + 1])
            fixture = Path(command[-1])
            self.calls.append((evidence, fixture, command, kwargs))
            self.assertTrue(fixture.is_file())
            (evidence / "trace.zip").write_bytes(b"unit-test diagnostic placeholder")
            if timeout:
                raise subprocess.TimeoutExpired(command, 30, output=stdout, stderr=stderr)
            return subprocess.CompletedProcess(command, returncode, stdout, stderr)
        return execute

    def invoke(self, **kwargs):
        with patch.object(CSP.subprocess, "run", side_effect=self.runner(**kwargs)):
            return self.case.run_fixture("<p>Test</p>", "assert True")

    def test_evidence_survives_temporary_fixture_cleanup(self):
        self.invoke()
        evidence, fixture, _, _ = self.calls[0]
        self.assertFalse(fixture.exists())
        self.assertTrue(evidence.is_relative_to(Path.cwd() / "browser-results" / "csp" / "chromium"))
        self.assertTrue((evidence / "trace.zip").is_file())
        self.assertTrue((evidence / "fixture.py").is_file())
        self.assertEqual((evidence / "stdout.txt").read_text(), PASS)
        self.assertEqual((evidence / "exit-code.txt").read_text(), "0\n")

    def test_fixture_assertion_failure_keeps_diagnostics_and_fails(self):
        with self.assertRaises(AssertionError):
            self.invoke(returncode=1, stdout="", stderr="fixture assertion failed")
        evidence, fixture, _, _ = self.calls[0]
        self.assertFalse(fixture.exists())
        self.assertEqual((evidence / "stderr.txt").read_text(), "fixture assertion failed")
        self.assertEqual((evidence / "exit-code.txt").read_text(), "1\n")

    def test_negative_control_return_code_is_not_rewritten(self):
        result = self.invoke(returncode=1, stderr="unexpected page or console error: console error:")
        self.assertEqual(result.returncode, 1)

    def test_clean_select_still_rejects_runner_failure(self):
        with patch.object(CSP.subprocess, "run", side_effect=self.runner(returncode=1)):
            with self.assertRaises(AssertionError):
                self.case.test_select_does_not_violate_csp()

    def test_negative_control_still_rejects_a_successful_runner(self):
        with patch.object(CSP.subprocess, "run", side_effect=self.runner(returncode=0)):
            with self.assertRaises(AssertionError):
                self.case.assert_rejected("<p>Test</p>", "assert True")

    def test_timeout_retains_bytes_output_and_still_raises(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.invoke(timeout=True, stdout=b"partial stdout\xff", stderr=b"partial stderr")
        evidence, fixture, _, _ = self.calls[0]
        self.assertFalse(fixture.exists())
        self.assertEqual((evidence / "stdout.txt").read_text(), "partial stdout\ufffd")
        self.assertEqual((evidence / "stderr.txt").read_text(), "partial stderr")
        self.assertTrue((evidence / "timeout.txt").is_file())

    def test_timeout_with_no_output_is_retained(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.invoke(timeout=True, stdout=None, stderr=None)
        evidence = self.calls[0][0]
        self.assertEqual((evidence / "stdout.txt").read_text(), "")
        self.assertEqual((evidence / "stderr.txt").read_text(), "")

    def test_strict_gate_and_deadlines_are_preserved(self):
        with patch.dict(os.environ, {"C4_BROWSER_ERRORS": "ignore"}):
            self.invoke()
        _, _, command, kwargs = self.calls[0]
        self.assertNotIn("C4_BROWSER_ERRORS", kwargs["env"])
        self.assertEqual(command[command.index("--suite-timeout") + 1], "20")
        self.assertEqual(command[command.index("--capture-timeout") + 1], "5")
        self.assertEqual(kwargs["timeout"], 30)

    def test_repeated_runs_do_not_overwrite_evidence(self):
        self.invoke(stdout=PASS + "first")
        self.invoke(stdout=PASS + "second")
        first, second = [item[0] for item in self.calls]
        self.assertNotEqual(first, second)
        self.assertTrue((first / "stdout.txt").read_text().endswith("first"))
        self.assertTrue((second / "stdout.txt").read_text().endswith("second"))

    def test_generated_fixtures_remain_valid_python(self):
        for browser in ("chromium", "webkit"):
            for violation in (True, False):
                source = CSP.fixture_source(browser, "<p>Test</p>", "assert True", violation=violation)
                compile(source, "fixture.py", "exec")
                self.assertIn(CSP.CSP, source)


if __name__ == "__main__":
    unittest.main()
