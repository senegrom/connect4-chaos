#!/usr/bin/env python3
"""Exercise the shared evidence runner with real, deliberately failing browsers."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from browser_evidence import FailureEvidence, recorded_playwright, run_suite

BROWSER = "chromium"
EXECUTABLE = None
RUNNER = Path(__file__).with_name("browser_evidence.py")


def launch(pw):
    options = {"headless": True}
    if EXECUTABLE:
        options["executable_path"] = EXECUTABLE
    return getattr(pw, BROWSER).launch(**options)


class EvidenceTests(unittest.TestCase):
    def test_failure_captures_before_driver_teardown(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = FailureEvidence(directory, "injected failure")
            original = AssertionError("original browser assertion")
            with self.assertRaises(AssertionError) as caught:
                with recorded_playwright(evidence) as pw:
                    browser = launch(pw)
                    old = browser.new_context()
                    old.close()
                    self.assertEqual(len(evidence.contexts), 0)
                    for number in range(2):
                        context = browser.new_context()
                        page = context.new_page()
                        page.set_content(f'<title>Evidence {number}</title><p id="opponentHint">Loading</p>')
                        page.evaluate("console.error('diagnostic message')")
                    raise original
            self.assertIs(caught.exception, original)
            files = list(evidence.directory.iterdir())
            self.assertEqual(len(list(evidence.directory.glob("*.png"))), 2)
            self.assertEqual(len(list(evidence.directory.glob("*.html"))), 2)
            self.assertEqual(len(list(evidence.directory.glob("*.json"))), 2)
            for trace in evidence.directory.glob("*-trace.zip"):
                with zipfile.ZipFile(trace) as archive:
                    self.assertTrue(any(name.endswith(".trace") for name in archive.namelist()))
            self.assertEqual(len(list(evidence.directory.glob("*-trace.zip"))), 2)
            self.assertIn("original browser assertion", (evidence.directory / "failure.txt").read_text())
            self.assertTrue(any("diagnostic message" in path.read_text() for path in files if path.name.endswith("console.txt")))

    def test_success_discards_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = FailureEvidence(directory)
            with recorded_playwright(evidence) as pw:
                browser = launch(pw)
                page = browser.new_page()
                page.set_content("<p>successful scenario</p>")
                page.close()
                browser.close()
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(evidence.contexts, {})

    def test_broken_screenshot_does_not_replace_assertion_or_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = FailureEvidence(directory)
            original = AssertionError("keep this assertion")
            with self.assertRaises(AssertionError) as caught:
                with recorded_playwright(evidence) as pw:
                    browser = launch(pw)
                    page = browser.new_context().new_page()
                    page.set_content("<p>still capture this HTML</p>")
                    def broken(**_):
                        raise RuntimeError("screenshot unavailable")
                    page.screenshot = broken
                    raise original
            self.assertIs(caught.exception, original)
            self.assertTrue(list(evidence.directory.glob("*.html")))
            self.assertTrue(list(evidence.directory.glob("*-trace.zip")))
            self.assertIn("screenshot unavailable", (evidence.directory / "capture-errors.txt").read_text())

    def test_cli_preserves_failure_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "failing-suite.py"
            fixture.write_text(f'''from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
    browser = pw.{BROWSER}.launch(headless=True, executable_path={EXECUTABLE!r})
    page = browser.new_context().new_page()
    page.set_content("<p>CLI evidence</p>")
    assert False, "CLI assertion retained"
''')
            result = subprocess.run([sys.executable, str(RUNNER), "--output", str(root / "evidence"), str(fixture)],
                                    capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("AssertionError: CLI assertion retained", result.stderr)
            self.assertTrue(list((root / "evidence").glob("**/*.png")))
            self.assertTrue(list((root / "evidence").glob("**/*-trace.zip")))

    def test_failure_before_browser_still_has_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "startup.py"
            fixture.write_text('raise RuntimeError("startup failed")\n')
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                run_suite(fixture, [], Path(directory) / "evidence")
            files = list((Path(directory) / "evidence").glob("**/failure.txt"))
            self.assertEqual(len(files), 1)
            self.assertIn("startup failed", files[0].read_text())

    def test_successful_system_exit_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "success.py"
            fixture.write_text('raise SystemExit(0)\n')
            output = Path(directory) / "evidence"
            with self.assertRaises(SystemExit) as caught:
                run_suite(fixture, [], output)
            self.assertEqual(caught.exception.code, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", choices=("chromium", "webkit"), default="chromium")
    parser.add_argument("--executable")
    args, rest = parser.parse_known_args()
    BROWSER, EXECUTABLE = args.browser, args.executable
    unittest.main(argv=[sys.argv[0], *rest])
