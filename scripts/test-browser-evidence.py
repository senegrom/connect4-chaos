#!/usr/bin/env python3
"""Exercise the shared evidence runner with real, deliberately failing browsers."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from browser_evidence import FailureEvidence, recorded_playwright, run_suite, supervise_suite

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


    def run_bounded_cli(self, root, source, *options):
        fixture = root / "supervised-suite.py"
        fixture.write_text(source, encoding="utf-8")
        started = time.monotonic()
        result = subprocess.run([sys.executable, str(RUNNER), "--output", str(root / "evidence"),
                                 *options, str(fixture)], capture_output=True, text=True, timeout=20)
        return result, time.monotonic() - started

    def test_frozen_renderer_cannot_hold_capture_open(self):
        # This is a real blocked renderer/driver operation, not a mocked
        # timeout exception. subprocess.run supplies an independent watchdog.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, elapsed = self.run_bounded_cli(root, f'''import time
from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
    browser = pw.{BROWSER}.launch(headless=True, executable_path={EXECUTABLE!r})
    page = browser.new_context().new_page()
    page.set_content("<p>Renderer freeze regression</p>")
    page.evaluate("() => {{ setTimeout(() => {{ while (true) {{}} }}, 1000); }}")
    time.sleep(1.3)
    raise AssertionError("original frozen renderer failure")
''', "--capture-timeout", "1", "--suite-timeout", "15")
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertLess(elapsed, 12, result.stderr)
            self.assertIn("original frozen renderer failure", result.stderr)
            files = list((root / "evidence").glob("**/failure.txt"))
            self.assertEqual(len(files), 1)
            self.assertIn("original frozen renderer failure", files[0].read_text())
            # A browser may fail its RPC promptly, or require a hard stop.
            # Either result must retain the original failure and exit promptly.
            if "terminating the suite process tree" in result.stderr:
                self.assertTrue(list((root / "evidence").glob("**/capture-timeout.txt")))

    def test_hung_trace_preserves_original_exit_code_and_partial_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, elapsed = self.run_bounded_cli(root, f'''import time
from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
    browser = pw.{BROWSER}.launch(headless=True, executable_path={EXECUTABLE!r})
    context = browser.new_context()
    page = context.new_page()
    page.set_content("<p>Preserve this partial evidence</p>")
    def blocked_stop(**kwargs):
        time.sleep(60)
    context.tracing.stop = blocked_stop
    raise SystemExit(7)
''', "--capture-timeout", "2")
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertLess(elapsed, 12, result.stderr)
            self.assertIn("capture/teardown exceeded", result.stderr)
            self.assertTrue(list((root / "evidence").glob("**/failure.txt")))
            self.assertTrue(list((root / "evidence").glob("**/*.html")))
            self.assertTrue(list((root / "evidence").glob("**/*.json")))

    def test_suite_deadline_also_covers_a_hang_before_an_assertion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, elapsed = self.run_bounded_cli(root, "import time; time.sleep(60)\n", "--suite-timeout", "1")
            self.assertEqual(result.returncode, 124, result.stderr)
            self.assertLess(elapsed, 10)
            self.assertIn("Browser suite exceeded", result.stderr)
            self.assertTrue(list((root / "evidence").glob("**/capture-timeout.txt")))

    def test_capture_budget_does_not_limit_a_passing_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, _ = self.run_bounded_cli(root, "import time; time.sleep(0.3); raise SystemExit(0)\n",
                                             "--capture-timeout", "0.1", "--suite-timeout", "10")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "evidence").exists())

    def test_invalid_deadlines_fail_before_launch(self):
        for value in (0, -1, float("inf"), float("nan")):
            for argument in ("capture_timeout", "suite_timeout"):
                with self.subTest(argument=argument, value=value), self.assertRaises(ValueError):
                    supervise_suite("never-started.py", [], **{argument: value})

    @unittest.skipUnless(os.name == "posix", "POSIX detached-process cleanup")
    def test_capture_deadline_kills_detached_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_file = root / "child.pid"
            result, _ = self.run_bounded_cli(root, f'''import subprocess, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
    browser = pw.{BROWSER}.launch(headless=True, executable_path={EXECUTABLE!r})
    context = browser.new_context()
    page = context.new_page()
    page.set_content("<p>owned detached process</p>")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    Path({str(pid_file)!r}).write_text(str(child.pid))
    def blocked_screenshot(**kwargs):
        time.sleep(60)
    page.screenshot = blocked_screenshot
    raise AssertionError("detached child regression")
''', "--capture-timeout", "1")
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("capture/teardown exceeded", result.stderr)
            pid = int(pid_file.read_text())
            # Containers may retain a killed child briefly as a zombie, which
            # holds no browser resources and cannot keep output pipes open.
            status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
            self.assertTrue(not status or status.startswith("Z"), f"descendant {pid} still running: {status}")


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux subreaper ownership")
class ProcessOwnershipTests(unittest.TestCase):
    def run_fixture(self, source, *, cancel=False):
        import signal
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_file = root / "children"
            suite = root / "suite.py"
            suite.write_text("from pathlib import Path\nPID_FILE = Path(" + repr(str(pid_file)) + ")\n" + source)
            runner = subprocess.Popen([sys.executable, str(RUNNER), "--output", str(root / "evidence"),
                                       "--suite-timeout", "12", str(suite)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                if cancel:
                    until = time.monotonic() + 8
                    while not pid_file.exists() and time.monotonic() < until and runner.poll() is None:
                        time.sleep(0.01)
                    self.assertTrue(pid_file.exists(), "suite never started")
                    runner.terminate()
                # Inherited pipe EOF is part of the assertion, not merely wait().
                out, err = runner.communicate(timeout=15)
                self.assertTrue(pid_file.exists(), err)
                children = [int(pid) for pid in pid_file.read_text().split()]
                self.assertTrue(children, "fixture created no children")
                for pid in children:
                    self.assertFalse(Path(f"/proc/{pid}").exists(), f"owned child {pid} survived or became a zombie")
                evidence = list((root / "evidence").glob("**/failure.txt"))
                return runner.returncode, out + err, [path.read_text() for path in evidence]
            finally:
                if runner.poll() is None:
                    runner.terminate()
                    try:
                        runner.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        runner.kill()
                        runner.wait(timeout=3)
                # A failing regression must not leave its own test children alive.
                if pid_file.exists():
                    for pid in pid_file.read_text().split():
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                runner.stdout.close()
                runner.stderr.close()

    def test_prompt_exits_clean_up_attached_and_detached_children(self):
        for detach in (False, True):
            for ending, expected in (("raise SystemExit(0)", 0), ("raise SystemExit(7)", 7),
                                     ("os._exit(9)", 9), ("os.kill(os.getpid(), signal.SIGKILL)", 137)):
                with self.subTest(detach=detach, ending=ending):
                    code, output, evidence = self.run_fixture(f'''import os, signal, subprocess, sys
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session={detach!r})
PID_FILE.write_text(str(child.pid))
{ending}
''')
                    self.assertEqual(code, expected, output)
                    self.assertNotIn("suite exceeded", output)
                    if expected == 7:
                        self.assertTrue(any("SystemExit: 7" in text for text in evidence))
                    elif expected == 0:
                        self.assertFalse(evidence)

    def test_double_forked_grandchild_is_still_owned(self):
        code, output, _ = self.run_fixture('''import subprocess, sys
middle = "import os,subprocess,sys; from pathlib import Path; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); Path(sys.argv[1]).write_text(str(child.pid)); os._exit(0)"
subprocess.run([sys.executable, "-c", middle, str(PID_FILE)], check=True, start_new_session=True)
raise SystemExit(0)
''')
        self.assertEqual(code, 0, output)

    def test_cancellation_reaps_detached_children(self):
        code, output, _ = self.run_fixture('''import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
PID_FILE.write_text(str(child.pid))
time.sleep(60)
''', cancel=True)
        self.assertEqual(code, 143, output)

    def test_imported_supervision_does_not_touch_unrelated_children(self):
        import ctypes
        libc = ctypes.CDLL(None)
        before, after = ctypes.c_int(), ctypes.c_int()
        self.assertEqual(libc.prctl(37, ctypes.byref(before), 0, 0, 0), 0)  # PR_GET_CHILD_SUBREAPER
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        try:
            with tempfile.TemporaryDirectory() as directory:
                suite = Path(directory) / "pass.py"
                suite.write_text("raise SystemExit(0)\n")
                self.assertEqual(supervise_suite(suite, [], Path(directory) / "evidence"), 0)
            self.assertIsNone(unrelated.poll())
            self.assertEqual(libc.prctl(37, ctypes.byref(after), 0, 0, 0), 0)
            self.assertEqual(before.value, after.value)
        finally:
            unrelated.kill()
            unrelated.wait(timeout=3)

    def test_abrupt_browser_exit_reaps_driver_and_browser(self):
        code, output, _ = self.run_fixture(f'''import os, subprocess
from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
    browser = pw.{BROWSER}.launch(headless=True, executable_path={EXECUTABLE!r})
    browser.new_context().new_page().set_content("<p>abrupt browser exit</p>")
    listing = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True, check=True).stdout
    pairs = [tuple(map(int, line.split())) for line in listing.splitlines()]
    owned, parents = set(), {{os.getpid()}}
    while parents:
        children = {{pid for pid, ppid in pairs if ppid in parents}} - owned
        owned.update(children)
        parents = children
    PID_FILE.write_text(" ".join(map(str, owned)))
    os._exit(7)
''')
        self.assertEqual(code, 7, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", choices=("chromium", "webkit"), default="chromium")
    parser.add_argument("--executable")
    args, rest = parser.parse_known_args()
    BROWSER, EXECUTABLE = args.browser, args.executable
    unittest.main(argv=[sys.argv[0], *rest])
