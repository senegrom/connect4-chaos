#!/usr/bin/env python3
"""Run any synchronous Playwright suite with failure evidence captured before teardown.

Usage: python scripts/browser_evidence.py scripts/ui-browser-regressions.py --browser chromium
The CLI supervises a separate suite process. A failure starts a hard capture
and teardown deadline outside Playwright, so a frozen renderer cannot hold CI
open. Partial artifacts and the original failure survive a forced shutdown.
Passing scenarios discard their traces when their contexts close.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import math
import os
import signal
import subprocess
import tempfile
import time
import re
import runpy
import sys
import traceback
from unittest.mock import patch
from uuid import uuid4

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "browser-results"


class FailureEvidence:
    def __init__(self, output=DEFAULT_OUTPUT, label="browser", failure_state=None):
        self.output = Path(output)
        self.failure_state = Path(failure_state) if failure_state is not None else None
        self.label = re.sub(r"[^a-zA-Z0-9_-]", "-", label)[:100] or "browser"
        self.contexts = {}
        self.capture_errors = []
        self.directory = None

    def observe(self, context):
        if context in self.contexts:
            return context
        log = deque(maxlen=200)
        self.contexts[context] = log
        context.on("close", lambda *_: self.contexts.pop(context, None))
        context.on("console", lambda message: log.append(f"console {message.type}: {message.text}"))
        context.on("weberror", lambda event: log.append(f"page error: {event.error}"))
        try:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)
        except Exception as error:
            self.capture_errors.append(f"trace start: {error}")
        return context

    def capture(self, error):
        """Best effort per artifact; never replace the assertion being diagnosed."""
        if self.directory is not None:
            return
        try:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            self.directory = self.output / f"{self.label}-{stamp}-{uuid4().hex[:8]}"
            original = "".join(traceback.format_exception(error))
            # Notify the supervisor BEFORE any browser or artifact operation.
            # The control file is in its own local temp directory, independent
            # of output permissions; replace avoids partially read JSON.
            if self.failure_state is not None:
                state = {"started": time.monotonic(), "exit_code": failure_exit_code(error),
                         "traceback": original, "directory": str(self.directory)}
                staging = self.failure_state.with_suffix(".tmp")
                staging.write_text(json.dumps(state), encoding="utf-8")
                staging.replace(self.failure_state)
            print(original, file=sys.stderr, end="", flush=True)
            self.directory.mkdir(parents=True)
            self._write("failure.txt", original)
            for index, (context, log) in enumerate(list(self.contexts.items())):
                prefix = f"context-{index}"
                self._write(f"{prefix}-console.txt", "\n".join(log))
                for number, page in enumerate(context.pages):
                    name = f"{prefix}-page-{number}"
                    self._attempt(f"{name} screenshot", lambda: page.screenshot(
                        path=str(self.directory / f"{name}.png"), full_page=True, timeout=5000))
                    self._attempt(f"{name} HTML", lambda: self._write(f"{name}.html", page.content()))
                    self._attempt(f"{name} state", lambda: self._write(f"{name}.json", json.dumps({
                        "url": page.url,
                        "state": page.evaluate("""() => ({
                          title: document.title,
                          activeElement: document.activeElement?.id,
                          text: Object.fromEntries(['statusText', 'opponentHint', 'activeRulesSummary', 'aiErrorText']
                            .map(id => [id, document.getElementById(id)?.textContent ?? null])),
                          controls: [...document.querySelectorAll('input, select, button')].map(el => ({
                            id: el.id, value: el.value, disabled: el.disabled, checked: el.checked
                          }))
                        })"""),
                    }, indent=2)))
                self._attempt(f"{prefix} trace", lambda: context.tracing.stop(
                    path=str(self.directory / f"{prefix}-trace.zip")))
            if self.capture_errors:
                self._write("capture-errors.txt", "\n".join(self.capture_errors))
            print(f"Browser failure evidence: {self.directory}", file=sys.stderr)
        except Exception as capture_error:
            print(f"Could not finish browser evidence: {capture_error}", file=sys.stderr)

    def _write(self, name, content):
        (self.directory / name).write_text(content, encoding="utf-8")

    def _attempt(self, label, action):
        try:
            action()
        except Exception as error:
            self.capture_errors.append(f"{label}: {error}")


@contextmanager
def recorded_playwright(evidence, factory=None):
    from playwright import sync_api
    factory = factory or sync_api.sync_playwright
    original_context = sync_api.Browser.new_context
    original_page = sync_api.Browser.new_page

    def new_context(browser, *args, **kwargs):
        return evidence.observe(original_context(browser, *args, **kwargs))

    def new_page(browser, *args, **kwargs):
        page = original_page(browser, *args, **kwargs)
        evidence.observe(page.context)
        return page

    # Catch inside the driver's context manager: after it exits screenshots
    # and traces are no longer available. Restore public API hooks on exit.
    with factory() as playwright:
        with patch.object(sync_api.Browser, "new_context", new_context), \
                patch.object(sync_api.Browser, "new_page", new_page):
            try:
                yield playwright
            except BaseException as error:
                if not isinstance(error, SystemExit) or error.code not in (None, 0):
                    evidence.capture(error)
                raise


def run_suite(script, args, output=DEFAULT_OUTPUT, failure_state=None):
    """In-process worker/test helper; use supervise_suite or the CLI for deadlines."""
    from playwright import sync_api
    script = Path(script).resolve()
    evidence = FailureEvidence(output, script.stem, failure_state)
    original = sync_api.sync_playwright
    # Suites keep their ordinary imports and assertions. This shared runner
    # only adds evidence collection around their existing Playwright lifetime.
    with patch.object(sync_api, "sync_playwright", lambda: recorded_playwright(evidence, original)), \
            patch.object(sys, "argv", [str(script), *args]):
        try:
            runpy.run_path(str(script), run_name="__main__")
        except BaseException as error:
            if not isinstance(error, SystemExit) or error.code not in (None, 0):
                evidence.capture(error)  # also captures launch/import failures
            raise


def failure_exit_code(error):
    if isinstance(error, SystemExit):
        # Noninteger SystemExit values print an error and exit 1 in Python.
        return error.code if isinstance(error.code, int) and 0 < error.code < 256 else 1
    return 130 if isinstance(error, KeyboardInterrupt) else 1


def _kill_tree(process):
    """Kill only this suite and its descendants, including detached browsers."""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
    else:
        # Playwright can put browsers in separate process groups. Snapshot
        # descendants before killing their parent; killing only the suite's
        # group would miss those browsers once they are reparented.
        descendants = set()
        try:
            listing = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True,
                                     text=True, timeout=2, check=True).stdout
            pairs = [tuple(map(int, line.split())) for line in listing.splitlines()]
            parents = {process.pid}
            while parents:
                children = {pid for pid, ppid in pairs if ppid in parents} - descendants
                descendants.update(children)
                parents = children
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        for pid in descendants:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def supervise_suite(script, args, output=DEFAULT_OUTPUT, *, capture_timeout=15.0, suite_timeout=600.0):
    """Return the suite's exit code; browser RPCs never run in this supervisor."""
    for name, value in (("capture timeout", capture_timeout), ("suite timeout", suite_timeout)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    with tempfile.TemporaryDirectory(prefix="connect4-browser-watchdog-") as temporary:
        state_file = Path(temporary) / "failure.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-state", str(state_file),
                   "--output", str(Path(output).resolve()), str(Path(script).resolve()), *args]
        options = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        process = subprocess.Popen(command, **options)
        started = time.monotonic()
        failure = None
        try:
            while process.poll() is None:
                if failure is None and state_file.exists():
                    failure = json.loads(state_file.read_text(encoding="utf-8"))
                now = time.monotonic()
                capture_expired = failure is not None and now - failure["started"] >= capture_timeout
                suite_expired = now - started >= suite_timeout
                if capture_expired or suite_expired:
                    reason = (f"Browser evidence capture/teardown exceeded {capture_timeout:g}s"
                              if capture_expired else f"Browser suite exceeded {suite_timeout:g}s")
                    print(reason + "; terminating the suite process tree.", file=sys.stderr, flush=True)
                    _kill_tree(process)
                    # Preserve diagnostics even when the child blocked before
                    # its output directory could be created.
                    directory = Path(failure["directory"]) if failure else Path(output) / "suite-timeout"
                    try:
                        directory.mkdir(parents=True, exist_ok=True)
                        (directory / "capture-timeout.txt").write_text(reason + "\n", encoding="utf-8")
                        if failure:
                            (directory / "failure.txt").write_text(failure["traceback"], encoding="utf-8")
                    except OSError as error:
                        print(f"Could not save timeout evidence: {error}", file=sys.stderr)
                    if failure:
                        print(failure["traceback"], file=sys.stderr, end="", flush=True)
                    return failure["exit_code"] if failure else 124
                time.sleep(0.05)
            return process.returncode if process.returncode >= 0 else 128 - process.returncode
        finally:
            if process.poll() is None:
                _kill_tree(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--capture-timeout", type=float, default=15.0)
    parser.add_argument("--suite-timeout", type=float, default=600.0)
    parser.add_argument("--worker-state", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("script", type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.worker_state is not None:
        run_suite(args.script, args.args, args.output, args.worker_state)
    else:
        def terminate(signum, _frame):
            # Run supervise_suite's finally block on CI cancellation too.
            raise SystemExit(128 + signum)
        previous = signal.signal(signal.SIGTERM, terminate)
        try:
            raise SystemExit(supervise_suite(args.script, args.args, args.output,
                                            capture_timeout=args.capture_timeout, suite_timeout=args.suite_timeout))
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
