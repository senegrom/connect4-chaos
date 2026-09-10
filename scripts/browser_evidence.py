#!/usr/bin/env python3
"""Run any synchronous Playwright suite with failure evidence captured before teardown.

Usage: python scripts/browser_evidence.py scripts/ui-browser-regressions.py --browser chromium
Only test-side context creation is instrumented; page code and network responses
are unchanged. Passing scenarios discard their traces when their contexts close.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import re
import runpy
import sys
import traceback
from unittest.mock import patch
from uuid import uuid4

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "browser-results"


class FailureEvidence:
    def __init__(self, output=DEFAULT_OUTPUT, label="browser"):
        self.output = Path(output)
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
            self.directory.mkdir(parents=True)
            self._write("failure.txt", "".join(traceback.format_exception(error)))
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


def run_suite(script, args, output=DEFAULT_OUTPUT):
    from playwright import sync_api
    script = Path(script).resolve()
    evidence = FailureEvidence(output, script.stem)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("script", type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    run_suite(args.script, args.args, args.output)


if __name__ == "__main__":
    main()
