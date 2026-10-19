#!/usr/bin/env python3
"""Check browser CSP enforcement and the evidence runner's strict error gate.

The clean select fixture catches microsoft/playwright#43497. Negative
controls must fail through browser_evidence.py, not merely emit an event.
No external server, production policy changes or console allowances are used.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

BROWSER = "chromium"
RUNNER = Path(__file__).with_name("browser_evidence.py")
CSP = "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'none'"


def fixture_source(browser: str, body: str, check: str, *, violation: bool) -> str:
    html = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>CSP regression control</title></head><body>" + body + "</body></html>")
    return textwrap.dedent(f"""\
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.{browser}.launch(headless=True)
            context = browser.new_context()
            context.add_init_script(""" + repr("""
                window.__cspViolations = [];
                document.addEventListener('securitypolicyviolation', event => {
                    window.__cspViolations.push(event.effectiveDirective);
                });
            """) + f""")
            context.route('http://csp-regression.test/', lambda route: route.fulfill(
                status=200, content_type='text/html',
                headers={{'Content-Security-Policy': {CSP!r}}}, body={html!r}))
            page = context.new_page()
            page.goto('http://csp-regression.test/')
            if {violation!r}:
                page.wait_for_function('window.__cspViolations.length > 0', timeout=5000)
            # Let rendering and asynchronous console/CSP notifications settle.
            page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
            page.wait_for_timeout(300)
            {check}
            print('CSP control assertions passed', flush=True)
            browser.close()
        """)


class CspTests(unittest.TestCase):
    def run_fixture(self, body: str, check: str, *, violation: bool = False) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory(prefix="connect4-csp-") as directory:
            root = Path(directory)
            fixture = root / "csp-control.py"
            fixture.write_text(fixture_source(BROWSER, body, check, violation=violation), encoding="utf-8")
            environment = os.environ.copy()
            # A caller's calibration setting must not turn negative controls green.
            environment.pop("C4_BROWSER_ERRORS", None)
            result = subprocess.run(
                [sys.executable, str(RUNNER), "--output", str(root / "evidence"),
                 "--suite-timeout", "20", "--capture-timeout", "5", str(fixture)],
                capture_output=True, text=True, timeout=30, env=environment,
            )
        self.assertIn("CSP control assertions passed", result.stdout, result.stdout + result.stderr)
        return result

    def test_select_does_not_violate_csp(self):
        result = self.run_fixture(
            '<select id="control"><option value="a">A</option><option value="b">B</option></select>',
            "page.select_option('#control', 'b'); "
            "assert page.input_value('#control') == 'b'; "
            "assert page.evaluate('window.__cspViolations') == []",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assert_rejected(self, body: str, check: str):
        result = self.run_fixture(body, check, violation=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        # A launch, syntax or fixture assertion failure is NOT proof of the gate.
        self.assertIn("unexpected page or console error", result.stderr)
        self.assertIn("console error:", result.stderr)

    def test_inline_style_element_is_blocked_and_fails_the_runner(self):
        self.assert_rejected(
            '<style>#target { font-style: italic; }</style><p id="target">Text</p>',
            "assert page.evaluate(\"getComputedStyle(document.getElementById('target')).fontStyle\") == 'normal'; "
            "assert any(item.startswith('style-src') for item in page.evaluate('window.__cspViolations'))",
        )

    def test_inline_style_attribute_is_blocked_and_fails_the_runner(self):
        self.assert_rejected(
            '<p id="target" style="font-style: italic">Text</p>',
            "assert page.evaluate(\"getComputedStyle(document.getElementById('target')).fontStyle\") == 'normal'; "
            "assert any(item.startswith('style-src') for item in page.evaluate('window.__cspViolations'))",
        )

    def test_inline_script_is_blocked_and_fails_the_runner(self):
        self.assert_rejected(
            '<script>window.__blockedScriptRan = true;</script><p>Text</p>',
            "assert page.evaluate('window.__blockedScriptRan === undefined'); "
            "assert any(item.startswith('script-src') for item in page.evaluate('window.__cspViolations'))",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", choices=("chromium", "webkit"), default="chromium")
    args, remaining = parser.parse_known_args()
    BROWSER = args.browser
    unittest.main(argv=[sys.argv[0], *remaining])
