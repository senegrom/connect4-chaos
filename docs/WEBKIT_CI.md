# WebKit CSP regression guard

## History: the Playwright 1.64 WebKit pin

The Playwright 1.64.0 update (#133) brought WebKit 27.2. That build reported a
`style-src-attr` CSP violation for an ordinary `<select>` with no page-supplied
inline style. It emitted both a console error and a `securitypolicyviolation`
event, which cannot reliably be told apart from genuine inline-style
violations. It was a WebKit regression, present in Playwright 1.64.0 and
1.64.1: https://github.com/microsoft/playwright/issues/43497 (a duplicate of
https://github.com/microsoft/playwright/issues/43416).

From #135 the WebKit job therefore ran on a pin of its own,
`scripts/webkit-requirements.txt` (Playwright 1.63.0 with its WebKit 26.6),
while Chromium moved on. Playwright 1.65.0 ships WebKit 27.2 build v2388,
which passes the guard below, its negative controls and every WebKit suite.
Both jobs again install `scripts/browser-requirements.txt`, and the separate
pin is gone.

Do not whitelist the message, allow inline styles, bypass CSP or set
`C4_BROWSER_ERRORS=report` in CI to make a suite pass. If a browser release
regresses this way again, pin that browser alone in its own requirements file
and matrix entry, as #135 did, rather than relaxing the guard. Return it to
the common requirements file once an upgrade passes the guard and every
suite.

## Regression guard

Both jobs run `scripts/test-browser-csp.py` before any suite. It checks that:

- A clean dropdown works under strict CSP without a violation.
- A real inline style element, inline style attribute and inline script are
  blocked and each makes the actual `browser_evidence.py` CLI fail. So does
  an inline style that the page's own script sets after it has loaded.

The negative controls check both browser enforcement and the runner's exit
status/error text. They cannot pass just because the fixture crashes before
its assertions. The page-error, console-error and failure-evidence
regressions of `scripts/test-browser-evidence.py` also still run.

Just before the guard, `scripts/test-csp-evidence-persistence.py` checks,
without a browser, that it keeps the evidence of each control under
`browser-results/`, which CI uploads when a job fails.

To reproduce the guard locally:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/browser-requirements.txt
.venv/bin/python -m playwright install --with-deps webkit
.venv/bin/python scripts/test-browser-csp.py --browser webkit
```
