# WebKit CI compatibility pin

## Why the browser jobs use separate requirements

The Playwright 1.64.0 update (#133) brings WebKit 27.2. That browser reports a
`style-src-attr` CSP violation for an ordinary `<select>` with no page-supplied
inline style. It emits both a console error and a `securitypolicyviolation`
event; these cannot reliably be distinguished from genuine inline-style
violations. Upstream report: https://github.com/microsoft/playwright/issues/43497.

Do not whitelist the message, allow inline styles, bypass CSP or set
`C4_BROWSER_ERRORS=report` in CI to make the suite pass.

The Chromium job continues to use `scripts/browser-requirements.txt` (1.64.0).
The WebKit job temporarily uses `scripts/webkit-requirements.txt` (1.63.0 and
its matching WebKit 26.6). Each job installs the driver and browser together
in its own runner; no unsupported cross-version executable override is used.
All existing browser suites and the Pages publication gate remain required.
This restores coverage on the last working WebKit; it is not a fix to WebKit
27.2 itself and does not claim coverage of that broken version.

## Regression guard

Both jobs first run `scripts/test-browser-csp.py`. It checks that:

- A clean dropdown works under strict CSP without a violation.
- A real inline style element, inline style attribute and inline script are
  blocked and each makes the actual `browser_evidence.py` CLI fail.

The negative controls check both browser enforcement and the runner's exit
status/error text. They cannot pass just because the fixture crashes before
its assertions. The existing page-error, console-error and failure-evidence
regressions also still run without modification.

To reproduce the guard locally, use a separate virtual environment for each
requirements file (do not install the two incompatible pins together):

```sh
python3 -m venv .venv-webkit
.venv-webkit/bin/python -m pip install -r scripts/webkit-requirements.txt
.venv-webkit/bin/python -m playwright install --with-deps webkit
.venv-webkit/bin/python scripts/test-browser-csp.py --browser webkit
```

## Removing the workaround

Keep the WebKit requirements file visible to the existing `/scripts`
Dependabot configuration; no dependency is ignored. A proposed upgrade must
pass the clean-dropdown guard, all three negative controls and every existing
WebKit suite. Then return WebKit to the common requirements file and remove
its temporary requirements file/matrix override. Keep the CSP guard. Never
relax the assertions just to accept the affected WebKit release.
