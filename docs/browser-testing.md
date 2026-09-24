# Browser regression testing

Two kinds of browser check run in CI.

**The browser smoke** is `scripts/browser-smoke.mjs`, run as `npm run
test:browser`. The CI test job runs it against the assembled Pages artifact
(`_site`), not the checkout, so a file the site needs but the artifact
omits fails there. It drives Chrome directly over the DevTools Protocol:
page load, the AI worker, the lazy strategy fetch, layout and animation
timing. `scripts/browser-smoke-retry.mjs` retries a browser that fails to
start.

**The Playwright suites** run in the Browser regressions workflow, on
Chromium and on WebKit, each with desktop and 390-pixel-wide touch-enabled
mobile contexts. Among other things they cover:
- persisted rounds, undo and rotations;
- rectangular-board sizing and install assets;
- delayed policy catalogs and proof display;
- the verified model cache;
- neural startup, cancellation, worker recovery and Move now;
- mid-game handoffs, failed writes, and accessibility and touch.

Each suite runs through `scripts/browser_evidence.py`, which keeps
screenshots, console output and traces of a failure in `browser-results/`.
CI uploads that directory.

Run locally, as CI does:

```sh
# The browser smoke, against the built site
bash scripts/build-site.sh _site
BROWSER_SMOKE_ROOT=_site CHROME_BIN=/path/to/chrome-or-edge npm run test:browser

# The Playwright suites, for --browser chromium and again for webkit
python -m pip install -r scripts/browser-requirements.txt
python -m playwright install --with-deps chromium webkit
python scripts/test-browser-evidence.py --browser chromium
python scripts/test-browser-persistence.py
for suite in browser-regressions neural-worker-regressions model-cache-browser-regressions \
    review-browser-regressions rereview-browser-regressions failure-browser-regressions \
    handoff-browser-regressions ui-browser-regressions; do
  python scripts/browser_evidence.py scripts/$suite.py --browser chromium
done
```

CI also passes `--real-model` to `neural-worker-regressions.py`, which
downloads the shipped network for a short real-model game.

Neural lifecycle scenarios deliberately inject a controllable runtime so that
late startup, cancellation and interruption are deterministic. They exercise
the real page, request controller and neural search, not the accuracy or speed
of the shipped model. The existing neural-model tests continue to cover model
assets. Failure screenshots, HTML and page exceptions are uploaded by CI.

## Physical iPhone release check

WebKit with a mobile viewport is not a physical iPhone or an installed Safari
web app. Before claiming a release has been device-tested, perform these
checks on a real iPhone and record the iOS version and commit:

1. Open the deployed site in Safari, use Add to Home Screen, and check the icon
   and standalone launch. Reinstall an old shortcut separately to distinguish
   cached install metadata from the current manifest.
2. Play several moves, leave and relaunch the app, then reload and Undo. Check
   that the same round resumes and scores remain correct.
3. Rotate a 4×10 board to 10×4, test 10×10, change phone orientation, and check
   column taps and controls without horizontal clipping.
4. Start Neural, cancel while downloading and again during startup, and check
   that no late move appears. Retry, then interrupt a search with Move now.
5. Repeat while offline and with interrupted connectivity. A network failure
   must show a recoverable error, not freeze the page or silently change AI.

No physical-device verification is implied by a passing CI run.
