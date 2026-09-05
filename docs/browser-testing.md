# Browser regression testing

The existing Chromium smoke suite remains in place. In addition, the Browser
regressions workflow runs the real page in Chromium and WebKit, with desktop
and 390-pixel-wide touch-enabled mobile contexts. It tests persisted rounds,
undo, rotations, rectangular-board sizing, install assets, delayed policy
catalogs, proof display, cancellation during neural startup, superseded
requests, and Move now accounting.

Run locally:

```sh
python -m pip install -r scripts/browser-requirements.txt
python -m playwright install --with-deps chromium webkit
python scripts/browser-regressions.py --browser chromium
python scripts/browser-regressions.py --browser webkit
node --test tests/review-regressions.test.js
```

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
