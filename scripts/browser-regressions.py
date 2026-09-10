#!/usr/bin/env python3
"""Real-page regression tests. Run against Chromium or WebKit, including touch.

Install: python -m pip install -r scripts/browser-requirements.txt
         python -m playwright install --with-deps chromium webkit
Run:     python scripts/browser-regressions.py --browser chromium
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import traceback
from time import monotonic

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"rows": 6, "cols": 7, "connect": 4, "opponent": "human", "startingPlayer": 1, "chaosMode": False}


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


@contextmanager
def site():
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(ROOT)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


NEURAL_STUB = """
let phase = 'idle';
export const DOWNLOAD_BYTES = { model: 1, runtime: 1 };
export const neuralLoadState = () => phase;
export const invalidateNeuralNetwork = () => { phase = 'idle'; };
export const cancelNeuralLoad = () => { phase = 'idle'; window.cancelledLoads = (window.cancelledLoads || 0) + 1; };
export const simulationsFor = () => 75;
export const recordSearch = (_network, _elapsed, evaluations) => { window.recordedEvaluations = evaluations; };
export async function loadNeuralNetwork({ onProgress }) {
  phase = 'loading';
  onProgress({ stage: 'session', backend: 'wasm' });
  const network = { backend: 'wasm', perEvaluation: 10, evaluate: async () => {
    window.evaluations = (window.evaluations || 0) + 1;
    await new Promise(resolve => setTimeout(resolve, 15));
    return { policy: new Float32Array(13), value: new Float32Array(3), q: new Float32Array(39) };
  }};
  if (window.automaticNeural) { phase = 'ready'; return network; }
  // Deliberately ignores cancellation to exercise late native completion.
  return new Promise(resolve => { window.finishNeuralStartup = () => { phase = 'ready'; resolve(network); }; });
}
"""


def wait_for(page, expression: str, timeout_ms: int = 30_000):
    # Playwright's in-page wait_for_function evaluates predicates from a timer,
    # which conflicts with this application's no-unsafe-eval CSP. Poll through
    # the automation evaluation API instead; keep the page policy unchanged.
    deadline = monotonic() + timeout_ms / 1000
    while monotonic() < deadline:
        if page.evaluate(expression):
            return
        page.wait_for_timeout(50)
    raise AssertionError(f"Timed out waiting for: {expression}")


def run(browser_name: str, executable: str | None):
    results = []
    failures = []
    page_number = 0
    with site() as url, sync_playwright() as pw:
        launch = {"headless": True}
        if executable:
            launch["executable_path"] = executable
        browser = getattr(pw, browser_name).launch(**launch)

        @contextmanager
        def page_for(config=None, mobile=False, init=None, runtime=None,
                     service_workers="block"):
            nonlocal page_number
            page_number += 1
            context = browser.new_context(
                service_workers=service_workers,
                viewport={"width": 390 if mobile else 1280, "height": 844 if mobile else 800},
                is_mobile=mobile, has_touch=mobile, reduced_motion="reduce",
            )
            if config is not None:
                context.add_init_script("if (!localStorage.getItem('connect4-chaos.settings.v1')) "
                    f"localStorage.setItem('connect4-chaos.settings.v1', JSON.stringify({json.dumps(config)}));")
            if init:
                context.add_init_script(init)
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            if runtime:
                page.route("**/src/neural-client.js", lambda route: route.fulfill(
                    status=200, content_type="text/javascript", body=runtime))
            try:
                yield page, errors
                assert not errors, f"Uncaught page errors: {errors}"
            except Exception as error:
                failures.append(f"Context {page_number}: {traceback.format_exc()}")
                print(f"FAIL [{browser_name}] context {page_number}: {error}", flush=True)
                output = ROOT / 'browser-results'
                output.mkdir(exist_ok=True)
                try:
                    diagnostics = page.evaluate("""() => {
                      const frame = document.querySelector('#boardFrame');
                      const board = document.querySelector('#gameBoard');
                      const style = frame && getComputedStyle(frame);
                      return { viewport: [innerWidth, innerHeight], frame: frame?.getBoundingClientRect().toJSON(),
                        board: board?.getBoundingClientRect().toJSON(), frameStyle: style && {
                          width: style.width, maxWidth: style.maxWidth, transform: style.transform,
                          rows: style.getPropertyValue('--rows'), cols: style.getPropertyValue('--cols'),
                          height: style.getPropertyValue('--board-height'), cell: style.getPropertyValue('--height-cell')
                        }};
                    }""")
                    (output / f'{browser_name}-{page_number}-diagnostics.json').write_text(
                        json.dumps({"failure": traceback.format_exc(), "pageErrors": errors, "layout": diagnostics}, indent=2))
                    page.screenshot(path=str(output / f'{browser_name}-{page_number}.png'), full_page=True)
                    (output / f'{browser_name}-{page_number}.html').write_text(page.content())
                    (output / f'{browser_name}-{page_number}-errors.json').write_text(json.dumps(errors))
                except Exception:
                    pass
            finally:
                context.close()

        def passed(name):
            results.append(name)
            print(f"PASS [{browser_name}] {name}", flush=True)

        with page_for() as (page, _):
            page.goto(url)
            page.wait_for_selector(".cell")
            assert page.locator(".cell").count() == 42
            summary = page.locator("#activeRulesSummary").inner_text()
            assert summary.startswith("Classic · 6×7"), summary
            passed("fresh desktop launch")

        with page_for({**CONFIG, "rows": 4, "cols": 10, "chaosMode": True}) as (page, _):
            page.goto(url)
            page.locator("#rotateCwButton").click()
            wait_for(page, "document.querySelector('#gameBoard').getAttribute('aria-rowcount') === '10'")
            wait_for(page, "!document.querySelector('#undoButton').disabled")
            # Wait for the final layout, not an intermediate frame of the transform.
            wait_for(page, "document.querySelector('#gameBoard').getBoundingClientRect().height <= 515")
            size = page.locator("#gameBoard").bounding_box()
            assert size["height"] <= 800 - 18 * 16 + 3, size
            assert page.locator("#gameBoard").get_attribute("aria-colcount") == "4"
            page.locator("#undoButton").click()
            assert page.locator("#gameBoard").get_attribute("aria-rowcount") == "4"
            page.locator("#restartButton").click()
            assert "Move 0" in page.locator("#moveInfo").text_content()
            passed("tall rotated board fits desktop; undo and restart restore orientation")

        with page_for(CONFIG, mobile=True) as (page, _):
            page.goto(url)
            assert page.locator("#settingsBody").is_hidden()
            page.locator("#cell-5-2").tap()
            wait_for(page, "document.querySelector('#moveInfo').textContent.includes('Move 1') && !document.querySelector('#undoButton').disabled")
            page.locator("#cell-5-3").tap()
            wait_for(page, "document.querySelector('#moveInfo').textContent.includes('Move 2') && !document.querySelector('#undoButton').disabled")
            page.reload()
            wait_for(page, "document.querySelector('#moveInfo').textContent.includes('Move 2')")
            assert page.locator(".cell.red").count() == 1
            assert page.locator(".cell.yellow").count() == 1
            page.locator("#undoButton").tap()
            page.reload()
            wait_for(page, "document.querySelector('#moveInfo').textContent.includes('Move 1')")
            assert page.locator(".cell.yellow").count() == 0
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            icon = page.locator('link[rel="apple-touch-icon"]').get_attribute("href")
            assert page.request.get(url + icon.removeprefix("./")).ok
            manifest = page.request.get(url + "manifest.json").json()
            assert manifest["display"] == "standalone"
            passed("mobile touch, reload, undo, no overflow and install assets")

        mobile_failures_before = len(failures)
        for rows, cols in [(10, 10), (10, 4), (4, 10)]:
            with page_for({**CONFIG, "rows": rows, "cols": cols, "chaosMode": True}, mobile=True) as (page, _):
                page.goto(url)
                page.locator("#rotateCwButton").tap()
                wait_for(page, f"document.querySelector('#gameBoard').getAttribute('aria-rowcount') === '{cols}' && !document.querySelector('#undoButton').disabled")
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (rows, cols)
                box = page.locator(".cell").first.bounding_box()
                assert box["width"] >= 24, box
                assert abs(box["width"] - box["height"]) < 1, box
        if len(failures) == mobile_failures_before:
            passed("mobile 10×10 and rectangular rotations retain usable square cells")

        worker = """
        window.Worker = class extends EventTarget {
          postMessage({requestId}) {
            this.requestId = requestId; window.testWorker = this;
            setTimeout(() => this.dispatchEvent(new MessageEvent('message', {data: {
              kind: 'progress', requestId, progress: {solver: 'bitboard-exact', score: 0, solved: false, nodes: 5}
            }})), 0);
          }
          terminate() {}
        };
        """
        with page_for({**CONFIG, "opponent": "medium", "startingPlayer": 2}, init=worker) as (page, _):
            page.goto(url)
            wait_for(page, "document.querySelector('#exactBadge').textContent === 'Searching'")
            assert "draw" not in page.locator("#exactResultText").inner_text()
            page.evaluate("""testWorker.dispatchEvent(new MessageEvent('message', {data: {
              kind: 'result', requestId: testWorker.requestId,
              result: {action: {type: 'drop', column: 3}, solver: 'bitboard-exact', solved: true, score: 0}
            }}))""")
            wait_for(page, "document.querySelector('#statusText').textContent === 'Red to move'")
            assert page.locator("#exactBadge").text_content() == "Proved"
            page.locator("#cell-5-2").click()
            wait_for(page, "document.querySelector('#exactBadge').textContent === 'Searching'")
            assert "draw" not in page.locator("#exactResultText").inner_text()
            passed("unfinished proof is not a draw; completed proof invalidates after human move")

        with page_for({**CONFIG, "opponent": "neural", "startingPlayer": 2}, mobile=True, runtime=NEURAL_STUB) as (page, _):
            page.goto(url)
            page.locator("#downloadConfirmButton").tap()
            wait_for(page, "typeof finishNeuralStartup === 'function'")
            page.locator("#downloadCancelButton").tap()
            assert not page.locator("#downloadDialog").is_visible()
            wait_for(page, "document.querySelector('#statusText').textContent === 'AI unavailable'")
            page.evaluate("finishNeuralStartup()")
            page.wait_for_timeout(150)
            assert page.locator(".cell.yellow").count() == 0
            assert "Move 0" in page.locator("#moveInfo").text_content()
            assert page.evaluate("cancelledLoads") == 1
            page.locator("#settingsToggle").tap()
            page.locator("#opponentInput").select_option("human")
            page.locator('#settingsForm [type="submit"]').tap()
            assert page.locator("#aiRecovery").is_hidden()
            passed("mobile Cancel during startup prevents late moves and restores human controls")

        with page_for({**CONFIG, "opponent": "neural", "startingPlayer": 2}, runtime=NEURAL_STUB) as (page, _):
            page.goto(url)
            page.locator("#downloadConfirmButton").click()
            wait_for(page, "typeof finishNeuralStartup === 'function'")
            page.evaluate("window.oldStartup = finishNeuralStartup; document.querySelector('#restartButton').click()")
            wait_for(page, "finishNeuralStartup !== oldStartup")
            page.evaluate("oldStartup()")
            page.wait_for_timeout(100)
            assert page.locator("#downloadDialog").is_visible(), "old completion closed the replacement dialog"
            page.locator("#downloadCancelButton").click()
            assert page.locator(".cell.yellow").count() == 0
            passed("restart invalidates an old startup without closing the new dialog")

        with page_for({**CONFIG, "opponent": "neural", "startingPlayer": 2},
                      runtime=NEURAL_STUB, init="window.automaticNeural = true;") as (page, _):
            page.goto(url)
            page.locator("#downloadConfirmButton").click()
            wait_for(page, "(window.evaluations || 0) >= 3")
            page.locator("#moveNowButton").click()
            wait_for(page, "document.querySelector('#statusText').textContent === 'Red to move'")
            info = page.locator("#searchInfo").text_content()
            actual = page.evaluate("window.recordedEvaluations")
            assert 1 < actual < 75, (actual, info)
            assert f"{actual - 1} simulations" in info, (actual, info)
            assert page.locator("#evaluationDescription").inner_text() == "Heuristic position estimate"
            passed("Move now reports actual work; neural label accurately describes the heuristic")

        # The site makes itself cross-origin isolated through a service worker
        # so WebAssembly inference can use several threads. The embedder
        # policy that isolation requires also governs Web Workers, and this
        # game runs its AI in one: if the worker's script were served without
        # a resource policy it would fail to start and no opponent would move
        # at all. Every other scenario blocks service workers so its request
        # stubs are honoured; this one keeps them, which is the only place
        # that combination is exercised.
        with page_for({**CONFIG, "opponent": "medium", "startingPlayer": 1},
                      service_workers="allow") as (page, _):
            page.goto(url)
            wait_for(page, "!!navigator.serviceWorker.controller")
            page.goto(url)          # isolation is decided when a page is navigated to
            wait_for(page, "self.crossOriginIsolated === true")
            page.locator("#cell-5-3").click()
            # The AI answering at all means its worker started under the
            # embedder policy; the move number is the proof.
            wait_for(page, "document.querySelector('#moveInfo').textContent.includes('Move 2')",
                     timeout_ms=60_000)
            assert page.evaluate("self.crossOriginIsolated") is True
            passed("the isolation worker leaves the AI's own Web Worker able to start")

        # Resolve a delayed catalog after the user chooses another opponent.
        with page_for({**CONFIG, "rows": 4, "cols": 4, "opponent": "perfect"}) as (page, _):
            pending = []
            page.route("**/data/perfect-classic/manifest.json", lambda route: pending.append(route))
            page.goto(url, wait_until="domcontentloaded")
            page.locator("#settingsToggle").click()
            assert page.locator("#opponentInput").input_value() == "perfect"
            assert page.locator('#settingsForm [type="submit"]').is_disabled()
            page.locator("#opponentInput").select_option("easy")
            assert pending
            pending[0].fulfill(status=200, content_type="application/json",
                               body=(ROOT / "data/perfect-classic/manifest.json").read_text())
            wait_for(page, "!document.querySelector('#perfectOpponentOption').disabled")
            assert page.locator("#opponentInput").input_value() == "easy"
            assert "forgiving" in page.locator("#opponentHint").inner_text()
            passed("late catalog cannot override a newer opponent selection")

        browser.close()
    if failures:
        raise AssertionError("Browser regression failures:\n" + "\n".join(failures))
    print(f"{len(results)} browser regression scenarios passed ({browser_name}).", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", choices=["chromium", "webkit"], default="chromium")
    parser.add_argument("--executable", default=None)
    args = parser.parse_args()
    run(args.browser, args.executable)