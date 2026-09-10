#!/usr/bin/env python3
"""Focused UI/accessibility regressions from the September 2026 UI review."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_KEY = "connect4-chaos.settings.v1"


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


def settings_script(config):
    return f"localStorage.setItem({json.dumps(SETTINGS_KEY)}, JSON.stringify({json.dumps(config)}));"


def check_catalog_recovery(browser, url):
    """Hold retry responses so every assertion observes a deliberate state."""
    perfect = {"rows": 5, "cols": 7, "connect": 4, "opponent": "perfect", "startingPlayer": 1, "chaosMode": False}
    context = browser.new_context(service_workers="block", viewport={"width": 1000, "height": 800})
    context.add_init_script(settings_script(perfect))
    page = context.new_page()
    pending = []
    requests = 0

    def catalog(route):
        nonlocal requests
        requests += 1
        if requests == 1:
            route.fulfill(status=500, body="initial catalog failure")
        else:
            pending.append(route)

    def retry_response():
        # Pump browser events until the intercepted request arrives, rather
        # than sleeping for a guessed network duration.
        deadline = time.monotonic() + 5
        while not pending and time.monotonic() < deadline:
            page.wait_for_timeout(10)
        assert len(pending) == 1, "expected one pending catalog retry"
        return pending.pop()

    page.route("**/data/perfect-classic/manifest.json", catalog)
    page.goto(url)
    page.wait_for_selector(".cell")
    hint = page.locator("#opponentHint")
    opponent = page.locator("#opponentInput")
    submit = page.locator("#settingsForm button[type=submit]")
    expect(hint).to_contain_text("could not be loaded")
    page.evaluate("window.catalogRecoverySentinel = 'same page'")

    # Opening settings starts a new request. It must not be mistaken for the
    # old failed request, and focusing more fields must not duplicate it.
    page.locator("#settingsToggle").click()
    failed_retry = retry_response()
    expect(hint).to_contain_text("Loading the verified policy catalog")
    expect(opponent).to_have_value("perfect")
    expect(page.locator("#activeRulesSummary")).to_contain_text("Perfect AI")
    expect(submit).to_be_disabled()
    page.locator("#rowsInput").focus()
    page.locator("#colsInput").focus()
    assert requests == 2
    failed_retry.fulfill(status=500, body="retry catalog failure")
    expect(hint).to_contain_text("could not be loaded")
    expect(opponent).to_have_value("perfect")
    expect(submit).to_be_disabled()

    # A later successful retry makes Perfect usable on this same page.
    page.locator("#rowsInput").focus()
    successful_retry = retry_response()
    expect(hint).to_contain_text("Loading the verified policy catalog")
    expect(submit).to_be_disabled()
    successful_retry.fulfill(status=200, content_type="application/json",
                             body=(ROOT / "data/perfect-classic/manifest.json").read_text(encoding="utf-8"))
    expect(hint).to_contain_text("Game-theoretically optimal play")
    expect(opponent).to_have_value("perfect")
    expect(page.locator("#perfectOpponentOption")).to_be_enabled()
    expect(submit).to_be_enabled()
    assert page.evaluate("window.catalogRecoverySentinel") == "same page"
    page.locator("#colsInput").focus()
    assert requests == 3
    context.close()


def run(browser_name: str, executable: str | None = None):
    with site() as url, sync_playwright() as pw:
        launch = {"headless": True}
        if executable:
            launch["executable_path"] = executable
        browser = getattr(pw, browser_name).launch(**launch)

        # First-use phones expose rules instead of silently starting a default
        # Medium-AI game with Chaos hidden behind a collapsed panel.
        context = browser.new_context(service_workers='block', viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
        page = context.new_page()
        page.goto(url)
        page.wait_for_selector(".cell")
        assert page.locator("#settingsBody").is_visible()
        assert page.locator("#activeRulesSummary").inner_text().startswith("Classic · 6×7")
        page.locator(".chaos-control").click()
        assert page.locator("#chaosInput").is_checked()
        page.locator("#opponentInput").select_option("human")
        page.locator("#settingsForm button[type=submit]").tap()
        page.wait_for_timeout(50)
        assert page.locator("#settingsBody").is_hidden()
        assert page.locator("#activeRulesSummary").inner_text().startswith("Chaos · 6×7")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert page.evaluate("document.querySelector('#transformToolbar').nextElementSibling === document.querySelector('#boardFrame')")
        context.close()

        # Numeric rule fields reject blanks/non-integers rather than allowing
        # normalizeConfig() to silently substitute defaults.
        context = browser.new_context(service_workers='block', viewport={"width": 1000, "height": 800})
        page = context.new_page()
        page.goto(url)
        page.wait_for_selector(".cell")
        page.locator("#rowsInput").fill("")
        assert page.locator("#rowsInput").get_attribute("aria-invalid") == "true"
        assert "required" in page.locator("#rowsInput").locator("xpath=..").locator(".field-error").inner_text().lower()
        assert page.locator("#settingsForm button[type=submit]").is_disabled()
        assert page.locator("#gameBoard").get_attribute("aria-rowcount") == "6"
        page.locator("#rowsInput").fill("6.5")
        assert "whole number" in page.locator("#rowsInput").locator("xpath=..").locator(".field-error").inner_text().lower()
        page.locator("#rowsInput").fill("6")
        page.locator("#colsInput").fill("7")
        page.locator("#connectInput").fill("6")
        page.locator("#rowsInput").fill("4")
        page.locator("#colsInput").fill("4")
        assert page.locator("#connectInput").input_value() == "6"
        assert page.locator("#connectInput").get_attribute("aria-invalid") == "true"
        assert page.locator("#settingsForm button[type=submit]").is_disabled()
        page.locator("#rowsInput").fill("6")
        assert page.locator("#connectInput").input_value() == "6"
        assert page.locator("#connectInput").get_attribute("aria-invalid") is None
        context.close()

        check_catalog_recovery(browser, url)

        # During an AI turn, board focus remains reachable for navigation but
        # is explicitly disabled and no longer advertises impossible move keys.
        ai_first = {"rows": 6, "cols": 7, "connect": 4, "opponent": "medium", "startingPlayer": 2, "chaosMode": False}
        context = browser.new_context(service_workers='block', viewport={"width": 1000, "height": 800})
        context.add_init_script(settings_script(ai_first) + "\nwindow.Worker=class extends EventTarget{postMessage(){} terminate(){}};")
        page = context.new_page()
        page.goto(url)
        page.wait_for_selector(".cell")
        page.wait_for_timeout(50)
        assert page.locator("#gameBoard").get_attribute("aria-disabled") == "true"
        assert page.locator("#gameBoard").get_attribute("aria-activedescendant") is None
        assert "AI is thinking" in page.locator("#boardInstructions").inner_text()
        assert page.locator("#keyboardHelp").is_hidden()
        page.locator("#gameBoard").focus()
        assert "AI is thinking" in page.locator("#selectedColumnStatus").inner_text()
        context.close()

        # Every AI failure has a generic recovery route and announces its reason.
        context = browser.new_context(service_workers='block', viewport={"width": 1000, "height": 800})
        context.add_init_script(settings_script(ai_first) + r"""
          window.Worker = class extends EventTarget {
            postMessage({requestId}) {
              setTimeout(() => this.dispatchEvent(new MessageEvent('message', {data: {
                kind: 'error', requestId, error: 'Injected AI failure'
              }})), 0);
            }
            terminate() {}
          };
        """)
        page = context.new_page()
        page.goto(url)
        page.wait_for_selector("#aiRecovery:not([hidden])")
        assert page.locator("#changeOpponentButton").is_visible()
        assert "primary-button" in (page.locator("#retryAiButton").get_attribute("class") or "")
        assert "text-button" in (page.locator("#changeOpponentButton").get_attribute("class") or "")
        assert page.locator("#switchBrutalButton").text_content() == "Use Brutal"
        assert page.locator("#aiErrorText").get_attribute("role") == "alert"
        assert page.locator("#aiErrorText").get_attribute("aria-live") == "assertive"
        assert "Injected AI failure" in page.locator("#aiErrorText").inner_text()
        page.locator("#changeOpponentButton").click()
        assert page.locator("#settingsBody").is_visible()
        for _ in range(20):
            if page.locator("#opponentInput").evaluate("el => document.activeElement === el"):
                break
            page.wait_for_timeout(25)
        else:
            raise AssertionError("opponent control did not receive focus")
        context.close()

        # The visible blue board—including its padding/gaps—is a column target,
        # matching the touch copy rather than requiring a direct hole hit.
        # Returning desktop games use compact one-line rules chrome and score tiles.
        chaos_human = {"rows": 6, "cols": 7, "connect": 4, "opponent": "human", "startingPlayer": 1, "chaosMode": True}
        context = browser.new_context(service_workers='block', viewport={"width": 1200, "height": 900})
        context.add_init_script(settings_script(chaos_human))
        page = context.new_page()
        page.goto(url)
        page.wait_for_selector(".cell")
        assert page.locator("#settingsBody").is_hidden()
        assert page.locator("#setupPanel .section-heading > div").first.is_hidden()
        assert page.locator("#activeRulesSummary").evaluate("el => getComputedStyle(el).whiteSpace") == "nowrap"
        assert page.evaluate("document.querySelector('#boardFrame').nextElementSibling === document.querySelector('#transformToolbar')")
        columns = page.locator(".score-card").first.evaluate("el => getComputedStyle(el).gridTemplateColumns")
        assert len(columns.split()) == 1
        context.close()

        human = {"rows": 6, "cols": 7, "connect": 4, "opponent": "human", "startingPlayer": 1, "chaosMode": False}
        context = browser.new_context(service_workers='block', viewport={"width": 1000, "height": 800})
        context.add_init_script(settings_script(human))
        page = context.new_page()
        page.goto(url)
        page.wait_for_selector(".cell")
        box = page.locator("#gameBoard").bounding_box()
        page.mouse.click(box["x"] + box["width"] * (4.5 / 7), box["y"] + 3)
        page.wait_for_timeout(50)
        assert page.locator(".cell.red").count() == 1
        assert page.locator("#cell-5-4").get_attribute("class").find("red") >= 0
        context.close()

        # Exact-table download copy does not promise worker bytes persist forever.
        context = browser.new_context(service_workers='block', viewport={"width": 1000, "height": 800})
        context.add_init_script(settings_script(human))
        page = context.new_page()
        page.goto(url)
        page.wait_for_selector(".cell")
        page.evaluate(r"""() => {
          import('./src/download-gate.js').then(({requestDownload}) => {
            window.uiReviewDownload = requestDownload({
              id: 'ui-review-copy-' + Date.now(), title: 'Exact table',
              description: 'Copy test', bytes: 9000000, remember: false,
              persistence: 'Reused for this AI session; your browser may cache the download.'
            });
          });
        }""")
        page.wait_for_selector("#downloadDialog[open]")
        detail = page.locator("#downloadDetail").inner_text()
        assert "Reused for this AI session" in detail
        assert "kept by your browser" not in detail
        page.locator("#downloadCancelButton").click()
        context.close()

        # Operational score errors are styled and disappear after the same
        # operation succeeds; persistent-storage fallback warnings are separate.
        context = browser.new_context(service_workers='block', viewport={"width": 1000, "height": 800})
        context.add_init_script(settings_script(human) + r"""
          (() => {
            const original = IDBDatabase.prototype.transaction;
            IDBDatabase.prototype.transaction = function(...args) {
              if (window.failNextScoreTransaction) {
                window.failNextScoreTransaction = false;
                throw new DOMException('Injected score failure', 'NotAllowedError');
              }
              return original.apply(this, args);
            };
          })();
        """)
        page = context.new_page()
        page.goto(url)
        page.wait_for_selector(".cell")
        page.wait_for_timeout(100)
        page.evaluate("window.failNextScoreTransaction = true")
        page.locator("#resetScoreButton").click()
        assert page.locator("#resetScoreButton").inner_text() == "Confirm reset"
        assert page.locator("#resetScoreButton").get_attribute("data-confirming") == "true"
        page.locator("#resetScoreButton").click()
        page.wait_for_timeout(50)
        notice = page.locator("#scoreStorageStatus")
        assert notice.is_visible()
        assert notice.get_attribute("data-tone") == "error"
        page.locator("#resetScoreButton").click()
        assert page.locator("#resetScoreButton").inner_text() == "Confirm reset"
        page.locator("#resetScoreButton").click()
        page.wait_for_timeout(50)
        assert notice.is_hidden()
        assert page.locator("#resetScoreButton").inner_text() == "Reset score"
        context.close()

        browser.close()
    print(f"PASS [{browser_name}] UI review regressions", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", choices=("chromium", "webkit"), default="chromium")
    parser.add_argument("--executable")
    args = parser.parse_args()
    run(args.browser, args.executable)


if __name__ == "__main__":
    main()
