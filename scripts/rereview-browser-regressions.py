#!/usr/bin/env python3
"""Fault-inject actual IndexedDB transactions and exercise the real Undo UI."""
import argparse
import importlib.util
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('browser_helpers', ROOT/'scripts/browser-regressions.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

# Abort a put *after* its request succeeds, before transaction completion.
# This exercises real database rollback, not just a mocked rejected promise.
ABORT_HOOK = """(() => {
  window.abortNextScoreWrite = false;
  window.abortedScoreWrites = 0;
  const put = IDBObjectStore.prototype.put;
  IDBObjectStore.prototype.put = function (...args) {
    const request = put.apply(this, args);
    if (this.name === 'scores' && window.abortNextScoreWrite) {
      window.abortNextScoreWrite = false;
      request.addEventListener('success', () => {
        window.abortedScoreWrites++;
        this.transaction.abort();
      });
    }
    return request;
  };
})();"""


def run(name, executable=None):
    with helpers.site() as url, sync_playwright() as pw:
        browser = getattr(pw, name).launch(headless=True, **({'executable_path': executable} if executable else {}))
        ctx = browser.new_context(reduced_motion='reduce')
        origin = json.dumps(url.rstrip('/'))
        ctx.add_init_script(f"if (location.origin === {origin} && !localStorage.getItem('connect4-chaos.settings.v1')) localStorage.setItem('connect4-chaos.settings.v1', JSON.stringify({json.dumps(helpers.CONFIG)}));")
        ctx.add_init_script(ABORT_HOOK)
        page = ctx.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        wait = lambda expression: helpers.wait_for(page, expression)
        try:
            page.goto(url)
            page.wait_for_selector('.cell')
            for move, col in enumerate((0,1,0,1,0,1,0), 1):
                page.locator(f'#cell-5-{col}').click()
                wait(f"document.querySelector('#moveInfo').textContent.includes('Move {move}') && document.querySelector('#gameBoard').getAttribute('aria-busy') === 'false'")
            wait("document.querySelector('#resultDialog').open && document.querySelector('#redScore').textContent === '1'")
            page.locator('#reviewBoardButton').click()
            # Ordinary finished games are not resumed; pending reversals are.
            assert page.evaluate("localStorage.getItem('connect4-chaos.round.v1')") is None
            page.evaluate('abortNextScoreWrite = true')
            page.locator('#undoButton').click()
            wait("abortedScoreWrites === 1 && !document.querySelector('#undoButton').disabled")
            assert page.locator('#statusText').text_content() == 'Red wins!'
            assert page.locator('#redScore').text_content() == '1'
            saved = page.evaluate("JSON.parse(localStorage.getItem('connect4-chaos.round.v1'))")
            assert saved['pendingScoreUndo'] is True
            assert saved['history'][-1]['status'] == 'won'
            assert saved['history'][-1]['scoreReceipt']
            # A fresh page must recover the same receipt, then retry exactly once.
            page.reload()
            wait("document.querySelector('#statusText').textContent === 'Red wins!' && document.querySelector('#redScore').textContent === '1'")
            if page.locator('#resultDialog').is_visible():
                page.locator('#reviewBoardButton').click()
            page.locator('#undoButton').click()
            wait("document.querySelector('#redScore').textContent === '0' && document.querySelector('#gameBoard').getAttribute('aria-busy') === 'false'")
            assert page.locator('#statusText').text_content() == 'Red to move'
            assert 'Move 6' in page.locator('#moveInfo').text_content()
            page.locator('#undoButton').click()
            wait("document.querySelector('#moveInfo').textContent.includes('Move 5') && document.querySelector('#gameBoard').getAttribute('aria-busy') === 'false'")
            assert page.locator('#redScore').text_content() == '0'
            assert not errors, errors
            print(f'PASS [{name}] aborted IndexedDB Undo preserves board and receipt; reload/retry reverses once', flush=True)
        except Exception:
            output = ROOT/'browser-results'; output.mkdir(exist_ok=True)
            try: page.screenshot(path=str(output/f'undo-transaction-{name}.png'), full_page=True)
            except Exception: pass
            raise
        finally:
            ctx.close()
            browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser', choices=['chromium', 'webkit'], default='chromium')
    parser.add_argument('--executable')
    args = parser.parse_args()
    run(args.browser, args.executable)
