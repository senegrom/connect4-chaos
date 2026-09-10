#!/usr/bin/env python3
"""Real worker handoffs and deterministic restart-during-animation regressions."""
import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('browser_helpers', ROOT/'scripts/browser-regressions.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

NEURAL = """
let moves = 0;
export async function runNeuralRequest(_request, callbacks) {
  if (++moves === 1) callbacks.finish({action:{type:'drop',column:0},solver:'neural',solved:false});
  else callbacks.fail('Injected Neural failure for handoff coverage.');
}
"""


def run(name, executable=None):
    with helpers.site() as url, sync_playwright() as pw:
        browser = getattr(pw, name).launch(headless=True, **({'executable_path': executable} if executable else {}))
        @contextmanager
        def page_for(config, *, normal_motion=False):
            ctx = browser.new_context(service_workers='block', reduced_motion='no-preference' if normal_motion else 'reduce')
            ctx.add_init_script(f"if (!localStorage.getItem('connect4-chaos.settings.v1')) localStorage.setItem('connect4-chaos.settings.v1', JSON.stringify({json.dumps(config)}));")
            page = ctx.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            try:
                yield ctx, page
                assert not errors, errors
            except Exception:
                out = ROOT/'browser-results'; out.mkdir(exist_ok=True)
                page.screenshot(path=str(out/f'handoff-{name}.png'), full_page=True)
                raise
            finally:
                ctx.close()

        config = {**helpers.CONFIG, 'opponent': 'neural', 'chaosMode': True}
        with page_for(config) as (ctx, page):
            ctx.route('**/src/neural-app.js', lambda route: route.fulfill(content_type='text/javascript', body=NEURAL))
            # Observe, but do not replace, the real worker and its table loaders.
            page.add_init_script("""
              const NativeWorker = Worker;
              window.aiRequests = [];
              window.Worker = class extends NativeWorker {
                postMessage(data, ...args) {
                  aiRequests.push(structuredClone(data));
                  super.postMessage(data, ...args);
                }
              };
            """)
            page.goto(url); page.wait_for_selector('.cell')
            page.locator('#cell-5-0').click()
            helpers.wait_for(page, "document.querySelector('#moveInfo').textContent.includes('Move 2') && document.querySelector('#statusText').textContent === 'Red to move'")
            page.locator('#cell-5-0').click()
            helpers.wait_for(page, "!document.querySelector('#aiRecovery').hidden")
            page.locator('#switchBrutalButton').click()
            helpers.wait_for(page, "document.querySelector('#moveInfo').textContent.includes('Move 4') && document.querySelector('#statusText').textContent === 'Red to move'", 90_000)
            assert page.evaluate('aiRequests.at(-1).options.useChaosPolicy') is False
            assert page.evaluate("JSON.parse(localStorage.getItem('connect4-chaos.round.v1')).useChaosPolicy") is False
            assert page.locator('#aiRecovery').is_hidden()
            page.reload()
            helpers.wait_for(page, "document.querySelector('#moveInfo').textContent.includes('Move 4') && document.querySelector('#statusText').textContent === 'Red to move'")
            page.locator('#undoButton').click()
            helpers.wait_for(page, "document.querySelector('#moveInfo').textContent.includes('Move 2') && document.querySelector('#gameBoard').getAttribute('aria-busy') === 'false'")
            page.locator('#cell-5-0').click()
            helpers.wait_for(page, "document.querySelector('#moveInfo').textContent.includes('Move 4') && document.querySelector('#statusText').textContent === 'Red to move'", 90_000)
            assert page.evaluate('aiRequests.at(-1).options.useChaosPolicy') is False
            # A fresh Brutal round must still select the strict certified route.
            page.locator('#restartButton').click()
            page.locator('#cell-5-0').click()
            helpers.wait_for(page, "document.querySelector('#moveInfo').textContent.includes('Move 2') && document.querySelector('#statusText').textContent === 'Red to move'")
            assert page.evaluate('aiRequests.at(-1).options.useChaosPolicy') is True
            assert page.evaluate("JSON.parse(localStorage.getItem('connect4-chaos.round.v1')).useChaosPolicy") is True
            page.reload()
            helpers.wait_for(page, "document.querySelector('#moveInfo').textContent.includes('Move 2') && document.querySelector('#statusText').textContent === 'Red to move'")
            page.locator('#cell-5-0').click()
            helpers.wait_for(page, "document.querySelector('#moveInfo').textContent.includes('Move 4') && document.querySelector('#statusText').textContent === 'Red to move'")
            assert page.evaluate('aiRequests.at(-1).options.useChaosPolicy') is True
            print(f'PASS [{name}] real off-policy Brutal handoff, reload/Undo, and fresh certified restart', flush=True)

        for button, incoming, out_ms, in_ms in [
            ('rotateCwButton', 'anim-cw-in', 280, 360),
            ('rotateCcwButton', 'anim-ccw-in', 280, 360),
            ('flipButton', 'anim-flip-in', 320, 420),
        ]:
            with page_for({**helpers.CONFIG, 'chaosMode': True}, normal_motion=True) as (_ctx, page):
                # Hold only application animation timers. Releasing named promises
                # avoids flaky timing while retaining the real controller and CSS.
                page.add_init_script(f"""
                  const nativeTimeout = window.setTimeout.bind(window);
                  window.animationTimers = [];
                  window.setTimeout = (fn, ms, ...args) => {{
                    if ([{out_ms}, {in_ms}].includes(ms)) {{
                      const timer = {{ms, release: () => fn(...args)}};
                      animationTimers.push(timer);
                      return -animationTimers.length;
                    }}
                    return nativeTimeout(fn, ms, ...args);
                  }};
                  window.releaseAnimation = ms => {{
                    const at = animationTimers.findIndex(t => t.ms === ms);
                    if (at < 0) throw Error('Missing animation timer ' + ms);
                    animationTimers.splice(at,1)[0].release();
                  }};
                """)
                page.goto(url); page.wait_for_selector('.cell')
                page.locator('#'+button).click()
                page.evaluate(f'releaseAnimation({out_ms})')
                helpers.wait_for(page, f"document.querySelector('#boardFrame').classList.contains('{incoming}')")
                page.locator('#restartButton').click()
                page.locator('#'+button).click()
                page.evaluate(f'releaseAnimation({out_ms})')
                helpers.wait_for(page, f"document.querySelector('#boardFrame').classList.contains('{incoming}')")
                page.evaluate(f'releaseAnimation({in_ms})')  # old round completes first
                assert page.evaluate(f"document.querySelector('#boardFrame').classList.contains('{incoming}')")
                assert page.locator('#gameBoard').get_attribute('aria-busy') == 'true'
                page.evaluate(f'releaseAnimation({in_ms})')
                helpers.wait_for(page, "document.querySelector('#gameBoard').getAttribute('aria-busy') === 'false'")
                assert not page.evaluate(f"document.querySelector('#boardFrame').classList.contains('{incoming}')")
                assert 'Move 1' in page.locator('#moveInfo').text_content()
                print(f'PASS [{name}] stale {button} callback cannot remove the new animation', flush=True)
        browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser', choices=['chromium', 'webkit'], default='chromium')
    parser.add_argument('--executable')
    args = parser.parse_args()
    run(args.browser, args.executable)
