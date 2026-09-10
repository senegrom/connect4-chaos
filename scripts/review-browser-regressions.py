#!/usr/bin/env python3
"""Real-browser coverage for atomic results, loading recovery and pinned consent."""
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
wait_for = helpers.wait_for
CONFIG = helpers.CONFIG


def run(name, executable=None):
    results = []
    with helpers.site() as url, sync_playwright() as pw:
        browser = getattr(pw, name).launch(headless=True, **({'executable_path': executable} if executable else {}))
        @contextmanager
        def context(config=CONFIG, legacy=None):
            ctx = browser.new_context(service_workers='block', reduced_motion='reduce')
            origin = json.dumps(url.rstrip('/'))
            ctx.add_init_script(f"if (location.origin === {origin} && !localStorage.getItem('connect4-chaos.settings.v1')) localStorage.setItem('connect4-chaos.settings.v1', JSON.stringify({json.dumps(config)}));")
            if legacy:
                ctx.add_init_script(f"if (location.origin === {origin} && !localStorage.getItem('connect4-chaos.scores.v1')) localStorage.setItem('connect4-chaos.scores.v1', JSON.stringify({json.dumps(legacy)}));")
            errors = []
            ctx.on('page', lambda p: p.on('pageerror', lambda error: errors.append(str(error))))
            try:
                yield ctx
                assert not errors, errors
            except Exception:
                out = ROOT/'browser-results'; out.mkdir(exist_ok=True)
                for i, page in enumerate(ctx.pages):
                    try: page.screenshot(path=str(out/f'review-{name}-{len(results)}-{i}.png'), full_page=True)
                    except Exception: pass
                raise
            finally: ctx.close()
        def launch(ctx):
            page = ctx.new_page(); page.goto(url); page.wait_for_selector('.cell'); return page
        def passed(label):
            results.append(label); print(f'PASS [{name}] {label}', flush=True)
        def score(page, value):
            wait_for(page, f"document.querySelector('#redScore').textContent === '{value}'")
        def win(page):
            for move, col in enumerate((0, 1, 0, 1, 0, 1, 0), 1):
                page.locator(f'#cell-5-{col}').click()
                wait_for(page, f"document.querySelector('#moveInfo').textContent.includes('Move {move}') && document.querySelector('#gameBoard').getAttribute('aria-busy') === 'false'")
            page.locator('#reviewBoardButton').click()

        with context(legacy={'1': 7, '2': 3, 'draw': 2}) as ctx:
            a, b = launch(ctx), launch(ctx)
            score(a, 7); score(b, 7)
            for page, prefix in ((a, 'a'), (b, 'b')):
                page.evaluate("""async prefix => {
                  const {createScoreStore} = await import('./src/score-store.js');
                  window.store = createScoreStore();
                  window.writes = Promise.all(Array.from({length: 40}, (_, n) => store.record(`${prefix}-${n}`, 1)));
                }""", prefix)
            for page in (a, b):
                page.evaluate('async () => { await writes; }')
            for page in (a, b):
                page.evaluate('dispatchEvent(new Event("focus"))')
                score(page, 87)
            revision = a.evaluate('async () => (await store.read()).revision')
            a.locator('#cell-5-2').click()
            wait_for(a, "document.querySelector('#moveInfo').textContent.includes('Move 1') && document.querySelector('#gameBoard').getAttribute('aria-busy') === 'false'")
            assert a.evaluate('async () => (await store.read()).revision') == revision
            b.evaluate("localStorage.setItem('connect4-chaos.scores.v1', JSON.stringify({'1':0})); dispatchEvent(new Event('focus'))")
            a.reload(); score(a, 87)
            passed('simultaneous writers, legacy migration, ordinary move and reload preserve all 80 new results')

        with context() as ctx:
            a = launch(ctx); win(a); score(a, 1)
            a.locator('#resetScoreButton').click()
            assert a.locator('#resetScoreButton').inner_text() == 'Confirm reset'
            a.locator('#resetScoreButton').click(); score(a, 0)
            b = launch(ctx); win(b); score(b, 1); score(a, 1)
            a.locator('#undoButton').click()
            wait_for(a, "document.querySelector('#moveInfo').textContent.includes('Move 6')")
            score(a, 1)
            a.reload(); score(a, 1)
            b.locator('#undoButton').click(); score(b, 0); score(a, 0)
            passed('real wins: Reset invalidates old Undo; new-round Undo reverses only its own result')

        chaos = {**CONFIG, 'rows': 4, 'cols': 5, 'connect': 5, 'chaosMode': True, 'opponent': 'perfect', 'startingPlayer': 2}
        with context(chaos) as ctx:
            current = {'ready': False}
            manifest = (ROOT/'data/perfect-chaos-complete/manifest.json').read_text()
            ctx.route('**/data/perfect-chaos-complete/manifest.json', lambda route: route.fulfill(status=200, content_type='application/json', body=manifest if current['ready'] else '{"format":"connect4-perfect-chaos-complete-manifest-v1","policies":[]}'))
            downloads = []
            ctx.on('request', lambda request: downloads.append(request.url) if request.url.endswith('.bin') else None)
            page = launch(ctx)
            wait_for(page, "!document.querySelector('#aiRecovery').hidden")
            assert not downloads
            assert page.locator('#downloadDialog').is_hidden()
            current['ready'] = True
            page.locator('#retryAiButton').click()
            page.locator('#downloadConfirmButton').click()
            wait_for(page, "document.querySelector('#statusText').textContent === 'Red to move'")
            assert len(downloads) == 1, downloads
            assert downloads[0].endswith('4x5-c5-role1.bin')
            page.locator('#restartButton').click()
            wait_for(page, "document.querySelector('#statusText').textContent === 'Red to move'")
            assert len(downloads) == 2 and downloads[0] == downloads[1], downloads
            passed('missing-entry gate fails closed; Retry refreshes catalog and transfers one authorised download')

        for is_chaos in (False, True):
            config = {**CONFIG, 'rows': 4, 'cols': 4, 'chaosMode': is_chaos, 'opponent': 'perfect', 'startingPlayer': 2}
            with context(config) as ctx:
                source = (ROOT/'src/data-loader.js').read_text().replace('60_000', '500').replace('65_000', '3000')
                ctx.route('**/src/data-loader.js', lambda route, _request, source=source: route.fulfill(content_type='text/javascript', body=source))
                failed = {'stall': True}
                parked = []
                def binary(route):
                    if failed['stall']: parked.append(route)
                    else: route.continue_()
                ctx.route('**/*.bin', binary)
                page = launch(ctx)
                wait_for(page, "!document.querySelector('#aiRecovery').hidden")
                assert parked
                assert 'undefined' not in page.locator('#thinkingProgress').text_content()
                assert 'NaN' not in page.locator('#searchInfo').text_content()
                # Release intercepted requests only after the timeout was observed.
                # The terminated worker cannot consume these responses.
                for route in parked:
                    try: route.abort()
                    except Exception: pass
                failed['stall'] = False
                page.locator('#retryAiButton').click()
                wait_for(page, "document.querySelector('#statusText').textContent === 'Red to move'")
                passed(f'{"Chaos" if is_chaos else "variable Classic"} policy stall recovers in a fresh worker')

        for stuck in (True, False):
            with context({**CONFIG, 'opponent': 'medium', 'startingPlayer': 2}) as ctx:
                loader = (ROOT/'src/data-loader.js').read_text().replace('65_000', '1500')
                ctx.route('**/src/data-loader.js', lambda route: route.fulfill(content_type='text/javascript', body=loader))
                worker = "self.onmessage = ({data}) => { postMessage({requestId:data.requestId, kind:'phase', phase:'loading'}); while(true) {} };" if stuck else "self.onmessage = ({data}) => { postMessage({requestId:data.requestId,kind:'phase',phase:'searching'}); postMessage({requestId:data.requestId,kind:'progress',progress:{action:{type:'drop',column:3},solver:'bitboard',solved:false,depth:1,maximumDepth:4,nodes:1,elapsedMs:1}}); setTimeout(() => postMessage({requestId:data.requestId,kind:'result',result:{action:{type:'drop',column:3},solver:'bitboard',solved:false,depth:1,nodes:1,elapsedMs:2100}}),2100); };"
                ctx.route('**/src/ai-worker.js', lambda route: route.fulfill(content_type='text/javascript', body=worker))
                page = launch(ctx)
                def layout():
                    return page.evaluate('''() => ({scrollY,
                      boxes: Object.fromEntries(['#setupPanel','#gamePanel','.game-topline','#statusText','#thinkingIndicator','#thinkingBarRow','#boardFrame'].map(s => [s, document.querySelector(s).getBoundingClientRect().toJSON()]))})''')
                if stuck:
                    wait_for(page, "document.querySelector('#aiErrorText').textContent.includes('loading stalled')")
                else:
                    # Sample after the worker entered search and published progress,
                    # not during asynchronous page startup.
                    wait_for(page, "!document.querySelector('#thinkingBarRow').hidden")
                    before = layout()
                    wait_for(page, "document.querySelector('#statusText').textContent === 'Red to move'")
                    assert page.locator('#aiErrorText').is_hidden()
                    after = layout()
                    # Check the status/progress rows' effect on the board, independent
                    # of compact-header startup above the entire game panel.
                    offset = lambda measure: measure['boxes']['#boardFrame']['y'] - measure['boxes']['#gamePanel']['y']
                    assert abs(offset(after) - offset(before)) <= 0.5, json.dumps({'before':before,'after':after})
                passed('page watchdog terminates a blocked loading worker' if stuck else 'searching phase is not curtailed by the loading watchdog')
        browser.close()
    print(f'{len(results)} extended browser regressions passed ({name}).', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--browser', choices=['chromium', 'webkit'], default='chromium'); parser.add_argument('--executable')
    args = parser.parse_args(); run(args.browser, args.executable)
