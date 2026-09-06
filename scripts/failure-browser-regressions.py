#!/usr/bin/env python3
"""Real-browser result-write recovery, closed IndexedDB handles and invalid inference.

Transport/inference faults are injected only by this test. Result-write tests
abort actual IndexedDB transactions before commit, rather than mocking scores.
"""
import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('helpers', ROOT / 'scripts/browser-regressions.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
wait_for = helpers.wait_for
CONFIG = helpers.CONFIG

ABORT_RESULTS = """(() => {
  window.abortResults = false;
  window.abortedResults = 0;
  const put = IDBObjectStore.prototype.put;
  IDBObjectStore.prototype.put = function(value, ...args) {
    const request = put.call(this, value, ...args);
    if (this.name === 'scores' && value?.results && Object.keys(value.results).length && window.abortResults) {
      const transaction = this.transaction;
      request.addEventListener('success', () => {
        window.abortedResults++;
        transaction.abort();
      }, {once: true});
    }
    return request;
  };
})();"""

NEURAL = """
let phase = 'idle';
export const DOWNLOAD_BYTES = {model: 1, runtime: 1};
export const neuralLoadState = () => phase;
export const simulationsFor = () => 8;
export const recordSearch = () => {};
export const cancelNeuralLoad = () => { phase = 'idle'; };
export const invalidateNeuralNetwork = () => { if (phase !== 'idle') { phase = 'idle'; window.invalidations = (window.invalidations || 0) + 1; } };
export async function loadNeuralNetwork() {
  phase = 'ready'; window.networkLoads = (window.networkLoads || 0) + 1;
  return {backend:'wasm', evaluate:async () => ({
    policy:new Float32Array(13),
    value:new Float32Array(3).fill(window.validNetwork ? 0 : NaN),
    q:new Float32Array(39)
  })};
}
"""


def run(name, executable=None):
    passed = []
    with helpers.site() as url, sync_playwright() as pw:
        browser = getattr(pw, name).launch(headless=True, **({'executable_path': executable} if executable else {}))

        @contextmanager
        def page_for(config=CONFIG, neural=False, ai=False):
            ctx = browser.new_context(reduced_motion='reduce')
            origin = json.dumps(url.rstrip('/'))
            ctx.add_init_script(f"if (location.origin === {origin}) {{ if (!localStorage.getItem('connect4-chaos.settings.v1')) localStorage.setItem('connect4-chaos.settings.v1', JSON.stringify({json.dumps(config)})); localStorage.setItem('connect4-chaos.download.neural-opponent', 'yes'); }}")
            ctx.add_init_script(ABORT_RESULTS)
            if neural:
                ctx.route('**/src/neural-client.js', lambda route: route.fulfill(content_type='text/javascript', body=NEURAL))
            if ai:
                worker = "self.onmessage=({data})=>postMessage({requestId:data.requestId,kind:'result',result:{action:{type:'drop',column:0},solver:'test',solved:false}});"
                ctx.route('**/src/ai-worker.js', lambda route: route.fulfill(content_type='text/javascript', body=worker))
            page = ctx.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            try:
                page.goto(url)
                page.wait_for_selector('.cell')
                yield page
                assert not errors, errors
            except Exception:
                out = ROOT / 'browser-results'; out.mkdir(exist_ok=True)
                try:
                    page.screenshot(path=str(out / f'failure-recovery-{name}-{len(passed)}.png'), full_page=True)
                    (out / f'failure-recovery-{name}-{len(passed)}.json').write_text(json.dumps(errors))
                except Exception:
                    pass
                raise
            finally:
                ctx.close()

        def done(label):
            passed.append(label)
            print(f'PASS [{name}] {label}', flush=True)

        def idle(page, count):
            wait_for(page, f"document.querySelector('#moveInfo').textContent.includes('Move {count}') && document.querySelector('#gameBoard').getAttribute('aria-busy') === 'false'")

        for draw in (False, True):
            config = {**CONFIG, 'rows': 4, 'cols': 4} if draw else CONFIG
            moves = [0,0,0,2,2,0,2,1,2,3,3,3,3,1,1] if draw else [0,1,0,1,0,1]
            final = 1 if draw else 0
            with page_for(config) as page:
                for count, column in enumerate(moves, 1):
                    page.locator(f'.cell[data-column="{column}"]').first.click(); idle(page, count)
                page.evaluate('abortResults = true')
                page.locator(f'.cell[data-column="{final}"]').first.click()
                wait_for(page, 'abortedResults === 1')
                idle(page, len(moves))
                assert not page.locator('#resultDialog').is_visible()
                assert 'again' in page.locator('#scoreStorageStatus').text_content()
                saved = page.evaluate("JSON.parse(localStorage.getItem('connect4-chaos.round.v1'))")
                assert saved['history'][-1]['status'] == 'playing'
                assert saved['history'][-1]['moveCount'] == len(moves)
                page.reload(); idle(page, len(moves))
                page.locator(f'.cell[data-column="{final}"]').first.click()
                idle(page, len(moves) + 1)
                score_id = '#drawScore' if draw else '#redScore'
                wait_for(page, f"document.querySelector('{score_id}').textContent === '1'")
                page.locator('#reviewBoardButton').click()
                page.locator('#undoButton').click(); idle(page, len(moves))
                assert page.locator(score_id).text_content() == '0'
                done(f'{"draw" if draw else "win"}: aborted write preserves playable round across reload; retry counts once and Undo works')

        with page_for({**CONFIG, 'opponent':'medium', 'startingPlayer':2}, ai=True) as page:
            idle(page, 1)
            for count in (3, 5):
                page.locator('.cell[data-column="1"]').first.click(); idle(page, count)
            page.evaluate('abortResults = true')
            page.locator('.cell[data-column="1"]').first.click()
            wait_for(page, 'abortedResults === 1')
            idle(page, 6)
            assert not page.locator('#retryAiButton').is_disabled()
            assert 'final move' in page.locator('#aiErrorText').text_content()
            assert page.locator('#yellowScore').text_content() == '0'
            page.evaluate('abortResults = false')
            page.locator('#retryAiButton').click(); idle(page, 7)
            assert page.locator('#yellowScore').text_content() == '1'
            done('AI result-write abort exposes Retry and records exactly one recovered win')

        with page_for() as page:
            result = page.evaluate("""async () => {
              const {createScoreStore} = await import('./src/score-store.js');
              const handles = [];
              const store = createScoreStore({indexedDB: {open(...args) {
                const request = indexedDB.open(...args);
                request.addEventListener('success', () => handles.push(request.result));
                return request;
              }}});
              await store.record('closed-a', 1);
              handles[0].close(); handles[0].dispatchEvent(new Event('close'));
              const afterEvent = await store.record('closed-b', 1);
              handles[1].close(); // Deliberately no event: transaction creation must recover.
              const afterSilent = await store.record('closed-c', 1);
              handles[0].dispatchEvent(new Event('versionchange'));
              handles[0].dispatchEvent(new Event('close'));
              const afterOldEvent = await store.read();
              handles.forEach(db => db.close());
              return {afterEvent:afterEvent.scores[1], afterSilent:afterSilent.scores[1], afterOldEvent:afterOldEvent.scores[1], handles:handles.length};
            }""")
            assert result == {'afterEvent':2, 'afterSilent':3, 'afterOldEvent':3, 'handles':3}, result
            done('actual closed IndexedDB handles reopen without losing results or reusing retired connections')

        with page_for({**CONFIG, 'opponent':'neural', 'startingPlayer':2}, neural=True) as page:
            wait_for(page, "!document.querySelector('#aiRecovery').hidden")
            idle(page, 0)
            assert 'invalid' in page.locator('#aiErrorText').text_content()
            assert page.evaluate('invalidations') == 1
            page.evaluate('validNetwork = true')
            page.locator('#retryAiButton').click(); idle(page, 1)
            assert page.evaluate('networkLoads') == 2
            assert page.locator('#aiRecovery').is_hidden()
            done('invalid neural output plays no move; Retry loads a fresh evaluator and resumes')
        browser.close()
    print(f'{len(passed)} failure-recovery browser scenarios passed ({name}).', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser', choices=['chromium', 'webkit'], default='chromium')
    parser.add_argument('--executable')
    args = parser.parse_args()
    run(args.browser, args.executable)
