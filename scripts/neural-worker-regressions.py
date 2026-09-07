#!/usr/bin/env python3
"""Exercise the real worker boundary, responsiveness, replacement and UI.

The CPU fixture is served only by the test harness. --real-model also runs
an unmocked load/warm-up/inference using the committed ONNX and WASM files.
"""
import argparse
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from time import monotonic
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CPU_FIXTURE = """
export const DOWNLOAD_BYTES = {model: 1, runtime: 1};
export function simulationsFor() { return 128; }
export function recordSearch(network, elapsed, count) { network.perEvaluation = elapsed / count; }
export async function loadNeuralNetwork({onProgress, onBackend, onBackendFailure}) {
  if (typeof document !== 'undefined') throw new Error('Native inference started on the page');
  const mode = new URL(self.location).searchParams.get('mode');
  onProgress?.({stage:'session', phase:'warmup', backend:'wasm'});
  if (mode === 'startup-stall') { while (true) {} }
  onBackend?.('wasm');
  return {backend:'wasm', perEvaluation: 10, evaluate: async () => {
    if (mode === 'stall') { while (true) {} }
    const end = performance.now() + 20;
    while (performance.now() < end) {}
    return {policy:new Float32Array(13),value:new Float32Array(3),q:new Float32Array(39)};
  }};
}
"""


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if (urlsplit(self.path).path == '/src/neural-runtime.js'
                and self.server.runtime_fixture):
            body = CPU_FIXTURE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/javascript')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def log_message(self, *_args):
        pass


@contextmanager
def site():
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(ROOT)))
    server.runtime_fixture = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/', server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def wait_for(page, expression, timeout_ms=30_000):
    end = monotonic() + timeout_ms / 1000
    while monotonic() < end:
        if page.evaluate(expression):
            return
        page.wait_for_timeout(20)
    raise AssertionError(f'Timed out: {expression}')


def run(browser_name, executable, real_model):
    with site() as (url, server), sync_playwright() as pw:
        options = {'headless': True}
        if executable:
            options['executable_path'] = executable
        browser = getattr(pw, browser_name).launch(**options)
        context = browser.new_context(viewport={'width': 390, 'height': 844},
                                      is_mobile=True, has_touch=True, reduced_motion='reduce')
        context.add_init_script("""
          localStorage.setItem('connect4-chaos.settings.v1', JSON.stringify({
            rows:6,cols:7,connect:4,opponent:'neural',startingPlayer:2,chaosMode:false}));
          localStorage.setItem('connect4-chaos.download.neural-opponent','yes');
        """)
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        try:
            # The fixture throws if the application tries native work on the page.
            page.goto(url)
            wait_for(page, "document.querySelector('#thinkingBarRow').hidden === false")
            page.evaluate('window.heartbeats=0; window.heartbeatTimer=setInterval(()=>heartbeats++, 10)')
            page.wait_for_timeout(100)
            page.locator('#moveNowButton').tap()
            wait_for(page, "document.querySelector('#statusText').textContent === 'Red to move'")
            info = page.locator('#searchInfo').text_content()
            assert '128 simulations' not in info, info
            assert 'simulations' in info, info
            assert page.evaluate('heartbeats') > 2
            page.evaluate('clearInterval(heartbeatTimer)')
            print(f'PASS [{browser_name}] CPU worker leaves touch/Move now and page timers responsive', flush=True)

            # A reload during startup must not repeatedly relaunch a crashed
            # neural turn, including when no piece has been placed yet.
            page.locator('#restartButton').tap()
            wait_for(page, "document.querySelector('#thinkingBarRow').hidden === false")
            for _reload in range(2):
                page.reload()
                wait_for(page, "document.querySelector('#aiRecovery').hidden === false")
                assert 'Move 0' in page.locator('#moveInfo').text_content()
                assert 'board is restored' in page.locator('#aiErrorText').text_content()
                assert page.locator('#thinkingBarRow').is_hidden()
            page.locator('#retryAiButton').tap()
            wait_for(page, "document.querySelector('#thinkingBarRow').hidden === false")
            page.locator('#moveNowButton').tap()
            wait_for(page, "document.querySelector('#statusText').textContent === 'Red to move'")
            page.locator('#cell-5-3').tap()
            wait_for(page, "document.querySelector('#thinkingBarRow').hidden === false")
            saved = page.evaluate("JSON.parse(localStorage.getItem('connect4-chaos.round.v1'))")
            page.reload()
            wait_for(page, "document.querySelector('#aiRecovery').hidden === false")
            restored = page.evaluate("JSON.parse(localStorage.getItem('connect4-chaos.round.v1'))")
            assert restored['history'] == saved['history'] and restored['roundId'] == saved['roundId']
            page.locator('#retryAiButton').tap()
            wait_for(page, "document.querySelector('#thinkingBarRow').hidden === false")
            page.locator('#moveNowButton').tap()
            wait_for(page, "document.querySelector('#statusText').textContent === 'Red to move'")
            assert 'Move 3' in page.locator('#moveInfo').text_content()
            print(f'PASS [{browser_name}] opening and mid-game reloads preserve the board and wait for Retry', flush=True)

            # Control visibility notifications because headless engines differ
            # in whether an unfocused tab counts as hidden. Exercise the real
            # app handler and worker, including cancellation between evaluations.
            neural_state = 'async () => (await import("./src/neural-client.js")).neuralLoadState()'
            assert page.evaluate(neural_state) == 'ready'
            page.evaluate("""() => {
              window.testHidden=false;
              Object.defineProperty(document,'hidden',{configurable:true,get:()=>window.testHidden});
              window.setTestHidden=(hidden)=>{
                window.testHidden=hidden;
                document.dispatchEvent(new Event('visibilitychange'));
              };
            }""")
            saved = page.evaluate("localStorage.getItem('connect4-chaos.round.v1')")
            page.evaluate('setTestHidden(true)')
            assert page.evaluate(neural_state) == 'idle'
            page.evaluate('setTestHidden(false)')
            assert page.evaluate(neural_state) == 'idle', 'a human turn must not load the AI'
            assert page.evaluate("localStorage.getItem('connect4-chaos.round.v1')") == saved
            page.locator('#cell-5-2').tap()
            wait_for(page, "document.querySelector('#thinkingBarRow').hidden === false")
            saved = page.evaluate("localStorage.getItem('connect4-chaos.round.v1')")
            page.evaluate('setTestHidden(true)')
            assert page.evaluate(neural_state) == 'idle'
            assert page.evaluate("localStorage.getItem('connect4-chaos.round.v1')") == saved
            page.evaluate('setTestHidden(false)')
            wait_for(page, "document.querySelector('#thinkingBarRow').hidden === false")
            page.locator('#moveNowButton').tap()
            wait_for(page, "document.querySelector('#statusText').textContent === 'Red to move'")
            assert 'Move 5' in page.locator('#moveInfo').text_content()
            assert page.evaluate(neural_state) == 'ready'
            page.evaluate('delete document.hidden; delete window.setTestHidden; delete window.testHidden')
            if page.locator('#settingsBody').is_hidden():
                page.locator('#settingsToggle').tap()
            page.locator('#opponentInput').select_option('human')
            page.locator('#settingsForm button[type="submit"]').tap()
            assert page.evaluate(neural_state) == 'idle', 'changing opponents must release the cached neural worker'
            print(f'PASS [{browser_name}] background suspension preserves the board and releases memory; opponent changes unload idle neural workers', flush=True)

            # A real worker is killed while stuck synchronously, not just a rejected Promise.
            result = page.evaluate("""async () => {
              const {createNeuralClient} = await import('./src/neural-client.js');
              let attempts=0;
              const client=createNeuralClient({evaluationTimeoutMs:1000,
                createWorker:()=>new Worker('./src/neural-worker.js?mode='+ (++attempts===1?'stall':'ok'),{type:'module'})});
              const old=await client.load();
              let error='';
              try {await old.evaluate([],1,[],4,false,0);} catch(e) {error=e.message;}
              const afterTimeout=client.state();
              const fresh=await client.load();
              const output=await fresh.evaluate([],1,[],4,false,0);
              client.invalidate(old);
              const afterOldCleanup=client.state();
              client.invalidate(fresh);
              return {error,afterTimeout,attempts,afterOldCleanup,policy:output.policy.length};
            }""")
            assert 'timed out' in result['error'], result
            assert result['afterTimeout'] == 'idle' and result['attempts'] == 2, result
            assert result['afterOldCleanup'] == 'ready' and result['policy'] == 13, result
            print(f'PASS [{browser_name}] watchdog kills a synchronous stall; Retry starts a fresh working worker', flush=True)

            result = page.evaluate("""async () => {
              const {createNeuralClient} = await import('./src/neural-client.js');
              const client=createNeuralClient({evaluationTimeoutMs:1000,
                createWorker:()=>new Worker('./src/neural-worker.js?mode=startup-stall',{type:'module'})});
              try {await client.load();return {error:'not timed out'};}
              catch(e){return {error:e.message,state:client.state()};}
            }""")
            assert 'timed out' in result['error'] and result['state'] == 'idle', result
            print(f'PASS [{browser_name}] stalled warm-up is bounded by the page watchdog', flush=True)

            # Two pages in the same browser context share localStorage, not tab sessionStorage.
            page.evaluate("localStorage.setItem('connect4-chaos.neural.webgpu-active','1')")
            other = context.new_page()
            other.goto(url)
            check = 'async () => (await import("./src/neural-gpu-guard.js")).gpuGuard.avoided()'
            assert page.evaluate(check) is False
            assert other.evaluate(check) is False
            page.evaluate('async () => (await import("./src/neural-gpu-guard.js")).gpuGuard.failed()')
            assert page.evaluate(check) is True
            assert other.evaluate(check) is False
            other.close()
            print(f'PASS [{browser_name}] a live tab does not trigger the other tab’s GPU guard', flush=True)

            # Test the rendering function with both real live-state shapes.
            rendered = page.evaluate("""async () => {
              const {neuralSearchInfo}=await import('./src/search-info.js');
              return ['neural-loading','neural-searching'].map(solver=>neuralSearchInfo({solver}));
            }""")
            assert all(text and 'undefined' not in text and 'NaN' not in text for text in rendered)
            assert not errors, errors
        except Exception:
            output = ROOT / 'browser-results'
            output.mkdir(exist_ok=True)
            page.screenshot(path=str(output / f'neural-worker-{browser_name}.png'), full_page=True)
            (output / f'neural-worker-{browser_name}.json').write_text(json.dumps(errors))
            raise
        finally:
            context.close()

        if real_model:
            server.runtime_fixture = False
            context = browser.new_context(
                user_agent='Mozilla/5.0 (iPhone; CPU iPhone OS 26_6 like Mac OS X) AppleWebKit/605.1.15 Version/26.6 Mobile/15E148 Safari/604.1',
                is_mobile=True, has_touch=True)
            # The mobile platform policy must select CPU even when GPU is
            # advertised; run the committed model without mocking inference.
            context.add_init_script("""
              Object.defineProperty(navigator,'gpu',{value:{}});
              localStorage.setItem('connect4-chaos.settings.v1',JSON.stringify({opponent:'human'}));
            """)
            page = context.new_page()
            page.goto(url)
            result = page.evaluate("""async () => {
              const {loadNeuralNetwork,invalidateNeuralNetwork} = await import('./src/neural-client.js');
              let ticks=0;
              const timer=setInterval(()=>ticks++,10);
              try {
                const network=await loadNeuralNetwork();
                const board=Array.from({length:6},()=>Array(7).fill(0));
                const {searchPosition,bestAction}=await import('./src/neural-search.js');
                const {applyAction,otherPlayer}=await import('./src/engine.js');
                let position={board,currentPlayer:1,connect:4,chaosMode:false};
                for(let turn=0;turn<3;turn++) {
                  const search=await searchPosition(position,network.evaluate,{simulations:8});
                  const moved=applyAction(position.board,bestAction(search),position.currentPlayer);
                  if(!moved) throw new Error('Neural search returned an illegal move');
                  position={...position,board:moved.board,currentPlayer:otherPlayer(position.currentPlayer)};
                }
                const output=await network.evaluate(position.board,position.currentPlayer,[],4,false,0);
                const result={backend:network.backend,ticks,shapes:[output.policy.length,output.value.length,output.q.length],
                  finite:[...output.policy,...output.value,...output.q].every(Number.isFinite)};
                invalidateNeuralNetwork(network);
                return result;
              } finally {clearInterval(timer);}
            }""")
            assert result['backend'] == 'wasm' and result['finite'], result
            assert result['shapes'] == [13, 3, 39] and result['ticks'] > 2, result
            print(f'PASS [{browser_name}] iPhone policy, unmocked ONNX/WASM startup and repeated searches: {result}', flush=True)
            context.close()
        # Replay the shipped certificate through the real app/AI worker too.
        context = browser.new_context(reduced_motion='reduce')
        context.add_init_script("""
          localStorage.setItem('connect4-chaos.settings.v1', JSON.stringify({
            rows:4,cols:4,connect:3,opponent:'perfect',startingPlayer:1,chaosMode:true}));
        """)
        page = context.new_page()
        page.goto(url)
        for turn, column in enumerate([1, 3, 2, 1, None, None]):
            wait_for(page, "document.querySelector('#statusText').textContent === 'Red to move'")
            if column is None:
                page.locator('#flipButton').click()
            else:
                page.locator(f'.cell[data-column="{column}"]').first.click()
            wait_for(page, f"document.querySelector('#moveInfo').textContent.includes('Move {2 * turn + 2} ·')")
        wait_for(page, "document.querySelector('#statusText').textContent === 'Draw by repetition'")
        assert page.locator('#exactBadge').text_content() == 'Final'
        assert 'draw' in page.locator('#exactResultText').text_content()
        print(f'PASS [{browser_name}] shipped Perfect repetition sequence ends with an accurate draw result', flush=True)
        context.close()
        browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser', choices=['chromium', 'webkit'], default='chromium')
    parser.add_argument('--executable', default=None)
    parser.add_argument('--real-model', action='store_true')
    args = parser.parse_args()
    run(args.browser, args.executable, args.real_model)
