import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

// The isolation worker is a classic script (Safari registers no module
// service workers), so it runs here in a context shaped like its global.
const source = await readFile(new URL('../cross-origin-isolation-worker.js', import.meta.url), 'utf8');
const SITE = 'https://senegrom.github.io/connect4-chaos/';
const MODEL = 'https://connect4-model.connect4-chaos.workers.dev/model-abc.onnx';

function installWorker(upstream = () => new Response('body', { status: 200, headers: { 'Content-Type': 'text/plain' } })) {
  const listeners = new Map();
  const fetched = [];
  const context = vm.createContext({
    self: {
      location: new URL('cross-origin-isolation-worker.js', SITE),
      addEventListener: (type, listener) => listeners.set(type, listener),
      skipWaiting() {},
      clients: { async claim() {} },
    },
    fetch: async (request) => { fetched.push(request.url); return upstream(request); },
    Headers, Response, URL,
  });
  vm.runInContext(source, context);
  const dispatch = (url, { mode = 'cors', cache = 'default' } = {}) => {
    let answer = null;
    listeners.get('fetch')({ request: { url, mode, cache }, respondWith(response) { answer = response; } });
    return answer;
  };
  return { dispatch, fetched };
}

test('the isolation worker stamps this origin\'s responses with the isolation headers', async () => {
  const { dispatch, fetched } = installWorker();
  const response = await dispatch(new URL('src/app.js', SITE).href);
  assert.deepEqual(fetched, [new URL('src/app.js', SITE).href]);
  assert.equal(response.status, 200);
  assert.equal(await response.text(), 'body');
  assert.equal(response.headers.get('content-type'), 'text/plain', 'the response is otherwise untouched');
  assert.equal(response.headers.get('cross-origin-opener-policy'), 'same-origin');
  assert.equal(response.headers.get('cross-origin-embedder-policy'), 'require-corp');
  assert.equal(response.headers.get('cross-origin-resource-policy'), 'same-origin');
});

test('a response from another origin passes through the isolation worker untouched', () => {
  // Cross-Origin-Resource-Policy: same-origin on the model's response is the
  // instruction to block it, on every visit, isolated or not.
  const { dispatch, fetched } = installWorker();
  assert.equal(dispatch(MODEL), null, 'the browser fetches it as if no worker were there');
  assert.equal(dispatch('https://senegrom.github.io.evil.example/connect4-chaos/src/app.js'), null);
  assert.deepEqual(fetched, []);
});

test('cache-only range requests and opaque responses pass through as they are', async () => {
  const opaque = { status: 0, type: 'opaque', headers: new Headers() };
  const { dispatch } = installWorker(() => opaque);
  assert.equal(dispatch(new URL('assets/neural/model.json', SITE).href, { cache: 'only-if-cached', mode: 'no-cors' }), null);
  assert.equal(await dispatch(new URL('assets/neural/model.json', SITE).href), opaque);
});

test('an error status keeps its status when it is stamped', async () => {
  const { dispatch } = installWorker(() => new Response('missing', { status: 404, statusText: 'Not Found' }));
  const response = await dispatch(new URL('data/missing.json', SITE).href);
  assert.equal(response.status, 404);
  assert.equal(response.statusText, 'Not Found');
  assert.equal(response.headers.get('cross-origin-embedder-policy'), 'require-corp');
});
