import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

import { loadingWatchdog, readData } from '../src/data-loader.js';
import { PERFECT_BOOK_CERTIFICATE, loadPerfectBook } from '../src/perfect-book.js';
import { loadVerifiedPerfectChaosCompletePolicy } from '../src/perfect-chaos-complete.js';
import {
  PERFECT_CHAOS_RELEASED_POLICIES,
  PERFECT_CHAOS_ROLE_FIRST,
  loadPerfectChaosPolicy,
} from '../src/perfect-chaos-prefix.js';
import { loadVerifiedPerfectClassicPolicy } from '../src/perfect-classic-verified.js';
import { PERFECT_STRATEGY_CERTIFICATE, loadPerfectStrategy } from '../src/perfect-strategy.js';

const flush = () => new Promise((resolve) => setImmediate(resolve));

// A body the test feeds by hand, so time only moves when the test says so.
function streamedResponse() {
  let feed;
  const body = new ReadableStream({ start(controller) { feed = controller; } });
  return { response: new Response(body), feed: () => feed };
}

test('a slow download that keeps arriving finishes; one that stops fails', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const { response, feed } = streamedResponse();
  t.mock.method(globalThis, 'fetch', async () => response);
  const progress = [];
  const loading = readData('https://test.invalid/slow.bin', 'Slow table', {
    timeoutMs: 1_000, expectedBytes: 12, onDataProgress: (loaded, total) => progress.push([loaded, total]),
  });
  await flush();
  // Twelve bytes over four seconds: a fixed 1 s deadline would have failed.
  for (let chunk = 0; chunk < 4; chunk += 1) {
    t.mock.timers.tick(900);
    feed().enqueue(new Uint8Array([chunk, chunk, chunk]));
    await flush();
  }
  feed().close();
  assert.deepEqual([...await loading], [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
  // Zero-byte reports mark the start and the headers, then one per chunk.
  assert.deepEqual(progress, [[0, 12], [0, 12], [3, 12], [6, 12], [9, 12], [12, 12]]);

  const stalled = streamedResponse();
  globalThis.fetch.mock.mockImplementation(async () => stalled.response);
  const failing = readData('https://test.invalid/stalled.bin', 'Stalled table', { timeoutMs: 1_000 });
  const failure = assert.rejects(failing, /Stalled table did not finish loading: nothing arrived for 1 seconds/);
  await flush();
  stalled.feed().enqueue(new Uint8Array(4));
  await flush();
  t.mock.timers.tick(999);
  await flush();
  t.mock.timers.tick(1);
  await failure;
});

test('every re-arm of the load deadline also re-arms the page watchdog', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let stalls = 0;
  // The page allows a little more silence than the worker's load does.
  const watchdog = loadingWatchdog(() => { stalls += 1; }, { timeoutMs: 1_300 });
  const { response, feed } = streamedResponse();
  let answer;
  t.mock.method(globalThis, 'fetch', () => new Promise((resolve) => { answer = () => resolve(response); }));
  const loading = readData('https://test.invalid/slow-start.bin', 'Slow table', {
    timeoutMs: 1_000, onDataProgress: () => watchdog.touch(),
  });
  // Headers after 0.9 s, the first byte 0.9 s later: the load is never silent
  // for its full second, but the page hears nothing for 1.8 s unless the
  // headers are reported too.
  t.mock.timers.tick(900);
  answer();
  await flush();
  t.mock.timers.tick(900);
  feed().enqueue(new Uint8Array([7]));
  feed().close();
  assert.deepEqual([...await loading], [7]);
  watchdog();
  assert.equal(stalls, 0);
});

test('the page watchdog is re-armed by progress and ends at stop', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let stalls = 0;
  const stop = loadingWatchdog(() => { stalls += 1; }, { timeoutMs: 1_000 });
  for (let report = 0; report < 5; report += 1) {
    t.mock.timers.tick(900);
    stop.touch();
  }
  assert.equal(stalls, 0, 'four and a half seconds of arriving data is not a stall');
  t.mock.timers.tick(1_000);
  assert.equal(stalls, 1);
  const finished = loadingWatchdog(() => { stalls += 1; }, { timeoutMs: 1_000 });
  finished();
  finished.touch();
  t.mock.timers.tick(5_000);
  assert.equal(stalls, 1, 'a stopped watchdog cannot be re-armed');
});

test('a certified Chaos layer reports its download against the released size', async (t) => {
  const bytes = await readFile(new URL('../data/perfect-chaos-prefix/red/8-10.policy.bin', import.meta.url));
  t.mock.method(globalThis, 'fetch', async () => new Response(new ReadableStream({
    start(controller) {
      for (let offset = 0; offset < bytes.length; offset += 16_384) controller.enqueue(bytes.subarray(offset, offset + 16_384));
      controller.close();
    },
  })));
  const progress = [];
  const policy = await loadPerfectChaosPolicy(PERFECT_CHAOS_ROLE_FIRST, 8, 'https://test.invalid/red-8-10.bin',
    { onDataProgress: (loaded, total) => progress.push([loaded, total]) });
  const { bytes: released } = PERFECT_CHAOS_RELEASED_POLICIES.red['8-10.policy.bin'];
  // The start and the headers, then one report per chunk.
  assert.equal(progress.length, 2 + Math.ceil(bytes.length / 16_384));
  assert.deepEqual(progress.slice(0, 2), [[0, released], [0, released]]);
  assert.deepEqual(progress.at(-1), [released, released]);
  assert.equal(policy.boundary, 10);
});

test('every verified table a worker fetches reports its download against its known size', async (t) => {
  // Pages sends them gzipped, and then Content-Length counts compressed
  // bytes, so the page showed only how much had arrived, never of what.
  const read = (path) => readFile(new URL(`../${path}`, import.meta.url));
  const classic = JSON.parse(await read('data/perfect-classic/manifest.json'));
  const chaos = JSON.parse(await read('data/perfect-chaos-complete/manifest.json'));
  const classicEntry = classic.policies.find((entry) => entry.rows === 4 && entry.columns === 7 && entry.role === 1);
  const chaosEntry = chaos.policies.find((entry) => entry.rows === 4 && entry.columns === 4 && entry.connect === 3
    && entry.role === 1);
  const tables = [
    ['assets/perfect-strategy.bin', PERFECT_STRATEGY_CERTIFICATE.byteLength,
      (options) => loadPerfectStrategy('https://test.invalid/strategy.bin', options)],
    ['assets/perfect-book.bin', PERFECT_BOOK_CERTIFICATE.byteLength,
      (options) => loadPerfectBook('https://test.invalid/book.bin', options)],
    [`data/perfect-classic/${classicEntry.file}`, classicEntry.bytes,
      (options) => loadVerifiedPerfectClassicPolicy(4, 7, 4, 1,
        { ...options, manifest: classic, manifestUrl: 'https://test.invalid/classic/manifest.json' })],
    [`data/perfect-chaos-complete/${chaosEntry.file}`, chaosEntry.bytes,
      (options) => loadVerifiedPerfectChaosCompletePolicy(4, 4, 3, 1,
        { ...options, manifest: chaos, manifestUrl: 'https://test.invalid/chaos/manifest.json' })],
  ];
  for (const [path, size, load] of tables) {
    const bytes = await read(path);
    t.mock.method(globalThis, 'fetch', async () => new Response(bytes,
      { headers: { 'content-encoding': 'gzip', 'content-length': String(Math.ceil(size / 4)) } }));
    const progress = [];
    await load({ onDataProgress: (loaded, total) => progress.push([loaded, total]) });
    // The start and the headers, then the bytes as they arrive.
    assert.deepEqual(progress.slice(0, 2), [[0, size], [0, size]], path);
    assert.deepEqual(progress.at(-1), [size, size], path);
    t.mock.restoreAll();
  }
});

test('the page re-arms its watchdog and shows the bytes a worker reports', async () => {
  const source = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');
  const handler = source.slice(source.indexOf('function handleAiWorkerMessage('),
    source.indexOf('\nfunction handleAiWorkerError('));
  let touches = 0;
  const stopLoading = Object.assign(() => {}, { touch: () => { touches += 1; } });
  const worker = {};
  const request = { id: 7, stopLoading };
  const state = { aiWorker: worker, aiRequest: request, liveSearch: null };
  const context = vm.createContext({ state, renderAiState() {}, handleAiWorkerError() { throw new Error('protocol error'); } });
  vm.runInContext(handler, context);
  const report = (data) => context.handleAiWorkerMessage(worker, { data: { requestId: 7, kind: 'phase', ...data } });
  report({ phase: 'loading' });
  assert.equal(state.liveSearch.note, 'Loading verified AI data…');
  report({ phase: 'loading', loaded: 5_300_000, total: 21_181_376 });
  assert.equal(state.liveSearch.note, 'Loading verified AI data… 5.3 of 21.2 MB');
  report({ phase: 'loading', loaded: 1_000_000, total: 0 });
  assert.equal(state.liveSearch.note, 'Loading verified AI data… 1.0 MB');
  assert.equal(touches, 3);
});
