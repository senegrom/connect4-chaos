import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

import { loadingWatchdog, readData } from '../src/data-loader.js';
import {
  PERFECT_CHAOS_RELEASED_POLICIES,
  PERFECT_CHAOS_ROLE_FIRST,
  loadPerfectChaosPolicy,
} from '../src/perfect-chaos-prefix.js';

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
  assert.deepEqual(progress, [[3, 12], [6, 12], [9, 12], [12, 12]]);

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
  assert.equal(progress.length, Math.ceil(bytes.length / 16_384));
  assert.deepEqual(progress.at(-1), [released, released]);
  assert.equal(policy.boundary, 10);
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
