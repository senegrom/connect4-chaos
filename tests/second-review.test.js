import test from 'node:test';
import assert from 'node:assert/strict';
import { createNeuralClient } from '../src/neural-client.js';
import { createGpuGuard } from '../src/neural-gpu-guard.js';
import { neuralSearchInfo } from '../src/search-info.js';
import { searchPosition } from '../src/neural-search.js';
import { exactAnalysisCopy, searchSummary } from '../src/analysis-state.js';
import { createBoard, positionKey, applyAction, otherPlayer, resolveActionOutcome } from '../src/engine.js';
import { loadVerifiedPerfectChaosCompletePolicy } from '../src/perfect-chaos-complete.js';
import { choosePerfectChaosMove } from '../src/perfect-chaos-runtime.js';

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
const output = () => ({ policy: new Float32Array(13), value: new Float32Array(3), q: new Float32Array(39) });
class FakeWorker extends EventTarget {
  calls = [];
  terminated = false;
  postMessage(message) { this.calls.push(message); }
  terminate() { this.terminated = true; }
  send(data) { this.dispatchEvent(new MessageEvent('message', { data })); }
  ready() { this.send({ kind: 'result', id: this.calls[0].id, result: { backend: 'wasm', perEvaluation: 10 } }); }
}
function harness(options = {}) {
  const workers = [];
  const client = createNeuralClient({ createWorker: () => { const worker = new FakeWorker(); workers.push(worker); return worker; },
    downloadTimeoutMs: 1000, evaluationTimeoutMs: 1000,
    guard: createGpuGuard({ getStorage: () => undefined }), ...options });
  return { client, workers };
}
async function ready(client, workers) {
  const pending = client.load(); workers.at(-1).ready(); return pending;
}

test('CPU-style evaluation yields so a scheduled stop event can run', async () => {
  let stop = false;
  const timer = setTimeout(() => { stop = true; }, 5);
  const result = await searchPosition({ board: createBoard(6, 7), currentPlayer: 1, connect: 4, chaosMode: false }, () => {
    const until = performance.now() + 1;
    while (performance.now() < until) { /* CPU inference fixture */ }
    return Promise.resolve(output());
  }, { simulations: 128, shouldStop: () => stop });
  clearTimeout(timer);
  assert.equal(stop, true);
  assert.ok(result.completedSimulations >= 1 && result.completedSimulations < 128);
});

test('healthy worker is reused, and stale network handles cannot reset its replacement', async () => {
  const { client, workers } = harness();
  const old = await ready(client, workers);
  assert.equal(await client.load(), old);
  assert.equal(workers.length, 1);
  client.invalidate(old);
  assert.equal(client.state(), 'idle');
  assert.equal(workers[0].terminated, true);
  const fresh = await ready(client, workers);
  client.invalidate(old);
  assert.equal(await client.load(), fresh);
  assert.equal(workers[1].terminated, false);
  client.invalidate(fresh);
});

test('a timed-out native inference terminates its worker; Retry reaches a new inference', async () => {
  const { client, workers } = harness({ evaluationTimeoutMs: 30 });
  const old = await ready(client, workers);
  await assert.rejects(old.evaluate([], 1, [], 4, false, 0), /timed out/);
  assert.equal(client.state(), 'idle');
  assert.equal(workers[0].terminated, true);
  const fresh = await ready(client, workers);
  const retried = fresh.evaluate([], 1, [], 4, false, 0);
  const request = workers[1].calls.at(-1);
  workers[0].send({ kind: 'result', id: request.id, result: 'stale' });
  workers[1].send({ kind: 'result', id: request.id, result: output(), backend: 'wasm' });
  assert.deepEqual(await retried, output());
  assert.equal(workers.length, 2);
  client.invalidate();
});

test('aborting startup kills the worker immediately and ignores late completion', async () => {
  const { client, workers } = harness();
  const controller = new AbortController();
  const old = client.load({ signal: controller.signal });
  const rejected = assert.rejects(old, { name: 'AbortError' });
  controller.abort();
  assert.equal(client.state(), 'idle');
  assert.equal(workers[0].terminated, true);
  const fresh = client.load();
  workers[0].ready();
  assert.equal(client.state(), 'loading');
  workers[1].ready();
  await fresh; await rejected;
  assert.equal(client.state(), 'ready');
  client.invalidate();
});

test('cancelling an in-flight inference rejects all callers without marking a GPU crash', async () => {
  let failures = 0;
  const { client, workers } = harness({ guard: { avoided: () => false, failed: () => failures++ } });
  const network = await ready(client, workers);
  workers[0].send({ kind: 'backend', backend: 'webgpu' });
  const rejection = assert.rejects(network.evaluate(), { name: 'AbortError' });
  client.invalidate(network);
  await rejection;
  assert.equal(failures, 0);
});

test('worker errors invalidate the cached network rather than stranding its queue', async () => {
  const { client, workers } = harness();
  const network = await ready(client, workers);
  const failed = assert.rejects(network.evaluate(), /device failed/);
  workers[0].send({ kind: 'error', id: workers[0].calls.at(-1).id, error: 'device failed' });
  await failed;
  assert.equal(client.state(), 'idle');
  assert.equal(workers[0].terminated, true);
});

test('startup watchdog stays on the page and bounds a stalled warm-up phase', async () => {
  const { client, workers } = harness({ evaluationTimeoutMs: 20 });
  const loading = client.load();
  workers[0].send({ kind: 'progress', id: workers[0].calls[0].id,
    progress: { stage: 'session', phase: 'warmup', backend: 'wasm' } });
  await assert.rejects(loading, /startup.*timed out/);
  assert.equal(workers[0].terminated, true);
});

test('confirmed GPU failures disable GPU for the next worker, but not another tab', async () => {
  const storage = () => { const data = new Map(); return { getItem: (key) => data.get(key) ?? null, setItem: (key, value) => data.set(key, value) }; };
  let now = 100;
  const a = storage(), b = storage();
  const guardA = createGpuGuard({ getStorage: () => a, now: () => now });
  const guardB = createGpuGuard({ getStorage: () => b, now: () => now });
  // The old shared active flag, even if present, has no role in the new guard.
  a.setItem('connect4-chaos.neural.webgpu-active', '1');
  assert.equal(guardA.avoided(), false);
  assert.equal(guardB.avoided(), false);
  const { client, workers } = harness({ guard: guardA });
  const network = await ready(client, workers);
  assert.equal(workers[0].calls[0].allowWebgpu, true);
  workers[0].send({ kind: 'gpu-failure' });
  assert.equal(guardA.avoided(), true);
  assert.equal(guardB.avoided(), false);
  client.invalidate(network);
  await ready(client, workers);
  assert.equal(workers[1].calls[0].allowWebgpu, false);
  client.invalidate();
  now += 24 * 60 * 60 * 1000;
  assert.equal(guardA.avoided(), false);
});

test('GPU guard works with blocked storage and ignores future timestamps', () => {
  let now = 100;
  const guard = createGpuGuard({ getStorage: () => { throw new Error('denied'); }, now: () => now });
  assert.equal(guard.avoided(), false);
  guard.failed(); assert.equal(guard.avoided(), true);
  now = 50; assert.equal(guard.avoided(), false);
});

test('live neural details never format absent depth, nodes or elapsed time', () => {
  for (const solver of ['neural-loading', 'neural-searching']) {
    assert.doesNotMatch(neuralSearchInfo({ solver }), /NaN|undefined/);
    assert.ok(neuralSearchInfo({ solver }).length > 0);
  }
  assert.equal(neuralSearchInfo({ solver: 'neural-searching', note: 'CPU search', fraction: 0.25 }), 'CPU search · 25%');
  assert.equal(neuralSearchInfo({ solver: 'bitboard' }), null);
});

test('shipped Perfect policy reports the legal repetition sequence as a draw', async () => {
  const policy = await loadVerifiedPerfectChaosCompletePolicy(4, 4, 3, 2);
  let board = createBoard(4, 4), player = 1;
  const counts = new Map([[positionKey(board, player, 3, true), 1]]);
  const step = (action) => {
    const applied = applyAction(board, action, player);
    const outcome = resolveActionOutcome(applied.board, 3, player, action.type, action.type === 'drop' ? applied : null);
    board = applied.board; player = otherPlayer(player);
    const key = positionKey(board, player, 3, true);
    counts.set(key, (counts.get(key) ?? 0) + 1);
    return outcome.status === 'playing' && counts.get(key) >= 3 ? 'repetition' : outcome.status;
  };
  const humans = [1, 3, 2, 1, 'flip', 'flip'];
  let result;
  for (let i = 0; i < humans.length; i++) {
    const action = humans[i] === 'flip' ? { type: 'flip' } : { type: 'drop', column: humans[i] };
    assert.equal(step(action), 'playing');
    result = choosePerfectChaosMove({ board, currentPlayer: player, connect: 3, chaosMode: true,
      startingPlayer: 1, repetitionCounts: [...counts] }, { difficulty: 'perfect', perfectChaosCompletePolicy: policy });
    assert.equal(step(result.action), i === humans.length - 1 ? 'repetition' : 'playing');
  }
  assert.equal(result.value, 0);
  assert.equal(result.score, 0);
  assert.equal(result.solved, true);
  assert.equal(result.drawReason, 'repetition');
  assert.match(exactAnalysisCopy({ status: 'playing', search: result }).text, /draw/);
});

test('history-sensitive nonterminal policy values are explicitly conditional, not proved', () => {
  const board = createBoard(4, 4);
  const counts = new Map([[positionKey(board, 1, 4, true), 1], [positionKey(board, 2, 4, true), 1]]);
  const policy = { rows: 4, columns: 4, connect: 4, role: 2, lookup: () => ({ action: { type: 'flip' }, outcome: -1 }) };
  const result = choosePerfectChaosMove({ board, currentPlayer: 2, startingPlayer: 1, chaosMode: true,
    connect: 4, repetitionCounts: counts }, { difficulty: 'perfect', perfectChaosCompletePolicy: policy });
  assert.equal(result.solved, false);
  assert.equal(result.proofScope, 'board-only');
  assert.equal(exactAnalysisCopy({ status: 'playing', search: searchSummary(result) }).badge, 'Conditional');
});

test('a board win takes precedence over a purported third occurrence', () => {
  const board = createBoard(4, 4); board[3] = [2, 2, 0, 0];
  const action = { type: 'drop', column: 2 };
  const child = applyAction(board, action, 2).board;
  const policy = { rows: 4, columns: 4, connect: 3, role: 2, lookup: () => ({ action, outcome: 1 }) };
  const result = choosePerfectChaosMove({ board, currentPlayer: 2, startingPlayer: 1, chaosMode: true, connect: 3,
    repetitionCounts: [[positionKey(child, 1, 3, true), 2]] }, { difficulty: 'perfect', perfectChaosCompletePolicy: policy });
  assert.equal(result.value, 1); assert.equal(result.drawReason, null); assert.equal(result.solved, true);
});
