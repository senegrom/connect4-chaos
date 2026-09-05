import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { createBoard, RED, YELLOW, positionKey, applyAction, ACTION_DROP, normalizeConfig } from '../src/engine.js';
import { bestAction, searchPosition } from '../src/neural-search.js';
import { exactAnalysisCopy, searchIsExact, searchSummary } from '../src/analysis-state.js';
import { createResourceLoader, waitFor } from '../src/async-control.js';
import { startBackend, recordSearch, simulationsFor } from '../src/neural-runtime.js';
import { perfectCapability } from '../src/settings-controller.js';
import { makeSnapshot, restoreSnapshot, validSnapshot } from '../src/round-storage.js';

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done; }); return { resolve, promise }; };
const position = (currentPlayer = RED) => ({ board: createBoard(4, 4), currentPlayer, connect: 4, chaosMode: true });
const key = (p, player = p.currentPlayer) => positionKey(p.board, player, p.connect, p.chaosMode);
function network(action = 10, seen = []) {
  return async (_board, _mover, _actions, _connect, _chaos, repeated) => {
    seen.push(repeated);
    const policy = new Float32Array(13).fill(-100);
    policy[action] = 100;
    const q = new Float32Array(39);
    for (let i = 0; i < 13; i += 1) q[i * 3] = 1000;
    q[action * 3] = 0;
    q[action * 3 + 2] = 1000;
    return { policy, value: new Float32Array([1000, 0, 0]), q };
  };
}

test('Chaos search recognises an immediate historical threefold draw', async () => {
  const p = position(YELLOW);
  p.repetitionCounts = [[key(p, RED), 2], [key(p, YELLOW), 2]];
  const original = structuredClone(p);
  const seen = [];
  const result = await searchPosition(p, network(10, seen), { simulations: 1 });
  assert.equal(bestAction(result).type, 'flip');
  assert.equal(result.value, 0);
  assert.equal(result.evaluations, 1, 'terminal child is not sent to the network');
  assert.deepEqual(seen, [1]);
  assert.deepEqual(p, original, 'the real game history must not be mutated');
});

test('Repetition features include the complete simulated path through reused children', async () => {
  const p = position();
  const seen = [];
  const result = await searchPosition(p, network(10, seen), { simulations: 12 });
  assert.equal(result.completedSimulations, 12);
  assert.deepEqual(seen, [0, 0, 1, 1], 'root, other player, root twice, other player twice; then draw');
  assert.equal(result.evaluations, 4);
});

test('A previously visited non-root position is terminal on its third visit', async () => {
  const p = position();
  const child = applyAction(p.board, { type: ACTION_DROP, column: 0 }, RED).board;
  p.repetitionCounts = [[key(p), 1], [positionKey(child, YELLOW, 4, true), 2]];
  const result = await searchPosition(p, network(0), { simulations: 1 });
  assert.equal(result.value, 0);
  assert.equal(result.evaluations, 1);
});

test('An already drawn root does not spend an evaluation or return an action', async () => {
  const p = position();
  p.repetitionCounts = [[key(p), 3]];
  const result = await searchPosition(p, () => { throw new Error('must not evaluate'); });
  assert.equal(bestAction(result), null);
  assert.equal(result.completedSimulations, 0);
});

test('Move now reports completed simulations and actual evaluations, not its budget', async () => {
  const result = await searchPosition(position(), network(), { simulations: 75, shouldStop: () => true });
  assert.equal(result.completedSimulations, 1);
  assert.equal(result.visits.reduce((a, b) => a + b, 0), 1);
  assert.equal(result.evaluations, 2);
  assert.ok(Number.isFinite(result.value), 'large finite logits must not overflow');
  const calibration = { perEvaluation: 20, backend: 'wasm' };
  recordSearch(calibration, 150, result.evaluations);
  assert.equal(calibration.perEvaluation, 47.5);
  assert.equal(simulationsFor(calibration), 32);
});

test('Aborted searches do not run the evaluator', async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(searchPosition(position(), () => { throw Error('must not evaluate'); }, {
    signal: controller.signal,
  }), { name: 'AbortError' });
});

test('Cancellation during evaluation stops the search before another simulation', async () => {
  const controller = new AbortController();
  const gate = deferred();
  let calls = 0;
  const result = searchPosition(position(), async () => {
    calls += 1; await gate.promise; return network()();
  }, { signal: controller.signal });
  controller.abort(); gate.resolve();
  await assert.rejects(result, { name: 'AbortError' });
  assert.equal(calls, 1);
});

test('Exact solver progress with a placeholder zero is not a proved draw', () => {
  for (const solver of ['bitboard-exact', 'classic-exact', 'perfect-book', 'perfect-strategy']) {
    const search = { solver, solved: false, score: 0 };
    assert.equal(searchIsExact(search), false);
    const copy = exactAnalysisCopy({ status: 'playing', search, thinking: true });
    assert.equal(copy.badge, 'Searching');
    assert.doesNotMatch(copy.text, /draw|force a win/);
  }
});

test('Completed finite proofs alone announce exact outcomes', () => {
  for (const score of [-1, 0, 1]) {
    const search = { solved: true, score };
    assert.equal(searchIsExact(search), true);
    assert.equal(exactAnalysisCopy({ status: 'playing', search }).badge, 'Proved');
  }
  assert.equal(searchIsExact(searchSummary({ solved: true })), false);
  assert.equal(searchIsExact({ solved: true, score: NaN }), false);
  assert.equal(exactAnalysisCopy({ status: 'draw' }).badge, 'Final');
});

test('Resource cancellation is immediate and disposes a late result', async () => {
  const gate = deferred();
  let released = 0;
  const loader = createResourceLoader(() => gate.promise);
  const pending = loader.load();
  await tick();
  loader.cancel();
  assert.equal(loader.state(), 'idle');
  await assert.rejects(pending, { name: 'AbortError' });
  gate.resolve({ dispose() { released += 1; } });
  await tick();
  assert.equal(released, 1);
  assert.equal(loader.state(), 'idle');
});

test('A cancelled startup cannot clear or publish over a newer startup', async () => {
  const first = deferred(); const second = deferred();
  let calls = 0; let released = 0;
  const loader = createResourceLoader(() => ++calls === 1 ? first.promise : second.promise);
  const old = loader.load(); await tick(); loader.cancel();
  const fresh = loader.load();
  await assert.rejects(old, { name: 'AbortError' });
  first.resolve({ dispose() { released += 1; } }); await tick();
  assert.equal(loader.state(), 'loading');
  const resource = {};
  second.resolve(resource);
  assert.equal(await fresh, resource);
  assert.equal(await loader.load(), resource);
  assert.equal(loader.state(), 'ready');
  assert.equal(released, 1);
});

test('Bounded waits reject stalls and release eventual resources', async () => {
  const gate = deferred(); let released = 0;
  await assert.rejects(waitFor(gate.promise, { timeoutMs: 5, onLate: () => released++ }), /did not finish/);
  gate.resolve({}); await tick();
  assert.equal(released, 1);
});

function fakeOrt(create) {
  return { Tensor: class {}, InferenceSession: { create } };
}
const outputs = () => ({ policy: { data: new Float32Array(13) }, value: { data: new Float32Array(3) }, q: { data: new Float32Array(39) } });

test('Cancelling session creation releases a session that arrives after cancellation', async () => {
  const gate = deferred(); const controller = new AbortController(); let released = 0;
  const pending = startBackend(fakeOrt(() => gate.promise), new ArrayBuffer(0), 'wasm', { signal: controller.signal });
  controller.abort();
  await assert.rejects(pending, { name: 'AbortError' });
  gate.resolve({ release() { released += 1; } }); await tick();
  assert.equal(released, 1);
});

test('Cancelling warm-up rejects promptly and releases after native work drains', async () => {
  const gate = deferred(); const controller = new AbortController(); let released = 0; let runs = 0;
  const session = { run() { runs += 1; return gate.promise; }, release() { released += 1; } };
  const pending = startBackend(fakeOrt(async () => session), new ArrayBuffer(0), 'wasm', { signal: controller.signal });
  await tick(); controller.abort();
  await assert.rejects(pending, { name: 'AbortError' });
  assert.equal(runs, 1);
  assert.equal(released, 0);
  gate.resolve(outputs()); await tick();
  assert.equal(released, 1);
  assert.equal(runs, 1, 'warm-up must not keep evaluating after cancellation');
});

test('Warm-up has its own deadline rather than relying on session creation', async () => {
  const gate = deferred(); let released = 0;
  const session = { run: () => gate.promise, release() { released += 1; } };
  await assert.rejects(startBackend(fakeOrt(async () => session), new ArrayBuffer(0), 'wasm', { timeoutMs: 5 }), /warm-up/);
  gate.resolve(outputs()); await tick();
  assert.equal(released, 1);
});

test('Perfect capability distinguishes missing, loading and failed catalogs', () => {
  const standard = normalizeConfig({ opponent: 'perfect' });
  assert.equal(perfectCapability(standard).available, true);
  const small = { ...standard, rows: 4, cols: 4 };
  assert.equal(perfectCapability(small).status, 'loading');
  assert.equal(perfectCapability(small, { classic: { status: 'error' } }).status, 'error');
  assert.equal(perfectCapability(small, { classic: { status: 'ready', manifest: { policies: [] } } }).status, 'unavailable');
  assert.equal(perfectCapability({ ...small, rows: 10 }).available, false);
});

test('Catalog availability respects starting role and certified starting orientation', async () => {
  const manifest = JSON.parse(await readFile(new URL('../data/perfect-chaos-complete/manifest.json', import.meta.url)));
  const rules = normalizeConfig({ rows: 4, cols: 5, connect: 4, chaosMode: true, startingPlayer: RED });
  const catalogs = { chaos: { status: 'ready', manifest } };
  assert.equal(perfectCapability(rules, catalogs).available, true);
  assert.equal(perfectCapability({ ...rules, rows: 5, cols: 4 }, catalogs).available, false);
  const role = 2;
  const filtered = { ...manifest, policies: manifest.policies.filter((entry) => entry.role !== role) };
  assert.equal(perfectCapability(rules, { chaos: { status: 'ready', manifest: filtered } }).available, false);
});

test('Restored snapshots reject malformed history and discard stale analysis', () => {
  const config = normalizeConfig({ opponent: 'human' });
  const state = { config, board: createBoard(6, 7), currentPlayer: RED, status: 'playing', winner: 0,
    winningCells: [], simultaneousWin: false, drawReason: null, lastMove: null, lastMover: null,
    moveCount: 0, selectedColumn: 3, repetitionCounts: new Map(), scores: { 1: 0, 2: 0, draw: 0 },
    lastSearch: { solved: true, score: 1 } };
  const snapshot = makeSnapshot(state);
  assert.equal(validSnapshot(snapshot, config), true);
  assert.equal(validSnapshot({ ...snapshot, repetitionCounts: [null] }, config), false);
  assert.equal(validSnapshot({ ...snapshot, winningCells: [null] }, config), false);
  assert.equal(validSnapshot({ ...snapshot, scores: { 1: 'not a score', 2: 0, draw: 0 } }, config), false);
  restoreSnapshot(state, snapshot);
  assert.equal(state.lastSearch, null);
});
