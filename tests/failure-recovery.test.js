import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import * as engine from '../src/engine.js';
import { makeSnapshot, restoreSnapshot } from '../src/round-storage.js';
import { createScoreStore, newLedger, scoreTransition } from '../src/score-store.js';
import { searchPosition, bestAction } from '../src/neural-search.js';

// Execute the production move controller with the real engine and snapshots;
// only animation, DOM rendering and the failing storage transport are replaced.
const source = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');
const performSource = source.slice(source.indexOf('async function performAction('), source.indexOf('\nfunction isLegalAiAction('));
const pause = () => new Promise((resolve) => setImmediate(resolve));
function moveHarness({ draw = false, ai = false } = {}) {
  const config = engine.normalizeConfig({ rows: draw ? 4 : 6, cols: draw ? 4 : 7,
    connect: 4, opponent: ai ? 'medium' : 'human', startingPlayer: ai ? 2 : 1, chaosMode: false });
  const state = { config, board: engine.createBoard(config.rows, config.cols), currentPlayer: config.startingPlayer,
    status: 'playing', winner: 0, winningCells: [], simultaneousWin: false, drawReason: null,
    lastMove: null, lastMover: null, moveCount: 0, selectedColumn: 0, repetitionCounts: new Map(),
    scores: { 1: 0, 2: 0, draw: 0 }, history: [], roundId: 'recoverable-round', version: 1,
    busy: false, touchHintDismissed: true, lastSearch: null, liveSearch: null, aiError: null };
  const push = (receipt = null) => state.history.push({ ...makeSnapshot(state), scoreReceipt: receipt });
  state.repetitionCounts.set(engine.positionKey(state.board, state.currentPlayer, 4, false), 1);
  push();
  const moves = draw ? [0,0,0,2,2,0,2,1,2,3,3,3,3,1,1] : [0,1,0,1,0,1];
  for (const column of moves) {
    const actor = state.currentPlayer;
    const next = engine.applyAction(state.board, { type: 'drop', column }, actor);
    state.board = next.board; state.currentPlayer = engine.otherPlayer(actor);
    state.moveCount++; state.lastMover = actor; state.lastMove = { row: next.row, column };
    state.repetitionCounts.set(engine.positionKey(state.board, state.currentPlayer, 4, false), 1);
    push();
  }
  const ledger = newLedger();
  let fail = true; let saved; let override;
  const warnings = [];
  const save = () => { saved = structuredClone(state.history); };
  const context = { ...engine, state, elements: { boardFrame: { classList: { add() {}, remove() {} } } },
    canHumanAct: () => !state.busy && !state.aiThinking,
    renderGuidance() {}, renderStatus() {}, renderActions() {}, renderAll() {}, animationPlan: () => null,
    clearBoardAnimations() {}, pause: async () => {}, disposeAiWorker() {},
    scoreStore: { async record(id, winner) {
      if (override) return override(id, winner);
      if (fail) throw new Error('Injected transaction abort');
      return scoreTransition(ledger, { type: 'record', id, winner });
    } },
    acceptScore(result) { state.scores = result.scores; return result.receipt; },
    scoreWarning(message) { warnings.push(message); },
    stopAiWithError(message) { state.aiThinking = false; state.aiError = message; },
    restoreSnapshot(snapshot, options) { restoreSnapshot(state, snapshot, options); },
    pushSnapshot: push, saveRound: save, showResultDialog() {}, isAiGame: () => ai,
    requestAiMove() { throw new Error('A failed terminal write must not auto-start another search'); },
  };
  const perform = vm.runInNewContext(`${performSource}\nperformAction;`, context);
  return { state, ledger, warnings, action: { type: 'drop', column: draw ? 1 : 0 },
    perform: (action) => perform(action, ai ? 'ai' : 'human'),
    recover: () => { fail = false; }, saved: () => saved,
    override: (record) => { override = record; } };
}

for (const mode of ['human win', 'AI win', 'draw']) test(`${mode}: failed result storage restores the playable snapshot and permits one retry`, async () => {
  const h = moveHarness({ draw: mode === 'draw', ai: mode === 'AI win' });
  const before = structuredClone(h.state.history.at(-1));
  await h.perform(h.action);
  assert.equal(h.state.status, 'playing');
  assert.equal(h.state.busy, false);
  assert.deepEqual(h.state.board, before.board);
  assert.equal(h.state.moveCount, before.moveCount);
  assert.deepEqual(h.saved().at(-1).board, before.board, 'reload retains the retryable position');
  assert.equal(Object.keys(h.ledger.results).length, 0);
  if (mode === 'AI win') assert.match(h.state.aiError, /Retry/);
  else assert.match(h.warnings.at(-1), /again|retry/i);
  h.recover();
  await h.perform(h.action);
  assert.equal(h.state.status, mode === 'draw' ? 'draw' : 'won');
  assert.equal(h.state.moveCount, before.moveCount + 1);
  assert.equal(Object.keys(h.ledger.results).length, 1);
  assert.ok(h.state.history.at(-1).scoreReceipt);
  assert.equal(h.state.scores[mode === 'draw' ? 'draw' : mode === 'AI win' ? 2 : 1], 1);
});

test('late rejected result storage never rolls back a restarted round', async () => {
  const h = moveHarness();
  let reject;
  h.override(() => new Promise((_resolve, no) => { reject = no; }));
  const moving = h.perform(h.action);
  await pause();
  assert.ok(reject);
  h.state.version++;
  h.state.board = engine.createBoard(4, 4); h.state.moveCount = 0;
  h.state.status = 'playing'; h.state.busy = false;
  reject(new Error('old write failed'));
  await moving;
  assert.deepEqual(h.state.board, engine.createBoard(4, 4));
  assert.equal(h.state.moveCount, 0);
  assert.equal(h.saved(), undefined);
});

// Minimal asynchronous IndexedDB transport. Real-browser tests also close an
// actual database handle, to cover transaction/open event ordering in browsers.
function fakeDatabase() {
  let ledger;
  let openError;
  let abortNext = false;
  const connections = [];
  const indexedDB = { open() {
    const request = {};
    if (openError) {
      request.error = openError; openError = null;
      setImmediate(() => request.onerror?.());
      return request;
    }
    const db = { closed: false, close() { this.closed = true; },
      transaction() {
        if (this.closed) throw new DOMException('Closed connection', 'InvalidStateError');
        const abortCommit = abortNext; abortNext = false;
        const tx = { abort() { this.aborted = true; setImmediate(() => this.onabort?.()); },
          objectStore() { return {
            get() { const read = {}; setImmediate(() => {
              read.result = structuredClone(ledger); read.onsuccess?.();
              setImmediate(() => {
                if (abortCommit) { tx.abort(); return; }
                if (!tx.aborted) { if (tx.draft) ledger = tx.draft; tx.oncomplete?.(); }
              });
            }); return read; },
            put(value) { tx.draft = structuredClone(value); },
          }; },
        }; return tx;
      },
    };
    connections.push(db);
    setImmediate(() => { request.result = db; request.onsuccess?.(); });
    return request;
  } };
  return { indexedDB, connections,
    failNextOpen() { openError = new DOMException('Storage unavailable during reopen', 'UnknownError'); },
    abortNextTransaction() { abortNext = true; },
  };
}

test('an unexpected score database closure reopens storage without losing saved wins', async () => {
  const h = fakeDatabase(); const warnings = [];
  const store = createScoreStore({ indexedDB: h.indexedDB, onWarning: (message) => warnings.push(message) });
  await store.record('first', 1);
  h.connections[0].close(); h.connections[0].onclose?.();
  const result = await store.record('second', 1);
  assert.equal(result.scores[1], 2);
  assert.equal(h.connections.length, 2);
  // A late event from the old connection must not invalidate the replacement.
  h.connections[0].onversionchange?.(); h.connections[0].onclose?.();
  assert.equal((await store.read()).scores[1], 2);
  assert.equal(h.connections.length, 2);
  assert.deepEqual(warnings, []);
});

test('a closed handle without a close event retries transaction creation once', async () => {
  const h = fakeDatabase(); const store = createScoreStore({ indexedDB: h.indexedDB });
  await store.record('before-close', 2);
  h.connections[0].close();
  assert.equal((await store.record('after-close', 2)).scores[2], 2);
  assert.equal(h.connections.length, 2);
});

test('a failed database reopen preserves totals, revisions and Undo receipts in memory', async () => {
  const h = fakeDatabase(), warnings = [];
  const store = createScoreStore({ indexedDB: h.indexedDB, onWarning: (message) => warnings.push(message) });
  const display = { state: { scores: {} }, renderScores() {}, saveJson() {}, SCORE_CHANGE_KEY: 'changed', resultId: () => 'notice' };
  vm.createContext(display);
  vm.runInContext(source.slice(source.indexOf('let scoreRevision ='), source.indexOf('async function refreshScores()')), display);
  const first = await store.record('first', 1);
  display.acceptScore(first);
  display.acceptScore(await store.record('second', 1));
  display.acceptScore(await store.record('third', 1));
  h.connections[0].close(); h.connections[0].onclose(); h.failNextOpen();
  const fourth = await store.record('fourth', 1);
  display.acceptScore(fourth);
  assert.equal(display.state.scores[1], 4);
  assert.equal(fourth.revision, 4);
  assert.equal(fourth.receipt.epoch, first.receipt.epoch);
  assert.equal((await store.record('first', 1)).changed, false, 'fallback retains result IDs');
  display.acceptScore(await store.undo([first.receipt]));
  assert.equal(display.state.scores[1], 3, 'existing receipts still reverse exactly one result');
  assert.equal((await store.undo([first.receipt])).changed, false);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /this tab only/);
});

test('fallback uses the last committed ledger, including reads, but excludes aborted writes', async () => {
  const h = fakeDatabase();
  const writer = createScoreStore({ indexedDB: h.indexedDB });
  const recorded = await writer.record('saved', 2);
  const reader = createScoreStore({ indexedDB: h.indexedDB });
  assert.equal((await reader.read()).scores[2], 1);
  h.abortNextTransaction();
  await assert.rejects(reader.record('aborted', 2), /Could not save score/);
  h.connections[1].close(); h.connections[1].onclose(); h.failNextOpen();
  const fallback = await reader.read();
  assert.equal(fallback.scores[2], 1);
  assert.equal(fallback.revision, recorded.revision);
  assert.equal((await reader.undo([recorded.receipt])).scores[2], 0);
});

const position = { board: engine.createBoard(6, 7), currentPlayer: 1, connect: 4, chaosMode: false };
const logits = () => ({ policy: new Float32Array(13), value: new Float32Array(3), q: new Float32Array(39) });
for (const head of ['policy', 'value', 'q']) {
  for (const bad of [NaN, Infinity, 'all-masked', 'wrong-size']) {
    test(`neural search rejects ${head} ${String(bad)} rather than inventing a move`, async () => {
      const output = logits();
      if (bad === 'wrong-size') output[head] = new Float32Array(1);
      else if (bad === 'all-masked') output[head].fill(-Infinity);
      else output[head][0] = bad;
      await assert.rejects(searchPosition(position, async () => output, { simulations: 1 }), /neural.*(logits|output)/i);
    });
  }
}

test('neural validation permits masked illegal moves and exact one-hot outcomes', async () => {
  const output = logits();
  output.policy.fill(-Infinity); output.policy[3] = 0;
  output.value.set([-Infinity, 0, -Infinity]);
  for (let i = 0; i < 13; i++) output.q.set([-Infinity, 0, -Infinity], i * 3);
  const result = await searchPosition(position, async (_board, _mover, actions) => {
    output.policy.fill(-Infinity);
    for (const action of actions) output.policy[action.column] = action.column === 3 ? 8 : 0;
    return output;
  }, { simulations: 8 });
  assert.equal(result.value, 0);
  assert.equal(bestAction(result).column, 3);
});

test('neural validation also rejects corrupt child evaluations', async () => {
  let calls = 0;
  await assert.rejects(searchPosition(position, async () => {
    const output = logits(); if (++calls > 1) output.value[0] = NaN; return output;
  }, { simulations: 1 }), /neural.*(logits|output)/i);
  assert.equal(calls, 2);
});
