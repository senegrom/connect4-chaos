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
    scores: { 1: 0, 2: 0, draw: 0 }, history: [], roundId: 'recoverable-round', version: 1, unsettledResults: [],
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
  const context = { ...engine, state, elements: { boardFrame: { classList: { add() {}, remove() {} } },
    selectedColumnStatus: { textContent: '' } },
    canHumanAct: () => !state.busy && !state.aiThinking,
    renderGuidance() {}, renderStatus() {}, renderActions() {}, renderAll() {}, animationPlan: () => null,
    clearBoardAnimations() {}, pause: async () => {}, disposeAiWorker() {},
    scoreStore: { async record(id, winner, supersedes) {
      if (override) return override(id, winner, supersedes);
      if (fail) throw new Error('Injected transaction abort');
      return { ...scoreTransition(ledger, { type: 'record', id, winner, supersedes }), persistent: true };
    } },
    acceptScore(result) { state.scores = result.scores; return result.receipt; },
    scoreWarning(message) { warnings.push(message); },
    stopAiWithError(message) { state.aiThinking = false; state.aiError = message; },
    restoreSnapshot(snapshot) { restoreSnapshot(state, snapshot); },
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

// The scores moved to this tab on the replay's failed write; the note that
// said "saved successfully" replaced the tab-only warning, then went.
test('a replayed final move reports success only when the result reached storage', async () => {
  for (const persistent of [false, true]) {
    const h = moveHarness();
    await h.perform(h.action);
    h.override((id, winner) => ({ ...scoreTransition(h.ledger, { type: 'record', id, winner }), persistent }));
    await h.perform(h.action);
    assert.equal(h.state.status, 'won');
    assert.equal(h.warnings.includes('The round result was saved successfully.'), persistent);
  }
});

// A write that outlived its deadline can land after the move was put back;
// ending the round differently then counted it twice, with no Undo for the
// first result.
test('a result whose write failed is superseded by the next result of its round', async () => {
  const h = moveHarness();
  const calls = [];
  h.override(async (id, winner, supersedes) => {
    calls.push({ id, supersedes: [...supersedes] });
    if (calls.length === 1) throw new Error('Score update did not finish.');
    return { ...scoreTransition(h.ledger, { type: 'record', id, winner, supersedes }), persistent: true };
  });
  await h.perform(h.action);
  assert.deepEqual([...h.state.unsettledResults], [calls[0].id]);
  await h.perform(h.action);
  assert.deepEqual(calls.map((call) => call.supersedes), [[], [calls[0].id]]);
  assert.deepEqual([...h.state.unsettledResults], []);
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
  let abortNext = null;
  let holdNext = null;
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
        const abortCommit = abortNext; abortNext = null;
        const held = holdNext; holdNext = null;
        const tx = { abort() {
            // A held commit is already under way, as a real one is once its
            // requests have run: too late to abort, so its event comes late.
            if (held) throw new DOMException('Transaction is committing', 'InvalidStateError');
            this.aborted = true; setImmediate(() => this.onabort?.());
          },
          objectStore() { return {
            get() { const read = {}; setImmediate(() => {
              read.result = structuredClone(ledger); read.onsuccess?.();
              const finish = () => {
                if (abortCommit?.request) {
                  // As the spec orders it: the failed put's error event
                  // reaches the transaction first; the abort that sets
                  // transaction.error follows.
                  tx.onerror?.({ target: { error: abortCommit.error } });
                  tx.error = abortCommit.error; tx.abort(); return;
                }
                if (abortCommit) { tx.error = abortCommit.error; tx.abort(); return; }
                if (!tx.aborted) { if (tx.draft) ledger = tx.draft; tx.oncomplete?.(); }
              };
              if (held) held.release = finish; else setImmediate(finish);
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
    stored: () => structuredClone(ledger),
    damage(value) { ledger = value; },
    failNextOpen() { openError = new DOMException('Storage unavailable during reopen', 'UnknownError'); },
    abortNextTransaction(name) { abortNext = { error: name ? new DOMException('Write refused', name) : null }; },
    failNextPut(name) { abortNext = { error: new DOMException('Write refused', name), request: true }; },
    holdNextTransaction() { holdNext = {}; return holdNext; },
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

// The pre-ledger v1 totals were never written or removed: a player who reset
// their scores saw the old totals on every load until the ledger answered,
// and a database that failed seeded a tab-only ledger with them.
test('the v1 totals seed the ledger once and are then retired', async () => {
  const h = fakeDatabase();
  let legacy = { 1: 5, 2: 2, draw: 1 };
  let retired = 0;
  const store = createScoreStore({ indexedDB: h.indexedDB, legacyScores: () => legacy ?? {},
    retireLegacy: () => { retired += 1; legacy = null; } });
  const first = await store.read();
  assert.deepEqual([first.scores[1], first.scores[2], first.scores.draw], [5, 2, 1]);
  assert.equal(retired, 1);
  await store.reset();
  assert.equal(retired, 1, 'retired once');
  const failed = createScoreStore({ indexedDB: null, legacyScores: () => legacy ?? {}, onWarning() {} });
  assert.equal((await failed.read()).scores[1], 0, 'a tab-only ledger no longer starts from the old totals');
  // Only a committed ledger holds them; an aborted first write keeps the copy.
  const aborted = fakeDatabase();
  aborted.abortNextTransaction();
  let kept = true;
  const abortedStore = createScoreStore({ indexedDB: aborted.indexedDB, legacyScores: () => ({ 1: 5 }),
    retireLegacy: () => { kept = false; } });
  await assert.rejects(abortedStore.read());
  assert.equal(kept, true);
});

test('the page shows the scores as unknown until the ledger first answers', () => {
  const body = source.slice(source.indexOf('function renderScores()'), source.indexOf('\nfunction statusMessage('));
  const text = () => ({ textContent: '' });
  const elements = { redScore: text(), yellowScore: text(), drawScore: text(), yellowScoreLabel: text() };
  const context = vm.createContext({ elements, RED: 1, YELLOW: 2, isAiGame: () => false,
    state: { scores: { 1: 0, 2: 0, draw: 0 } }, scoreRevision: -1 });
  vm.runInContext(body, context);
  context.renderScores();
  assert.deepEqual([elements.redScore, elements.yellowScore, elements.drawScore].map((e) => e.textContent),
    ['–', '–', '–']);
  context.scoreRevision = 0;
  context.state.scores = { 1: 3, 2: 1, draw: 2 };
  context.renderScores();
  assert.deepEqual([elements.redScore, elements.yellowScore, elements.drawScore].map((e) => e.textContent),
    ['3', '1', '2']);
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

// A write that failed after the database opened used to reject every time:
// with a full quota no round could end, however often the move was replayed.
test('a full quota moves the scores to this tab at once, from the last committed ledger', async () => {
  const h = fakeDatabase(), warnings = [];
  const store = createScoreStore({ indexedDB: h.indexedDB, onWarning: (message) => warnings.push(message) });
  const first = await store.record('first', 1);
  h.abortNextTransaction('QuotaExceededError');
  const second = await store.record('second', 1);
  assert.equal(second.scores[1], 2, 'the round still ends');
  assert.equal(second.revision, first.revision + 1);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /this tab only/);
  assert.equal((await store.undo([first.receipt])).scores[1], 1, 'receipts from storage still undo');
  await store.record('third', 2);
  assert.deepEqual(Object.keys(h.stored().results), ['first'], 'storage keeps only what it committed');
  // A late close of the old handle must not reopen storage beneath the tab.
  h.connections[0].onclose?.();
  assert.equal((await store.read()).scores[2], 1);
  assert.equal(h.connections.length, 1);
});

// WebKit and Gecko fail the put itself when the quota is full; its error
// used to settle the operation as a generic failure before the abort named it.
test('a full quota reported on the request moves the scores to this tab at once', async () => {
  const h = fakeDatabase(), warnings = [];
  const store = createScoreStore({ indexedDB: h.indexedDB, onWarning: (message) => warnings.push(message) });
  await store.record('first', 1);
  h.failNextPut('QuotaExceededError');
  const second = await store.record('second', 1);
  assert.equal(second.scores[1], 2);
  assert.equal(second.persistent, false);
  assert.equal(warnings.length, 1);
  assert.deepEqual(Object.keys(h.stored().results), ['first']);
});

test('a result supersedes the failed results of its round in the same transaction', () => {
  const ledger = newLedger();
  scoreTransition(ledger, { type: 'record', id: 'round:9:late', winner: 1 });
  const next = scoreTransition(ledger, { type: 'record', id: 'round:11:ending', winner: 2,
    supersedes: ['round:9:late', 'round:9:never-landed'] });
  assert.deepEqual(Object.keys(ledger.results), ['round:11:ending']);
  assert.deepEqual([next.scores[1], next.scores[2]], [0, 1]);
  // Replaying the same ending is idempotent, whatever it supersedes.
  assert.equal(scoreTransition(ledger, { type: 'record', id: 'round:11:ending', winner: 2,
    supersedes: ['round:11:ending'] }).changed, false);
});

test('any other write failure puts the move back once, and moves the scores the second time in a row', async () => {
  const h = fakeDatabase(), warnings = [];
  const store = createScoreStore({ indexedDB: h.indexedDB, onWarning: (message) => warnings.push(message) });
  await store.record('first', 1);
  h.abortNextTransaction();
  await assert.rejects(store.record('second', 1), /Could not save score/);
  await store.record('second', 1);
  // The success in between cleared the count.
  h.abortNextTransaction('AbortError');
  await assert.rejects(store.record('third', 1), { name: 'AbortError' });
  assert.deepEqual(warnings, []);
  h.abortNextTransaction('AbortError');
  assert.equal((await store.record('third', 1)).scores[1], 3);
  assert.equal(warnings.length, 1);
  assert.deepEqual(Object.keys(h.stored().results), ['first', 'second']);
});

test('an operation the ledger refuses is not a storage failure', async () => {
  const h = fakeDatabase(), warnings = [];
  const store = createScoreStore({ indexedDB: h.indexedDB, onWarning: (message) => warnings.push(message) });
  await store.record('round', 1);
  for (let attempt = 0; attempt < 3; attempt += 1) {
    await assert.rejects(store.record('round', 2), /Another tab already recorded/);
  }
  assert.deepEqual(warnings, []);
  assert.equal((await store.record('next', 2)).scores[2], 1);
  assert.deepEqual(Object.keys(h.stored().results), ['round', 'next']);
});

// Reset validated the stored ledger first, so a damaged one could only be
// cleared by wiping the site's storage by hand.
test('Reset replaces a damaged ledger that nothing else can read', async () => {
  const h = fakeDatabase(), warnings = [];
  const store = createScoreStore({ indexedDB: h.indexedDB, legacyScores: () => ({ 1: 4 }),
    onWarning: (message) => warnings.push(message) });
  await store.record('before', 1);
  h.damage({ version: 2, epoch: 7, results: {} });
  await assert.rejects(store.read(), /damaged\. Reset the score/);
  await assert.rejects(store.record('after', 1), /damaged/);
  const reset = await store.reset();
  assert.deepEqual([reset.scores[1], reset.scores[2], reset.scores.draw], [0, 0, 0], 'not reseeded from old totals');
  assert.equal((await store.record('after', 1)).scores[1], 1);
  assert.deepEqual(Object.keys(h.stored().results), ['after']);
  assert.deepEqual(warnings, [], 'a damaged ledger is not a storage failure');
});

test('a score database that never opens leaves the scores in this tab, and closes if it opens late', async () => {
  const requests = [], warnings = [];
  const store = createScoreStore({ timeoutMs: 20, legacyScores: () => ({ 2: 1 }),
    onWarning: (message) => warnings.push(message),
    indexedDB: { open() { const request = {}; requests.push(request); return request; } } });
  assert.equal((await store.record('offline', 2)).scores[2], 2);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /this tab only/);
  let aborted = false, closed = false;
  requests[0].transaction = { abort() { aborted = true; } };
  requests[0].onupgradeneeded();
  requests[0].result = { close() { closed = true; } };
  requests[0].onsuccess();
  assert.equal(aborted, true, 'a late upgrade creates nothing');
  assert.equal(closed, true, 'a late handle is closed, not adopted');
  assert.equal((await store.read()).scores[2], 2);
  assert.equal(requests.length, 1);
});

test('a score update that outlives its deadline fails once, then the scores move to the tab', async () => {
  const h = fakeDatabase(), warnings = [];
  const store = createScoreStore({ indexedDB: h.indexedDB, timeoutMs: 20, onWarning: (message) => warnings.push(message) });
  await store.record('first', 1);
  const late = h.holdNextTransaction();
  await assert.rejects(store.record('slow', 1), /did not finish\. Your board is unchanged/);
  assert.deepEqual(warnings, []);
  // The commit lands after its caller had an answer; that answer stands, but
  // the ledger it committed is the one a tab-only fallback starts from.
  late.release();
  assert.deepEqual(Object.keys(h.stored().results), ['first', 'slow']);
  const again = h.holdNextTransaction();
  const fallback = await store.record('second', 2);
  assert.equal(warnings.length, 1);
  assert.deepEqual([fallback.scores[1], fallback.scores[2]], [2, 1]);
  again.release();
  assert.equal((await store.read()).revision, fallback.revision);
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
