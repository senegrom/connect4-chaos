import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

// Execute the shipped handler, not a second implementation of its ordering.
const source = readFileSync(new URL('../src/app.js', import.meta.url), 'utf8');
const handler = source.slice(source.indexOf('async function undoTurn()'), source.indexOf('async function resetScores()'));
function fixture(undo) {
  const receipt = { id: 'round-1', epoch: 'epoch-1' };
  const history = [{ board: 'before', status: 'playing' }, { board: 'won', status: 'won', scoreReceipt: receipt }];
  const state = { busy: false, version: 0, history, board: 'won', status: 'won', aiThinking: false };
  const saved = [];
  const warnings = [];
  const env = { state, scoreStore: { undo }, findUndoIndex: () => state.history.length - 2,
    cancelAiSearch() {}, closeResultDialog() {}, clearBoardAnimations() {}, renderAll() {},
    restoreSnapshot(snapshot) { state.board = snapshot.board; state.status = snapshot.status; },
    acceptScore(result) { state.scores = result.scores; },
    scoreWarning(message) { warnings.push(message); },
    saveRound() { saved.push(structuredClone(state.history)); },
  };
  vm.createContext(env);
  vm.runInContext(handler, env);
  return { ...env, saved, warnings, originalHistory: history, receipt };
}

test('failed Undo preserves board/history/receipt; retry reverses the same result', async () => {
  let fail = true;
  const calls = [];
  const f = fixture(async (receipts) => {
    calls.push(receipts);
    if (fail) throw new Error('Injected transaction abort');
    return { scores: { 1: 0, 2: 0, draw: 0 } };
  });
  await f.undoTurn();
  assert.equal(f.state.board, 'won');
  assert.equal(f.state.history, f.originalHistory);
  assert.equal(f.state.history[1].scoreReceipt, f.receipt);
  assert.equal(f.state.busy, false);
  assert.equal(f.saved.length, 1);
  assert.equal(f.saved[0][1].scoreReceipt.id, f.receipt.id);
  assert.deepEqual(f.warnings, ['Injected transaction abort']);
  fail = false;
  await f.undoTurn();
  assert.equal(calls.length, 2);
  assert.deepEqual(calls[0], calls[1]);
  assert.equal(f.state.board, 'before');
  assert.equal(f.state.history.length, 1);
  assert.equal(f.state.scores[1], 0);
  assert.equal(f.saved.length, 3);
});

test('pending Undo does not mutate the board or permit a duplicate reversal', async () => {
  let resolve;
  let calls = 0;
  const f = fixture(() => { calls += 1; return new Promise((r) => { resolve = r; }); });
  const pending = f.undoTurn();
  assert.equal(f.state.busy, true);
  assert.equal(f.state.board, 'won');
  assert.equal(f.state.history, f.originalHistory);
  await f.undoTurn();
  assert.equal(calls, 1);
  resolve({ scores: { 1: 0 } });
  await pending;
  assert.equal(f.state.busy, false);
  assert.equal(f.state.board, 'before');
});

for (const fail of [false, true]) test(`late Undo ${fail ? 'failure' : 'success'} cannot overwrite a restarted round`, async () => {
  let resolve, reject;
  const f = fixture(() => new Promise((res, rej) => { resolve = res; reject = rej; }));
  const pending = f.undoTurn();
  f.state.version += 1;
  f.state.board = 'new round';
  f.state.history = [{ board: 'new round' }];
  f.state.busy = false;
  if (fail) reject(new Error('late abort'));
  else resolve({ scores: { 1: 0 } });
  await pending;
  assert.equal(f.state.board, 'new round');
  assert.equal(f.state.history.length, 1);
  assert.equal(f.saved.length, 1);
  assert.equal(f.saved[0][1].scoreReceipt.id, f.receipt.id);
});
