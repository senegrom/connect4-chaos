import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import * as engine from '../src/engine.js';
import { makeSnapshot, restoreSnapshot, sameConfig, validSnapshot } from '../src/round-storage.js';

// Run the production restore path with the real engine, normalisation and
// snapshot validators; only rendering, storage and the AI are replaced.
const source = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');
const start = source.indexOf('function restoreSavedRound(');
const restoreSource = source.slice(start, source.indexOf('\n}\n', start) + 3);

function harness(savedConnect) {
  const config = engine.normalizeConfig({ rows: 6, cols: 7, connect: 5, opponent: 'human', startingPlayer: 1, chaosMode: false });
  const state = { config, board: engine.createBoard(6, 7), currentPlayer: 1, status: 'playing', winner: 0,
    winningCells: [], simultaneousWin: false, drawReason: null, lastMove: null, lastMover: null, moveCount: 0,
    selectedColumn: 0, repetitionCounts: new Map(), scores: { 1: 0, 2: 0, draw: 0 }, history: [],
    roundId: 'fresh', version: 1, busy: false, aiThinking: false, aiError: null, touchHintDismissed: false };
  const snapshot = makeSnapshot(state);
  // Connect-6 was offered until 2026-09-14; a save from then normalises to 5 today.
  const saved = { version: 1, config: { ...config, connect: savedConnect }, history: [snapshot], roundId: 'saved-round' };
  const calls = [];
  const context = { ...engine, sameConfig, validSnapshot, state,
    cancelAiSearch() { calls.push('cancel'); }, saveRound() { calls.push('save'); }, renderAll() { calls.push('render'); },
    restoreSnapshot(entry, options) { restoreSnapshot(state, entry, options); }, resultId: () => 'generated',
    isAiGame: () => false, requestAiMove() {}, stopAiWithError() {} };
  const restore = vm.runInNewContext(`${restoreSource}\nrestoreSavedRound;`, context);
  return { state, calls, restore: () => restore(saved) };
}

test('a round saved under Connect-6 is not resumed as a Connect-5 game', () => {
  const { state, calls, restore } = harness(6);
  assert.equal(engine.normalizeConfig({ rows: 6, cols: 7, connect: 6 }).connect, 5, 'the settings clamp at 5');
  assert.equal(restore(), false);
  assert.equal(state.roundId, 'fresh');
  assert.deepEqual(state.history, []);
  assert.deepEqual(calls, []);
});

test('a round saved under an offered Connect length still resumes', () => {
  const { state, calls, restore } = harness(5);
  assert.equal(restore(), true);
  assert.equal(state.roundId, 'saved-round');
  assert.equal(state.history.length, 1);
  assert.ok(calls.includes('save') && calls.includes('render'), 'the resumed round is re-saved and rendered');
});
