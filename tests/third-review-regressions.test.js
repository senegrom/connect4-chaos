import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { createExactTableLoader } from '../src/exact-table.js';
import {
  makeSnapshot,
  mergeScoreDelta,
  restoreSnapshot,
} from '../src/round-storage.js';
import { createBoard, RED, YELLOW, normalizeConfig } from '../src/engine.js';

function baseState() {
  return {
    config: normalizeConfig({ opponent: 'human' }),
    board: createBoard(6, 7),
    currentPlayer: RED,
    status: 'playing',
    winner: 0,
    winningCells: [],
    simultaneousWin: false,
    drawReason: null,
    lastMove: null,
    lastMover: null,
    moveCount: 0,
    selectedColumn: 3,
    repetitionCounts: new Map(),
    scores: { [RED]: 2, [YELLOW]: 1, draw: 0 },
    lastSearch: null,
    liveSearch: null,
    dropAnimation: null,
    aiError: null,
  };
}

test('shared score deltas preserve results written by another tab', () => {
  const shared = { [RED]: 8, [YELLOW]: 4, draw: 3 };
  const before = { [RED]: 2, [YELLOW]: 1, draw: 0 };
  assert.deepEqual(
    mergeScoreDelta(shared, before, { ...before, [RED]: 3 }),
    { [RED]: 9, [YELLOW]: 4, draw: 3 },
  );
  assert.deepEqual(
    mergeScoreDelta(shared, { ...before, [RED]: 3 }, before),
    { [RED]: 7, [YELLOW]: 4, draw: 3 },
  );
  assert.deepEqual(
    mergeScoreDelta(shared, before, before),
    shared,
    'a non-scoring move must never overwrite a newer shared tally',
  );
});

test('restoring a round can preserve the current shared scoreboard', () => {
  const state = baseState();
  const snapshot = makeSnapshot(state);
  snapshot.scores = { [RED]: 0, [YELLOW]: 0, draw: 0 };
  state.scores = { [RED]: 9, [YELLOW]: 5, draw: 4 };
  restoreSnapshot(state, snapshot, { restoreScores: false });
  assert.deepEqual(state.scores, { [RED]: 9, [YELLOW]: 5, draw: 4 });
});

test('exact-table loader times out and removes the failed cached promise so Retry can start fresh', async () => {
  const originalFetch = globalThis.fetch;
  let calls = 0;
  try {
    globalThis.fetch = (_url, { signal }) => {
      calls += 1;
      return new Promise((_resolve, reject) => {
        signal.addEventListener('abort', () => {
          reject(new DOMException('aborted', 'AbortError'));
        }, { once: true });
      });
    };
    const load = createExactTableLoader((bytes) => bytes, 'Perfect strategy', { timeoutMs: 5 });
    await assert.rejects(load('https://example.test/perfect.bin'), /did not finish loading/);
    await assert.rejects(load('https://example.test/perfect.bin'), /did not finish loading/);
    assert.equal(calls, 2, 'Retry must start a new fetch instead of reusing the timed-out promise');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('Perfect Chaos consent gate fails closed when the catalog cannot be checked', async () => {
  const source = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');
  const gateStart = source.indexOf('async function gateExactTableThenPost');
  const gateEnd = source.indexOf('\nconst loadedExactTables', gateStart);
  const gate = source.slice(gateStart, gateEnd);
  assert.match(gate, /catch \(error\)[\s\S]*stopAiWithError\([\s\S]*return;/);
  assert.doesNotMatch(gate, /catch \{[\s\S]*steps aside/);
});
