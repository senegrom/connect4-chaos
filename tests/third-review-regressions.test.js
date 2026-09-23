import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import * as chaosComplete from '../src/perfect-chaos-complete.js';
import {
  makeSnapshot,
  mergeScoreDelta,
  restoreSnapshot,
} from '../src/round-storage.js';
import { boardDimensions, createBoard, RED, YELLOW, normalizeConfig } from '../src/engine.js';

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

// Runs the shipped gate in front of Perfect Chaos tables, not a pattern match
// on its source: a regex over it also matched the inner download catch, and
// passed with the outer catch falling through to the worker.
async function gate(loadManifest) {
  const source = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');
  const start = source.indexOf('async function gateExactTableThenPost(');
  // A module function run as a script: its import.meta and dynamic imports
  // are supplied by the harness.
  const body = source.slice(start, source.indexOf('\n}\n', start) + 2)
    .replaceAll('import.meta.url', 'moduleUrl').replaceAll('await import(', 'await importModule(');
  const manifest = JSON.parse(await readFile(new URL('../data/perfect-chaos-complete/manifest.json', import.meta.url)));
  // 4x4 Connect-3 with the AI moving first: a 3.5 KB table, below the size
  // that asks before downloading.
  const request = { id: 1, controller: new AbortController(), retrying: false, options: {},
    position: { board: createBoard(4, 4), startingPlayer: YELLOW, connect: 3 } };
  const calls = [];
  const context = vm.createContext({
    state: { aiRequest: request, aiRequestId: 1 }, YELLOW, boardDimensions, URL,
    moduleUrl: new URL('../src/app.js', import.meta.url).href, loadedExactTables: new Set(),
    LARGE_TABLE_BYTES: 8_000_000, TABLE_DOWNLOAD_TIMEOUT_MS: 600_000,
    settings: { acceptCatalog() {} },
    importModule: async (specifier) => {
      assert.equal(specifier, './perfect-chaos-complete.js', 'no download for a small table');
      return { ...chaosComplete, loadPerfectChaosCompleteManifest: async () => loadManifest(manifest) };
    },
    postToWorker: () => calls.push('worker'),
    stopAiWithError: (message) => calls.push(`stopped: ${message}`),
  });
  vm.runInContext(body, context);
  await context.gateExactTableThenPost(request);
  return calls;
}

test('Perfect Chaos consent gate fails closed when the catalog cannot be checked', async () => {
  const calls = await gate(() => { throw new Error('catalog unavailable'); });
  assert.equal(calls.length, 1);
  assert.match(calls[0], /^stopped: .*could not be checked.*catalog unavailable/);
  assert.ok(!calls.includes('worker'), 'nothing may reach the worker without an authorised catalog entry');
});

test('the same gate passes a checked small table straight to the worker', async () => {
  // The harness sees postToWorker, so the test above cannot pass by missing it.
  assert.deepEqual(await gate((manifest) => manifest), ['worker']);
});
