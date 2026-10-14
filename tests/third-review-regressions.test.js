import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import * as chaosComplete from '../src/perfect-chaos-complete.js';
import { perfectClassicRole } from '../src/perfect-classic-policy.js';
import { releaseUrl } from '../src/data-loader.js';
import {
  ACTION_ROTATE_CW, applyAction, boardDimensions, createBoard, RED, YELLOW,
} from '../src/engine.js';

const MANIFEST_URL = new URL('../data/perfect-chaos-complete/manifest.json', import.meta.url);

// Runs the shipped gate in front of Perfect Chaos tables, not a pattern match
// on its source: a regex over it also matched the inner download catch, and
// passed with the outer catch falling through to the worker. `page` replaces
// what the gate uses of the page, the download dialog included.
async function gate(loadManifest, { position, ...page } = {}) {
  const source = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');
  const start = source.indexOf('async function gateExactTableThenPost(');
  // A module function run as a script: its import.meta and imports are
  // supplied by the harness.
  const body = source.slice(start, source.indexOf('\n}\n', start) + 2)
    .replaceAll('import.meta.url', 'moduleUrl');
  const manifest = JSON.parse(await readFile(MANIFEST_URL));
  // By default 4x4 Connect-3 with the AI moving first: a 3.5 KB table, below
  // the size that asks before downloading.
  const request = { id: 1, controller: new AbortController(), retrying: false, options: {},
    position: position ?? { board: createBoard(4, 4), startingPlayer: YELLOW, connect: 3 } };
  const calls = [];
  const context = vm.createContext({
    state: { aiRequest: request, aiRequestId: 1 }, YELLOW, boardDimensions, URL, releaseUrl,
    AbortController, setTimeout, clearTimeout,
    moduleUrl: new URL('../src/app.js', import.meta.url).href, loadedExactTables: new Set(),
    LARGE_TABLE_BYTES: 8_000_000, TABLE_DOWNLOAD_STALL_MS: 60_000,
    settings: { acceptCatalog() {} }, siteBuild: { async ensureCurrent() {} },
    requestDownload: () => assert.fail('no download for a small table'),
    ...chaosComplete, perfectClassicRole, loadPerfectChaosCompleteManifest: async () => loadManifest(manifest),
    postToWorker: () => calls.push('worker'),
    stopAiWithError: (message) => calls.push(`stopped: ${message}`),
    ...page,
  });
  vm.runInContext(body, context);
  await context.gateExactTableThenPost(request);
  return calls;
}

// The AI's first move of a 4x6 Connect-4 round the human opened: Yellow plays
// second, from the 36.5 MB table, which is asked for before it downloads.
async function largeTableGate(board) {
  const seen = [];
  const calls = await gate((manifest) => manifest, {
    position: { board, startingPlayer: RED, connect: 4 },
    requestDownload: async ({ title }) => { seen.push(['asked', title]); return true; },
    showDownloadProgress: ({ title }) => { seen.push(['progress', title]); return { update() {}, close() {} }; },
    fetchWithProgress: async (url) => { seen.push(['fetched', url]); return new ArrayBuffer(8); },
  });
  return { seen, calls };
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

// A rotation is legal on any board that is not full, the empty one included,
// so the AI can face 6x4 in a round set up as 4x6. Settings never offer
// Perfect 6x4 Chaos.
test('the large-table prompt and its progress name the round as it began, not as it lies', async () => {
  const board = applyAction(createBoard(4, 6), { type: ACTION_ROTATE_CW }, RED).board;
  assert.equal(boardDimensions(board).rows, 6);
  const { seen, calls } = await largeTableGate(board);
  assert.deepEqual(seen.filter(([step]) => step !== 'fetched'),
    [['asked', 'Perfect 4×6 Chaos'], ['progress', 'Perfect 4×6 Chaos']]);
  assert.deepEqual(calls, ['worker']);
});

test('the page downloads a large table by the release its catalog lists', async () => {
  const manifest = JSON.parse(await readFile(MANIFEST_URL));
  const entry = manifest.policies.find((policy) => policy.rows === 4 && policy.columns === 6
    && policy.connect === 4 && policy.role === 2);
  const { seen } = await largeTableGate(createBoard(4, 6));
  const [, url] = seen.find(([step]) => step === 'fetched');
  const fetched = new URL(url);
  assert.equal(fetched.pathname, new URL(entry.file, MANIFEST_URL).pathname);
  assert.equal(fetched.searchParams.get('v'), entry.sha256.slice(0, 12));
});
