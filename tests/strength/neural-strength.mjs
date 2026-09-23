// How often the shipped network finds the only move that keeps a forced win.
//
// Every other test of the network checks that it runs, or that it plays a
// few hand-picked tactics. None of them notices the network or the search
// playing worse: for twelve days in September 2026 a batch-packing slip in
// makeEvaluateMany handed the WebGPU search empty or scrambled boards for
// every leaf of a batch, and every test passed. This one measures play.
//
// tests/fixtures/neural-strength-positions.json holds positions in which
// exactly one legal action keeps a forced win and at least one loses, each
// proved by the exact solvers (scripts/neural-strength-positions.mjs). Each
// is put to the network's raw policy, and to the search the page runs,
// through the batched evaluation WebGPU players get; from the search come
// both the move it plays and the move its own values rate highest. Each of
// the three must find nearly as many winning moves as it did when calibrated
// on the shipped network, and the search must not play fewer than the policy
// it starts from.
//
// The model is 106 MB and a run takes minutes, so this is not part of
// `npm test`: node's default patterns (`**/*.test.*`, `**/test/**`) do not
// match this file. Run it with `npm run test:strength`; CI runs it as the
// neural-strength job. Without a local model it skips, except in CI or with
// NEURAL_MODEL_DOWNLOAD=1, where a missing model fails. NEURAL_STRENGTH_THREADS
// sets the WebAssembly thread count (default: up to four).

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { availableParallelism } from 'node:os';
import test from 'node:test';

import { legalActions, sameAction } from '../../src/engine.js';
import { startBackend } from '../../src/neural-runtime.js';
import {
  SEARCH_BATCH, actionIndex, bestAction, searchPosition,
} from '../../src/neural-search.js';
import { modelManifest, readModelBytes } from '../../scripts/model-source.mjs';
import {
  FIXTURE, actionLabel, decodePosition, parseActionLabel, ruleName,
} from '../../scripts/neural-strength-positions.mjs';

// What chooses the move: the policy head alone, the search by its visit
// counts (what the page plays), and the search by its own values.
const MEASURES = ['prior', 'search', 'values'];

// The leaves a WebGPU search evaluates per network call; the calibration
// below used eight.
const BATCH = SEARCH_BATCH;
// The fewest that beat the policy head: on these positions the search plays
// 48 winning moves at 16 simulations and 49 at 24, against the policy head's
// 50, and 52 at 32. A run takes about five minutes on three threads.
const SIMULATIONS = 32;

// Measured on generation 504 (big504-808970a6d2) with this fixture on three
// WebAssembly threads: of the 60 only-winning moves the policy head finds 50,
// the search plays 52, and the search's values rate the winner highest in 58.
//
// With the batch-packing slip put back, the search still played 54. The root
// is evaluated on its own, so the prior guiding it was intact, and on these
// tactical positions the wins and losses the rules detect carried its choice.
// The garbage reached its values instead, which picked the winner in only 47:
// `values` is the measure that catches that class of bug.
const CALIBRATED = { positions: 60, prior: 50, search: 52, values: 58 };
// Each floor sits three positions below calibration, room for the float
// differences between platforms and thread counts that can tip a close call.
const MARGIN = 3;
// The search may not play fewer winning moves than its own policy head finds,
// less this: a search that trails its prior is reading broken values.
const SLACK = 2;

const download = process.env.NEURAL_MODEL_DOWNLOAD === '1';
const required = download || Boolean(process.env.CI);

function threadCount() {
  const requested = process.env.NEURAL_STRENGTH_THREADS;
  if (requested === undefined || requested === '') return Math.min(4, availableParallelism());
  const threads = Number(requested);
  if (!Number.isInteger(threads) || threads < 1) {
    throw new RangeError(`NEURAL_STRENGTH_THREADS must be a positive integer, not "${requested}".`);
  }
  return threads;
}

async function loadBackend() {
  const bytes = await readModelBytes({ allowDownload: download });
  if (!bytes) {
    if (required) {
      throw new Error('The exported network is required here (CI or NEURAL_MODEL_DOWNLOAD=1) '
        + 'but could not be found or downloaded.');
    }
    return null;
  }
  const ort = await import('onnxruntime-web');
  // Node runs the WebAssembly backend's threads on worker_threads. Where
  // they are unavailable the runtime itself falls back to one thread, and the
  // report below shows the count it settled on.
  ort.env.wasm.numThreads = threadCount();
  const backend = await startBackend(ort, bytes, 'wasm');
  return { ...backend, threads: ort.env.wasm.numThreads };
}

const fixture = JSON.parse(await readFile(FIXTURE, 'utf8'));
const backend = await loadBackend();
test.after(async () => { await backend?.session.release(); });

/** The legal action the policy head ranks first: the network with no search. */
function priorAction(policy, actions) {
  let best = actions[0];
  for (const action of actions) {
    if (policy[actionIndex(action)] > policy[actionIndex(best)]) best = action;
  }
  return best;
}

/** The root action whose mean value the search rates highest, among those
 * it visited: what its evaluation says, as opposed to its visit counts. */
function valuedAction(result) {
  let best = -1;
  for (let i = 0; i < result.actions.length; i += 1) {
    const value = result.actionValues[i];
    if (value !== null && (best < 0 || value > result.actionValues[best])) best = i;
  }
  return result.actions[best] ?? null;
}

function percent(count, total) {
  return `${String(count).padStart(3)} (${Math.round((100 * count) / total)}%)`.padEnd(11);
}

function report(results, { model, threads, seconds }) {
  const tally = (rows) => Object.fromEntries([['total', rows.length],
    ...MEASURES.map((measure) => [measure, rows.filter((row) => row[measure].found).length])]);
  const groups = Map.groupBy(results, (row) => row.rules);
  const all = tally(results);
  const lines = [
    `Only-winning-move benchmark: ${model}, ${SIMULATIONS} simulations in batches of ${BATCH}, `
      + `${threads} WebAssembly thread${threads === 1 ? '' : 's'}`,
    'prior: the policy head\'s top move; search: the move the search plays (most visits); '
      + 'values: the move the search\'s own values rate highest',
    `${'rules'.padEnd(18)}${'positions'.padEnd(11)}${MEASURES.map((measure) => measure.padEnd(11)).join('')}`,
  ];
  for (const [rules, counts] of [...[...groups].map(([name, rows]) => [name, tally(rows)]), ['all', all]]) {
    lines.push(`${rules.padEnd(18)}${String(counts.total).padStart(5).padEnd(11)}`
      + MEASURES.map((measure) => percent(counts[measure], counts.total)).join(''));
  }
  const misses = results.filter((row) => MEASURES.some((measure) => !row[measure].found));
  if (misses.length > 0) lines.push('misses (the move each chose, where it was not the only win):');
  for (const miss of misses) {
    lines.push(`  ${miss.id.padEnd(20)}${MEASURES.map((measure) => `${measure} `
      + `${(miss[measure].found ? 'ok' : miss[measure].chose).padEnd(10)}`).join(' ')} only ${miss.win} wins`);
  }
  lines.push(`${seconds.toFixed(1)} s, ${(seconds / results.length).toFixed(2)} s per position`);
  console.log(lines.join('\n'));
  return all;
}

const skip = backend ? false
  : 'no local model: set NEURAL_MODEL to the exported .onnx, or NEURAL_MODEL_DOWNLOAD=1 to fetch it';

test('the network and its search find the only move that keeps a forced win', { skip }, async () => {
  assert.equal(fixture.positions.length, CALIBRATED.positions,
    'The fixture changed: recalibrate CALIBRATED on the shipped network.');
  const started = performance.now();
  const results = [];
  for (const entry of fixture.positions) {
    const position = decodePosition(entry);
    const win = parseActionLabel(entry.win);
    const actions = legalActions(position.board, position.chaosMode);
    const output = await backend.evaluate(position.board, position.currentPlayer, actions,
      position.connect, position.chaosMode, 0);
    // The batched path is the one WebGPU players get; with batchSize 8 the
    // WebAssembly backend packs the same eight-position tensors.
    const result = await searchPosition(position, backend.evaluate, {
      simulations: SIMULATIONS, evaluateMany: backend.evaluateMany, batchSize: BATCH,
    });
    const chosen = {
      prior: priorAction(output.policy, actions),
      search: bestAction(result),
      values: valuedAction(result),
    };
    results.push({
      id: entry.id,
      rules: ruleName(entry),
      win: entry.win,
      ...Object.fromEntries(MEASURES.map((measure) => [measure, {
        found: sameAction(chosen[measure], win),
        chose: chosen[measure] ? actionLabel(chosen[measure]) : 'nothing',
      }])),
    });
  }
  const model = (await modelManifest()).source.replace(/\.pt$/, '');
  const all = report(results, {
    model, threads: backend.threads, seconds: (performance.now() - started) / 1000,
  });

  for (const measure of MEASURES) {
    assert.ok(all[measure] >= CALIBRATED[measure] - MARGIN,
      `${measure} found ${all[measure]} of ${all.total} only-winning moves; `
        + `calibrated ${CALIBRATED[measure]}, floor ${CALIBRATED[measure] - MARGIN}.`);
  }
  assert.ok(all.search >= all.prior - SLACK,
    `The search played ${all.search} winning moves, fewer than the ${all.prior} its own policy head `
      + 'finds: its values are broken.');
});
