// Compare the previous CPU startup policy with production in fresh processes.
// Uses the unchanged, committed model/runtime; RSS includes the whole Node
// process and is not an estimate of a physical iPhone's memory consumption.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { assetUrls, startBackend } from '../src/neural-runtime.js';
import { actionIndex } from '../src/neural-search.js';
import { applyAction, createBoard, legalActions, otherPlayer } from '../src/engine.js';

if (process.argv[2] === '--child') {
  const legacy = process.argv[3] === 'legacy';
  const urls = assetUrls();
  const ort = await import(urls.runtime);
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = urls.base;
  let bytes = await readFile(new URL(urls.model));
  const modelSha256 = createHash('sha256').update(bytes).digest('hex');
  const adapter = legacy ? { Tensor: ort.Tensor, InferenceSession: { create: (model, options) =>
    ort.InferenceSession.create(model, { ...options, graphOptimizationLevel: 'all' }) } } : ort;
  const started = performance.now();
  const starting = startBackend(adapter, bytes, 'wasm');
  if (!legacy) bytes = null;
  const network = await starting;
  bytes = null;
  globalThis.gc?.();
  const startupMs = performance.now() - started;
  const samples = [], times = [];
  for (const [rows, cols, connect, chaos] of [[6, 7, 4, false], [6, 7, 4, true], [4, 4, 3, true], [4, 10, 4, true]]) {
    let board = createBoard(rows, cols), mover = 1;
    for (let ply = 0; ply < 10; ply++) {
      const actions = legalActions(board, chaos);
      const start = performance.now();
      const output = await network.evaluate(board, mover, actions, connect, chaos, ply % 3);
      times.push(performance.now() - start);
      const logits = [...output.policy, ...output.value, ...output.q];
      assert.ok(logits.every(Number.isFinite));
      const best = actions.reduce((a, b) => output.policy[actionIndex(a)] >= output.policy[actionIndex(b)] ? a : b);
      samples.push({ logits, action: actionIndex(best) });
      board = applyAction(board, actions[(ply * 3 + 2) % actions.length], mover).board;
      mover = otherPlayer(mover);
    }
  }
  globalThis.gc?.();
  const result = { mode: legacy ? 'legacy' : 'production', modelSha256, startupMs: Math.round(startupMs),
    peakMiB: Math.round(process.resourceUsage().maxRSS / 1024),
    residentMiB: Math.round(process.memoryUsage().rss / 1048576),
    medianEvaluationMs: Math.round(times.sort((a, b) => a - b)[Math.floor(times.length / 2)]), samples };
  await network.session.release();
  console.log(JSON.stringify(result));
} else {
  const reports = ['legacy', 'production'].map((mode) => {
    const result = spawnSync(process.execPath, ['--expose-gc', fileURLToPath(import.meta.url), '--child', mode],
      { encoding: 'utf8', timeout: 120_000 });
    assert.equal(result.status, 0, result.stderr || String(result.error));
    return JSON.parse(result.stdout);
  });
  assert.equal(reports[0].modelSha256, reports[1].modelSha256);
  let maxLogitDifference = 0, changedActions = 0;
  for (let i = 0; i < reports[0].samples.length; i++) {
    const [a, b] = reports.map((report) => report.samples[i]);
    if (a.action !== b.action) changedActions++;
    for (let j = 0; j < a.logits.length; j++) maxLogitDifference = Math.max(maxLogitDifference, Math.abs(a.logits[j] - b.logits[j]));
  }
  console.log(JSON.stringify({ reports: reports.map(({ samples, ...report }) => report),
    comparedPositions: reports[0].samples.length, changedActions, maxLogitDifference }, null, 2));
}
