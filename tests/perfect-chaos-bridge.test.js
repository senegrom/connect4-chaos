import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  chaosFrontierStateToBoard,
  decodeChaosFrontier,
  encodeChaosFrontier,
  mergeChaosRejectionFiles,
  runPerfectChaosBridge,
} from '../scripts/perfect-chaos-bridge.mjs';
import { RED, YELLOW } from '../src/engine.js';

const MAGIC = Buffer.from('C4CFRN1\0', 'binary');

function rawFrontier(role, boundary, states) {
  const buffer = Buffer.alloc(16 + states.length * 19);
  MAGIC.copy(buffer, 0);
  buffer[8] = 1;
  buffer[9] = role;
  buffer[10] = boundary;
  buffer[11] = 19;
  buffer.writeUInt32LE(states.length, 12);
  for (let index = 0, offset = 16; index < states.length; index += 1, offset += 19) {
    const state = states[index];
    buffer.writeBigUInt64LE(state.mover, offset);
    buffer.writeBigUInt64LE(state.opponent, offset + 8);
    buffer[offset + 16] = state.rows;
    buffer[offset + 17] = state.columns;
    buffer[offset + 18] = state.aiTurn ? 1 : 0;
  }
  return buffer;
}

test('frontier encoding is canonical and decoding restores mover-relative boards', () => {
  const state = {
    mover: (1n << 0n) | (1n << 3n),
    opponent: 1n << 1n,
    rows: 2,
    columns: 2,
    aiTurn: true,
  };
  const frontier = decodeChaosFrontier(encodeChaosFrontier(1, 3, [state, state]));
  assert.equal(frontier.roleName, 'red');
  assert.equal(frontier.boundary, 3);
  assert.equal(frontier.states.length, 1);
  assert.deepEqual(chaosFrontierStateToBoard(frontier.states[0]), [
    [YELLOW, 0],
    [RED, RED],
  ]);
});

test('bridge scans emit generator-compatible rejection frontiers for proved AI losses', async (context) => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-bridge-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const frontierPath = join(directory, 'frontier.bin');
  const outputPath = join(directory, 'proof.ndjson');
  const rejectionPath = join(directory, 'reject-0.bin');
  await writeFile(frontierPath, encodeChaosFrontier(1, 0, [
    { mover: 0n, opponent: 0n, rows: 2, columns: 2, aiTurn: true },
    { mover: 0n, opponent: 0n, rows: 2, columns: 2, aiTurn: false },
  ]));

  const summary = await runPerfectChaosBridge({
    frontier: frontierPath,
    output: outputPath,
    rejections: rejectionPath,
    connect: 2,
    drop_depth: 4,
    maximum_states: 1_000,
    quiet: true,
  });
  const records = (await readFile(outputPath, 'utf8'))
    .trim()
    .split(/\r?\n/)
    .map((line) => JSON.parse(line));
  const rejections = decodeChaosFrontier(await readFile(rejectionPath));

  assert.equal(summary.selected, 2);
  assert.equal(summary.solved, 2);
  assert.equal(summary.aiWins, 1);
  assert.equal(summary.aiLosses, 1);
  assert.equal(summary.rejections, 1);
  assert.equal(summary.rejectionArtifact.records, 1);
  assert.match(summary.rejectionArtifact.sha256, /^[a-f0-9]{64}$/);
  assert.equal(records[0].aiValue, -1);
  assert.equal(records[0].rejected, true);
  assert.equal(records[1].aiValue, 1);
  assert.equal(records[1].rejected, false);
  assert.equal(rejections.states.length, 1);
  assert.equal(rejections.states[0].aiTurn, false);
  assert.ok(records.every((record) => record.action));
});

test('rejection merging sorts and deduplicates deterministic shard outputs', async (context) => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-merge-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const firstPath = join(directory, 'first.bin');
  const secondPath = join(directory, 'second.bin');
  const outputPath = join(directory, 'merged', 'reject-1.bin');
  const first = { mover: 1n, opponent: 0n, rows: 2, columns: 2, aiTurn: false };
  const second = { mover: 1n << 3n, opponent: 0n, rows: 2, columns: 2, aiTurn: true };
  await writeFile(firstPath, encodeChaosFrontier(1, 1, [second, first]));
  await writeFile(secondPath, encodeChaosFrontier(1, 1, [first]));

  const summary = await mergeChaosRejectionFiles([firstPath, secondPath], outputPath);
  const merged = decodeChaosFrontier(await readFile(outputPath));

  assert.equal(summary.role, 'red');
  assert.equal(summary.boundary, 1);
  assert.equal(summary.artifact.records, 2);
  assert.deepEqual(merged.states, [first, second]);
});

test('frontier decoding fails closed on unsorted states, sentinel bits and wrong counts', () => {
  assert.throws(
    () => decodeChaosFrontier(rawFrontier(1, 0, [
      { mover: 0n, opponent: 0n, rows: 2, columns: 2, aiTurn: true },
      { mover: 0n, opponent: 0n, rows: 2, columns: 2, aiTurn: false },
    ])),
    /strictly sorted/,
  );
  assert.throws(
    () => decodeChaosFrontier(rawFrontier(1, 0, [{
      mover: 1n << 2n,
      opponent: 0n,
      rows: 2,
      columns: 2,
      aiTurn: true,
    }])),
    /sentinel or out-of-board bits|frontier piece count/,
  );
});
