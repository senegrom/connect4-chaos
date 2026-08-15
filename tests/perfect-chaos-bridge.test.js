import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  chaosFrontierStateToBoard,
  decodeChaosFrontier,
  runPerfectChaosBridge,
} from '../scripts/perfect-chaos-bridge.mjs';
import { RED, YELLOW } from '../src/engine.js';

const MAGIC = Buffer.from('C4CFRN1\0', 'binary');

function encodeFrontier(role, boundary, states) {
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

test('frontier decoding restores gravity-settled mover-relative boards', () => {
  const state = {
    mover: (1n << 0n) | (1n << 3n),
    opponent: 1n << 1n,
    rows: 2,
    columns: 2,
    aiTurn: true,
  };
  const frontier = decodeChaosFrontier(encodeFrontier(1, 3, [state]));
  assert.equal(frontier.roleName, 'red');
  assert.equal(frontier.boundary, 3);
  assert.deepEqual(chaosFrontierStateToBoard(frontier.states[0]), [
    [YELLOW, 0],
    [RED, RED],
  ]);
});

test('bridge scans report exact mover and certificate-AI values separately', async (context) => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-bridge-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const frontierPath = join(directory, 'frontier.bin');
  const outputPath = join(directory, 'proof.ndjson');
  await writeFile(frontierPath, encodeFrontier(1, 0, [
    { mover: 0n, opponent: 0n, rows: 2, columns: 2, aiTurn: true },
    { mover: 0n, opponent: 0n, rows: 2, columns: 2, aiTurn: false },
  ]));

  const summary = await runPerfectChaosBridge({
    frontier: frontierPath,
    output: outputPath,
    connect: 2,
    drop_depth: 4,
    maximum_states: 1_000,
    quiet: true,
  });
  const records = (await readFile(outputPath, 'utf8'))
    .trim()
    .split(/\r?\n/)
    .map((line) => JSON.parse(line));

  assert.equal(summary.selected, 2);
  assert.equal(summary.solved, 2);
  assert.equal(summary.aiWins, 1);
  assert.equal(summary.aiLosses, 1);
  assert.equal(records[0].moverLower, 1);
  assert.equal(records[0].moverUpper, 1);
  assert.equal(records[0].aiValue, 1);
  assert.equal(records[1].moverLower, 1);
  assert.equal(records[1].moverUpper, 1);
  assert.equal(records[1].aiValue, -1);
  assert.ok(records.every((record) => record.action));
});

test('frontier decoding rejects sentinel bits and wrong boundary counts', () => {
  assert.throws(
    () => decodeChaosFrontier(encodeFrontier(1, 0, [{
      mover: 1n << 2n,
      opponent: 0n,
      rows: 2,
      columns: 2,
      aiTurn: true,
    }])),
    /sentinel or out-of-board bits|frontier piece count/,
  );
});
