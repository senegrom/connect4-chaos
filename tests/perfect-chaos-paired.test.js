import assert from 'node:assert/strict';
import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { crc32 } from 'node:zlib';

import { buildNative, findCompiler, runProcess } from '../scripts/native-build.mjs';

const SOURCE = fileURLToPath(new URL('../native/perfect-chaos-paired.cpp', import.meta.url));
// Block files open with a 32-byte header whose last word is the CRC-32 of
// everything after it.
const HEADER_BYTES = 32;
const CRC_OFFSET = 28;

// The pair-scheduled solver must reproduce the recorded counts exactly, and
// its canonical-composition slot space is locked by indexSpace: states are
// mirror-canonical in every solver, but only this one also halves the
// directories, so a drifting indexSpace means the geometry changed.
const EXPECTED = [
  {
    rows: 4, columns: 4, connect: 3, indexSpace: 454365,
    states: 31523, wins: 24888, draws: 864, losses: 5771, rootValue: 1,
  },
  {
    rows: 4, columns: 4, connect: 4, indexSpace: 454365,
    states: 239230, wins: 97779, draws: 110159, losses: 31292, rootValue: 0,
  },
];

function solve(binary, expected, output) {
  return runProcess(binary, [
    '--rows', String(expected.rows),
    '--columns', String(expected.columns),
    '--connect', String(expected.connect),
    '--threads', '2',
    '--output', output,
  ]);
}

function assertSolution(result, expected) {
  assert.equal(result.code, 0, `solve failed: ${result.stderr.slice(0, 2000)}`);
  const line = result.stdout.split('\n').find((entry) => entry.startsWith('{'));
  assert.ok(line, 'no solution line emitted');
  const solution = JSON.parse(line);
  assert.equal(solution.format, 'connect4-chaos-exact-solution-paired-v1');
  for (const field of ['indexSpace', 'states', 'wins', 'draws', 'losses', 'rootValue']) {
    assert.equal(solution[field], expected[field],
      `${expected.rows}x${expected.columns} c${expected.connect} ${field}`);
  }
}

async function solver(context) {
  if (!findCompiler()) {
    context.skip('no C++ compiler available');
    return null;
  }
  const { binary } = await buildNative(SOURCE, { name: 'perfect-chaos-paired' });
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-paired-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  return { binary, directory };
}

test('the pair-scheduled solver reproduces the recorded counts exactly', async (context) => {
  const built = await solver(context);
  if (!built) return;
  for (const expected of EXPECTED) {
    const out = join(built.directory, `${expected.rows}x${expected.columns}-c${expected.connect}`);
    assertSolution(await solve(built.binary, expected, out), expected);
  }
});

test('a block checkpoint whose tail reads back as zeros is solved again, not resumed', async (context) => {
  const built = await solver(context);
  if (!built) return;
  const expected = EXPECTED[1];
  const output = join(built.directory, 'solve');
  assertSolution(await solve(built.binary, expected, output), expected);

  // A power loss can leave a correctly sized file whose tail reads back as
  // zeros, and a packed LOSS is zero: loaded as is, the counts would drift.
  const values = join(output, 'pair-10-5.values');
  const bytes = await readFile(values);
  bytes.fill(0, HEADER_BYTES + Math.floor((bytes.length - HEADER_BYTES) / 2));
  await writeFile(values, bytes);

  const resumed = await solve(built.binary, expected, output);
  assert.match(resumed.stderr, /pair-10-5\.values: checksum mismatch/);
  assertSolution(resumed, expected);
});

test('a checkpoint that lost a reachable state stops the solve with an error', async (context) => {
  const built = await solver(context);
  if (!built) return;
  const expected = EXPECTED[0];
  const output = join(built.directory, 'solve');
  assertSolution(await solve(built.binary, expected, output), expected);

  // Drop the upper half of one block's reachable states and re-sign the file,
  // so only the successor check can notice. Every drop from the layer below
  // into that block is looked up while its summaries are built, and without
  // values every block is solved again on two threads.
  const bits = join(output, 'pair-8-4.bits');
  const bytes = await readFile(bits);
  const payload = bytes.subarray(HEADER_BYTES);
  payload.fill(0, Math.floor(payload.length / 2));
  bytes.writeUInt32LE(crc32(payload), CRC_OFFSET);
  await writeFile(bits, bytes);
  for (const name of await readdir(output)) {
    if (name.endsWith('.values')) await rm(join(output, name));
  }

  // The throw happens on a worker thread; it must end the run as an ordinary
  // error, not through std::terminate.
  const failed = await solve(built.binary, expected, output);
  assert.equal(failed.code, 1, failed.stderr);
  assert.equal(failed.signal, null);
  assert.match(failed.stderr, /a successor is missing from its block's reachable set/);
});
