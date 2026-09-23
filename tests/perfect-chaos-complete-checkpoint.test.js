import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { crc32 } from 'node:zlib';

import { buildNative, findCompiler, runProcess } from '../scripts/native-build.mjs';

const SOURCE = fileURLToPath(new URL('../native/perfect-chaos-complete.cpp', import.meta.url));
// Checkpoints open with a 32-byte header carrying the CRC-32 of everything
// after it at offset 24.
const HEADER_BYTES = 32;
const CRC_OFFSET = 24;
// 4x4 Chaos Connect Three, as recorded in the committed catalog.
const EXPECTED = { states: 31523, wins: 24888, draws: 864, losses: 5771, maximumRank: 13, rootValue: 1 };

async function solver(context) {
  if (!findCompiler()) {
    context.skip('no C++ compiler available');
    return null;
  }
  const { binary } = await buildNative(SOURCE, { name: 'perfect-chaos-complete' });
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-complete-checkpoint-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const checkpoint = join(directory, 'solve');
  const solve = () => runProcess(binary, ['--rows', '4', '--columns', '4', '--connect', '3',
    '--threads', '2', '--checkpoint', checkpoint, '--keep-checkpoint']);
  return { checkpoint, solve };
}

function assertSolution(result) {
  assert.equal(result.code, 0, result.stderr);
  const solution = JSON.parse(result.stdout.split('\n').find((line) => line.startsWith('{')));
  for (const [field, value] of Object.entries(EXPECTED)) assert.equal(solution[field], value, field);
}

test('a round checkpoint whose tail reads back as zeros is recomputed, not resumed', async (context) => {
  const built = await solver(context);
  if (!built) return;
  assertSolution(await built.solve());

  // Zeros are LOSS values with rank zero: resumed as is, the counts drift.
  const round = `${built.checkpoint}.round`;
  const bytes = await readFile(round);
  bytes.fill(0, HEADER_BYTES + Math.floor((bytes.length - HEADER_BYTES) / 2));
  await writeFile(round, bytes);

  const resumed = await built.solve();
  assert.match(resumed.stderr, /bitset checkpoint loaded/);
  assert.match(resumed.stderr, /round checkpoint .* rejected: checksum mismatch/);
  assertSolution(resumed);
});

test('a discovery checkpoint that lost reachable states stops the solve with an error', async (context) => {
  const built = await solver(context);
  if (!built) return;
  assertSolution(await built.solve());

  // Drop the upper half of the reachable states and re-sign the bitset, so
  // only the successor check can notice; without a round checkpoint, every
  // state is ranked again on two threads.
  const bitset = `${built.checkpoint}.bitset`;
  const bytes = await readFile(bitset);
  const words = bytes.subarray(HEADER_BYTES + 8);
  words.fill(0, Math.floor(words.length / 2));
  bytes.writeUInt32LE(crc32(bytes.subarray(HEADER_BYTES)), CRC_OFFSET);
  await writeFile(bitset, bytes);
  await rm(`${built.checkpoint}.round`);

  // The throw happens on a worker thread; it must end the run as an ordinary
  // error, not through std::terminate.
  const failed = await built.solve();
  assert.equal(failed.code, 1, failed.stderr);
  assert.equal(failed.signal, null);
  assert.match(failed.stderr, /a successor is missing from the reachable set/);
});
