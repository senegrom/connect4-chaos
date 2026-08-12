import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawn } from 'node:child_process';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const SCRIPT = join(ROOT, 'scripts', 'perfect-chaos.mjs');

function run(args) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(process.execPath, [SCRIPT, ...args], {
      cwd: ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    const stdout = [];
    const stderr = [];
    child.stdout.on('data', (chunk) => stdout.push(chunk));
    child.stderr.on('data', (chunk) => stderr.push(chunk));
    child.once('error', reject);
    child.once('close', (code) => {
      const output = Buffer.concat(stdout).toString('utf8');
      if (code === 0) {
        resolvePromise(output);
        return;
      }
      reject(new Error(Buffer.concat(stderr).toString('utf8') || output));
    });
  });
}

const records = [
  {
    id: 'tiny-win',
    position: {
      board: [[0, 0], [0, 0]],
      currentPlayer: 1,
      connect: 2,
      chaosMode: true,
    },
  },
  {
    id: 'tiny-draw',
    position: {
      board: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
      currentPlayer: 1,
      connect: 3,
      chaosMode: true,
    },
  },
  {
    id: 'rotation-win',
    position: {
      board: [
        [1, 1, 1, 2, 1, 0, 0],
        [2, 2, 2, 1, 2, 0, 0],
        [2, 1, 2, 1, 2, 1, 0],
        [2, 1, 1, 1, 2, 2, 0],
        [1, 2, 2, 2, 1, 2, 2],
        [1, 1, 2, 2, 1, 1, 1],
      ],
      currentPlayer: 1,
      connect: 4,
      chaosMode: true,
    },
  },
];

test('Perfect Chaos frontier solving is deterministic and shard-complete', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-frontier-'));
  try {
    const input = join(directory, 'positions.ndjson');
    const first = join(directory, 'first.ndjson');
    const second = join(directory, 'second.ndjson');
    await writeFile(input, `${records.map((record) => JSON.stringify(record)).join('\n')}\n`);

    const firstSummary = JSON.parse(await run([
      'frontier', '--input', input, '--output', first, '--shard-index', '0', '--shard-count', '2',
    ]));
    const secondSummary = JSON.parse(await run([
      'frontier', '--input', input, '--output', second, '--shard-index', '1', '--shard-count', '2',
    ]));
    assert.equal(firstSummary.solved, 2);
    assert.equal(secondSummary.solved, 1);

    const output = [];
    for (const path of [first, second]) {
      const lines = (await readFile(path, 'utf8')).trim().split(/\r?\n/).filter(Boolean);
      output.push(...lines.map((line) => JSON.parse(line)));
    }
    output.sort((left, right) => String(left.id).localeCompare(String(right.id)));
    assert.deepEqual(output.map(({ id, value, states, action }) => ({ id, value, states, action })), [
      { id: 'rotation-win', value: 1, states: 2_585, action: { type: 'rotateCW' } },
      { id: 'tiny-draw', value: 0, states: 628, action: { type: 'drop', column: 0 } },
      { id: 'tiny-win', value: 1, states: 6, action: { type: 'drop', column: 0 } },
    ]);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});


test('Perfect Chaos root enumeration is canonical and reproducible', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-enumeration-'));
  try {
    const output = join(directory, 'frontier.ndjson');
    const summary = JSON.parse(await run([
      'enumerate', '--depth', '4', '--output', output,
    ]));
    assert.deepEqual(summary.layers.map(({ frontier, cumulative }) => ({ frontier, cumulative })), [
      { frontier: 1, cumulative: 1 },
      { frontier: 5, cumulative: 6 },
      { frontier: 33, cumulative: 39 },
      { frontier: 164, cumulative: 203 },
      { frontier: 837, cumulative: 1_040 },
    ]);
    const lines = (await readFile(output, 'utf8')).trim().split(/\r?\n/).filter(Boolean);
    assert.equal(lines.length, 837);
    const ids = lines.map((line) => JSON.parse(line).id);
    assert.deepEqual(ids, [...ids].sort());
    assert.equal(new Set(ids).size, ids.length);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
