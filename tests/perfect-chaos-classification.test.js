import assert from 'node:assert/strict';
import { access, mkdtemp, readFile, rm } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawn } from 'node:child_process';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const NATIVE_SOURCE = join(ROOT, 'native', 'perfect-chaos-prefix.cpp');
const CLASSIFIER = join(ROOT, 'scripts', 'perfect-chaos-classify.py');
const MERGER = join(ROOT, 'scripts', 'perfect-chaos-merge-classification.py');

function run(command, args, options = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, {
      cwd: ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
      ...options,
    });
    const stdout = [];
    const stderr = [];
    child.stdout.on('data', (chunk) => stdout.push(chunk));
    child.stderr.on('data', (chunk) => stderr.push(chunk));
    child.once('error', reject);
    child.once('close', (code, signal) => {
      const output = Buffer.concat(stdout).toString('utf8');
      const errors = Buffer.concat(stderr).toString('utf8');
      if (code === 0) {
        resolvePromise(output);
        return;
      }
      reject(new Error(
        `${command} exited with ${code ?? signal}.\n${errors || output}`,
      ));
    });
  });
}

async function executable(path) {
  try {
    await access(path, fsConstants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function compiler() {
  if (process.env.CXX && await executable(process.env.CXX)) return process.env.CXX;
  for (const candidate of ['/usr/bin/g++', '/usr/bin/clang++']) {
    if (await executable(candidate)) return candidate;
  }
  return null;
}

async function python() {
  for (const candidate of ['python3', 'python']) {
    try {
      await run(candidate, ['--version']);
      return candidate;
    } catch {
      // Try the next interpreter.
    }
  }
  return null;
}

test('distributed Perfect Chaos classification matches the direct native segment', async (context) => {
  const cxx = await compiler();
  const pythonCommand = await python();
  if (!cxx || !pythonCommand) {
    context.skip('A C++20 compiler and Python are required for proof-tool integration.');
    return;
  }

  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-classification-'));
  try {
    const solver = join(directory, 'perfect-chaos-prefix');
    await run(cxx, [
      '-std=c++20', '-O2', '-DNDEBUG', '-Wall', '-Wextra', '-Wpedantic',
      NATIVE_SOURCE, '-o', solver,
    ]);

    const rootPolicy = join(directory, '0-4.policy.bin');
    const rootFrontier = join(directory, '0-4.frontier.bin');
    await run(solver, [
      'generate', '--role', 'red', '--frontier-pieces', '4',
      '--maximum-states', '1000000', '--policy', rootPolicy, '--frontier', rootFrontier,
    ]);

    const directPolicy = join(directory, 'direct.policy.bin');
    const directFrontier = join(directory, 'direct.frontier.bin');
    await run(solver, [
      'extend', '--input-frontier', rootFrontier, '--frontier-pieces', '6',
      '--maximum-states', '2000000', '--policy', directPolicy, '--frontier', directFrontier,
      '--rejected', join(directory, 'direct.rejected.bin'),
    ]);

    const shardDirectory = join(directory, 'shards');
    await run('mkdir', ['-p', shardDirectory]);
    for (let shard = 0; shard < 2; shard += 1) {
      const output = await run(pythonCommand, [
        CLASSIFIER,
        '--solver', solver,
        '--input', rootFrontier,
        '--role', 'red',
        '--target-pieces', '6',
        '--shard-index', String(shard),
        '--shard-count', '2',
        '--maximum-states', '10000',
        '--rejected', join(shardDirectory, `rejected-${shard}.bin`),
        '--policy', join(shardDirectory, `policy-${shard}.bin`),
        '--frontier', join(shardDirectory, `frontier-${shard}.bin`),
        '--summary', join(shardDirectory, `summary-${shard}.json`),
      ]);
      const summary = JSON.parse(output);
      assert.equal(summary.classificationComplete, true);
      assert.equal(summary.rejectedRoots, 0);
      assert.ok(summary.splitEvents > 0);
    }

    const mergedRejected = join(directory, 'merged.rejected.bin');
    const mergedPolicy = join(directory, 'merged.policy.bin');
    const mergedFrontier = join(directory, 'merged.frontier.bin');
    const mergedOutput = await run(pythonCommand, [
      MERGER,
      '--directory', shardDirectory,
      '--input', rootFrontier,
      '--role', 'red',
      '--target-pieces', '6',
      '--shard-count', '2',
      '--rejected', mergedRejected,
      '--policy', mergedPolicy,
      '--frontier', mergedFrontier,
      '--summary', join(directory, 'merged-summary.json'),
    ]);
    const merged = JSON.parse(mergedOutput);
    assert.equal(merged.classificationComplete, true);
    assert.equal(merged.inputRoots, 59);
    assert.equal(merged.safeInputRoots, 59);
    assert.equal(merged.rejectedRoots, 0);
    assert.equal(merged.policyConflicts, 0);
    assert.deepEqual(await readFile(mergedPolicy), await readFile(directPolicy));
    assert.deepEqual(await readFile(mergedFrontier), await readFile(directFrontier));
    assert.equal((await readFile(mergedRejected)).length, 16);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});


test('policy table writer fails closed on conflicting actions', async (context) => {
  const pythonCommand = await python();
  if (!pythonCommand) {
    context.skip('Python is required for proof-table validation.');
    return;
  }

  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-policy-conflict-'));
  try {
    const output = join(directory, 'conflicting.policy.bin');
    const script = `
import struct
from pathlib import Path
from perfect_chaos_tables import POLICY_MAGIC, POLICY_RECORD_SIZE, write_table

def record(action):
    value = bytearray(POLICY_RECORD_SIZE)
    struct.pack_into('<QQ', value, 0, 0, 0)
    value[16] = 6
    value[17] = 7
    value[18] = action
    value[19] = 0
    return bytes(value)

try:
    write_table(
        Path(${JSON.stringify(output)}),
        POLICY_MAGIC,
        1,
        2,
        POLICY_RECORD_SIZE,
        [record(1), record(2)],
    )
except RuntimeError as error:
    if 'Conflicting Perfect Chaos policy actions.' not in str(error):
        raise
else:
    raise RuntimeError('Conflicting policy actions were silently merged.')
`;
    await run(pythonCommand, ['-c', script], {
      env: {
        ...process.env,
        PYTHONPATH: join(ROOT, 'scripts'),
      },
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
