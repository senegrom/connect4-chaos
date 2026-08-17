import assert from 'node:assert/strict';
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawn } from 'node:child_process';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const NATIVE_SOURCE = join(ROOT, 'native', 'perfect-chaos-prefix.cpp');
const FRONTIER_HEADER_SIZE = 16;
const FRONTIER_RECORD_SIZE = 19;

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
      if (code === 0) resolvePromise(output);
      else reject(new Error(`${command} exited with ${code ?? signal}.\n${errors || output}`));
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

async function subsetFrontier(sourcePath, targetPath) {
  const source = await readFile(sourcePath);
  const count = source.readUInt32LE(12);
  assert.equal(source.length, FRONTIER_HEADER_SIZE + count * FRONTIER_RECORD_SIZE);
  const selected = [];
  for (let index = 0; index < count; index += 2) {
    const offset = FRONTIER_HEADER_SIZE + index * FRONTIER_RECORD_SIZE;
    selected.push(source.subarray(offset, offset + FRONTIER_RECORD_SIZE));
  }
  const target = Buffer.alloc(FRONTIER_HEADER_SIZE + selected.length * FRONTIER_RECORD_SIZE);
  source.copy(target, 0, 0, FRONTIER_HEADER_SIZE);
  target.writeUInt32LE(selected.length, 12);
  Buffer.concat(selected).copy(target, FRONTIER_HEADER_SIZE);
  await writeFile(targetPath, target);
  return selected.length;
}

function emptyFrontier(reference) {
  const result = Buffer.from(reference.subarray(0, FRONTIER_HEADER_SIZE));
  result.writeUInt32LE(0, 12);
  return result;
}

test('native policy slicing reproduces direct extension for a frontier subset', async (context) => {
  const cxx = await compiler();
  if (!cxx) {
    context.skip('A C++20 compiler is required for policy-slice verification.');
    return;
  }

  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-policy-slice-'));
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

    const fullPolicy = join(directory, '4-6.policy.bin');
    const fullFrontier = join(directory, '4-6.frontier.bin');
    await run(solver, [
      'extend', '--input-frontier', rootFrontier, '--frontier-pieces', '6',
      '--maximum-states', '2000000', '--policy', fullPolicy, '--frontier', fullFrontier,
      '--rejected', join(directory, 'full.rejected.bin'),
    ]);

    const subset = join(directory, 'subset.frontier.bin');
    const subsetCount = await subsetFrontier(rootFrontier, subset);
    assert.ok(subsetCount > 0);

    const slicedPolicy = join(directory, 'sliced.policy.bin');
    const slicedFrontier = join(directory, 'sliced.frontier.bin');
    const sliceOutput = JSON.parse(await run(solver, [
      'slice', '--input-frontier', subset,
      '--policy', fullPolicy,
      '--reference-frontier', fullFrontier,
      '--output-policy', slicedPolicy,
      '--output-frontier', slicedFrontier,
    ]));
    assert.equal(sliceOutput.format, 'connect4-chaos-policy-slice-v1');
    assert.equal(sliceOutput.inputRoots, subsetCount);
    assert.ok(sliceOutput.policyEntries > 0);
    assert.ok(sliceOutput.frontierStates > 0);

    const directPolicy = join(directory, 'direct.policy.bin');
    const directFrontier = join(directory, 'direct.frontier.bin');
    await run(solver, [
      'extend', '--input-frontier', subset, '--frontier-pieces', '6',
      '--maximum-states', '2000000', '--policy', directPolicy, '--frontier', directFrontier,
      '--rejected', join(directory, 'direct.rejected.bin'),
    ]);

    assert.deepEqual(await readFile(slicedPolicy), await readFile(directPolicy));
    assert.deepEqual(await readFile(slicedFrontier), await readFile(directFrontier));

    const incompleteReference = join(directory, 'incomplete.frontier.bin');
    await writeFile(incompleteReference, emptyFrontier(await readFile(fullFrontier)));
    await assert.rejects(
      run(solver, [
        'slice', '--input-frontier', subset,
        '--policy', fullPolicy,
        '--reference-frontier', incompleteReference,
        '--output-policy', join(directory, 'invalid.policy.bin'),
        '--output-frontier', join(directory, 'invalid.frontier.bin'),
      ]),
      /outside the reference certificate/,
    );

    const malformedInput = join(directory, 'malformed.frontier.bin');
    const malformed = Buffer.from(await readFile(subset));
    const mover = malformed.readBigUInt64LE(FRONTIER_HEADER_SIZE);
    malformed.writeBigUInt64LE(mover | (1n << 6n), FRONTIER_HEADER_SIZE);
    await writeFile(malformedInput, malformed);
    await assert.rejects(
      run(solver, [
        'slice', '--input-frontier', malformedInput,
        '--policy', fullPolicy,
        '--reference-frontier', fullFrontier,
        '--output-policy', join(directory, 'malformed.policy.bin'),
        '--output-frontier', join(directory, 'malformed-output.frontier.bin'),
      ]),
      /sentinel bit/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
