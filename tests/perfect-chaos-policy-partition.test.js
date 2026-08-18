import assert from 'node:assert/strict';
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';
import { spawn, spawnSync } from 'node:child_process';
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
  // Fall back to whatever the PATH offers, so a toolchain installed anywhere
  // other than /usr/bin still lets these tests run instead of skip.
  for (const candidate of ['g++', 'clang++']) {
    const probe = spawnSync(candidate, ['--version'], { encoding: 'utf8' });
    if (probe.status === 0) return candidate;
  }
  return null;
}

function frontierRecords(buffer) {
  assert.equal(buffer.subarray(0, 8).toString('binary'), 'C4CFRN1\0');
  assert.equal(buffer[8], 1);
  assert.equal(buffer[11], FRONTIER_RECORD_SIZE);
  const count = buffer.readUInt32LE(12);
  assert.equal(buffer.length, FRONTIER_HEADER_SIZE + count * FRONTIER_RECORD_SIZE);
  return Array.from({ length: count }, (_, index) => {
    const offset = FRONTIER_HEADER_SIZE + index * FRONTIER_RECORD_SIZE;
    return buffer.subarray(offset, offset + FRONTIER_RECORD_SIZE);
  });
}

function subsetFrontier(reference, records) {
  const output = Buffer.alloc(FRONTIER_HEADER_SIZE + records.length * FRONTIER_RECORD_SIZE);
  reference.copy(output, 0, 0, FRONTIER_HEADER_SIZE);
  output.writeUInt32LE(records.length, 12);
  Buffer.concat(records).copy(output, FRONTIER_HEADER_SIZE);
  return output;
}

function recordHex(record) {
  return record.toString('hex');
}

test('policy-root partition isolates exactly the roots that reach a new rejection', async (context) => {
  const cxx = await compiler();
  if (!cxx) {
    context.skip('A C++20 compiler is required for policy partition verification.');
    return;
  }

  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-policy-partition-'));
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

    const inputBuffer = await readFile(rootFrontier);
    const inputRecords = frontierRecords(inputBuffer);
    const fullBuffer = await readFile(fullFrontier);
    const boundaryRecords = frontierRecords(fullBuffer);
    assert.ok(inputRecords.length > 1);
    assert.ok(boundaryRecords.length > 1);

    let selected = null;
    for (const boundaryRecord of boundaryRecords) {
      const rejectPath = join(directory, 'reject-6.bin');
      await writeFile(rejectPath, subsetFrontier(fullBuffer, [boundaryRecord]));
      const unaffectedPath = join(directory, 'unaffected-4.bin');
      const affectedPath = join(directory, 'affected-4.bin');
      const summary = JSON.parse(await run(solver, [
        'partition',
        '--input-frontier', rootFrontier,
        '--policy', fullPolicy,
        '--reference-frontier', fullFrontier,
        '--reject-frontier', rejectPath,
        '--unaffected', unaffectedPath,
        '--affected', affectedPath,
      ]));
      if (summary.unaffectedRoots > 0 && summary.affectedRoots > 0) {
        selected = { boundaryRecord, rejectPath, unaffectedPath, affectedPath, summary };
        break;
      }
    }
    assert.ok(selected, 'Expected a frontier state reached by some but not all input roots.');
    assert.equal(selected.summary.format, 'connect4-chaos-policy-root-partition-v1');
    assert.equal(selected.summary.inputRoots, inputRecords.length);
    assert.equal(
      selected.summary.unaffectedRoots + selected.summary.affectedRoots,
      inputRecords.length,
    );
    assert.ok(selected.summary.rejectedBoundaryStatesReached > 0);

    const unaffectedRecords = frontierRecords(await readFile(selected.unaffectedPath));
    const affectedRecords = frontierRecords(await readFile(selected.affectedPath));
    const original = new Set(inputRecords.map(recordHex));
    const partitioned = new Set([...unaffectedRecords, ...affectedRecords].map(recordHex));
    assert.deepEqual(partitioned, original);
    assert.equal(
      unaffectedRecords.some((record) => affectedRecords.some((other) => record.equals(other))),
      false,
    );

    const slicedPolicy = join(directory, 'unaffected-sliced.policy.bin');
    const slicedFrontier = join(directory, 'unaffected-sliced.frontier.bin');
    await run(solver, [
      'slice',
      '--input-frontier', selected.unaffectedPath,
      '--policy', fullPolicy,
      '--reference-frontier', fullFrontier,
      '--output-policy', slicedPolicy,
      '--output-frontier', slicedFrontier,
    ]);
    const slicedBoundary = frontierRecords(await readFile(slicedFrontier));
    assert.equal(
      slicedBoundary.some((record) => record.equals(selected.boundaryRecord)),
      false,
      'An unaffected root still reaches the newly rejected frontier state.',
    );

    const directPolicy = join(directory, 'unaffected-direct.policy.bin');
    const directFrontier = join(directory, 'unaffected-direct.frontier.bin');
    await run(solver, [
      'extend',
      '--input-frontier', selected.unaffectedPath,
      '--frontier-pieces', '6',
      '--maximum-states', '2000000',
      '--policy', directPolicy,
      '--frontier', directFrontier,
      '--reject-frontier', selected.rejectPath,
      '--rejected', join(directory, 'unaffected.rejected.bin'),
    ]);
    assert.deepEqual(await readFile(slicedPolicy), await readFile(directPolicy));
    assert.deepEqual(await readFile(slicedFrontier), await readFile(directFrontier));

    const affectedSlicePolicy = join(directory, 'affected-sliced.policy.bin');
    const affectedSliceFrontier = join(directory, 'affected-sliced.frontier.bin');
    await run(solver, [
      'slice',
      '--input-frontier', selected.affectedPath,
      '--policy', fullPolicy,
      '--reference-frontier', fullFrontier,
      '--output-policy', affectedSlicePolicy,
      '--output-frontier', affectedSliceFrontier,
    ]);
    assert.equal(
      frontierRecords(await readFile(affectedSliceFrontier))
        .some((record) => record.equals(selected.boundaryRecord)),
      true,
      'Affected roots did not preserve a path to the selected rejection.',
    );

    const wrongBoundary = Buffer.from(await readFile(selected.rejectPath));
    wrongBoundary[9] = 2;
    const wrongBoundaryPath = join(directory, 'wrong-boundary.bin');
    await writeFile(wrongBoundaryPath, wrongBoundary);
    await assert.rejects(
      run(solver, [
        'partition',
        '--input-frontier', rootFrontier,
        '--policy', fullPolicy,
        '--reference-frontier', fullFrontier,
        '--reject-frontier', wrongBoundaryPath,
        '--unaffected', join(directory, 'invalid-unaffected.bin'),
        '--affected', join(directory, 'invalid-affected.bin'),
      ]),
      /metadata does not align/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
