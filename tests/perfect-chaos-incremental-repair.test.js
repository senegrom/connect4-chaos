import assert from 'node:assert/strict';
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const SCRIPT = join(ROOT, 'scripts', 'perfect-chaos-prefix.mjs');
const NATIVE_SOURCE = join(ROOT, 'native', 'perfect-chaos-prefix.cpp');
const FRONTIER_HEADER_SIZE = 16;
const FRONTIER_RECORD_SIZE = 19;

function runResult(command, args, options = {}) {
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
    child.once('close', (code, signal) => resolvePromise({
      code,
      signal,
      stdout: Buffer.concat(stdout).toString('utf8'),
      stderr: Buffer.concat(stderr).toString('utf8'),
    }));
  });
}

async function run(command, args, options = {}) {
  const result = await runResult(command, args, options);
  if (result.code !== 0) {
    throw new Error(`${command} exited with ${result.code ?? result.signal}.\n${result.stderr || result.stdout}`);
  }
  return result.stdout;
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

async function buildSmallCertificate(directory, cxx) {
  const solver = join(directory, 'perfect-chaos-prefix');
  await run(cxx, [
    '-std=c++20', '-O2', '-DNDEBUG', '-Wall', '-Wextra', '-Wpedantic',
    NATIVE_SOURCE, '-o', solver,
  ]);
  const inputPolicy = join(directory, '0-4.policy.bin');
  const inputFrontier = join(directory, '0-4.frontier.bin');
  await run(solver, [
    'generate', '--role', 'red', '--frontier-pieces', '4',
    '--maximum-states', '1000000', '--policy', inputPolicy, '--frontier', inputFrontier,
  ]);
  const seedPolicy = join(directory, '4-6.policy.bin');
  const seedFrontier = join(directory, '4-6.frontier.bin');
  await run(solver, [
    'extend', '--input-frontier', inputFrontier, '--frontier-pieces', '6',
    '--maximum-states', '2000000', '--policy', seedPolicy, '--frontier', seedFrontier,
    '--rejected', join(directory, 'seed.rejected.bin'),
  ]);
  return { solver, inputFrontier, seedPolicy, seedFrontier };
}

async function invokeRepair(directory, paths, rejectFrontier, suffix) {
  const outputPolicy = join(directory, `${suffix}.policy.bin`);
  const outputFrontier = join(directory, `${suffix}.frontier.bin`);
  const rejected = join(directory, `${suffix}.rejected.bin`);
  const output = await run(process.execPath, [
    SCRIPT,
    'repair-segment',
    '--input-frontier', paths.inputFrontier,
    '--seed-input-frontier', paths.inputFrontier,
    '--seed-policy', paths.seedPolicy,
    '--seed-frontier', paths.seedFrontier,
    '--reject-frontier', rejectFrontier,
    '--frontier-pieces', '6',
    '--maximum-states', '2000000',
    '--shards', '2',
    '--minimum-states-per-shard', '10000',
    '--shard-workers', '2',
    '--output-policy', outputPolicy,
    '--output-frontier', outputFrontier,
    '--rejected', rejected,
  ]);
  return {
    report: JSON.parse(output).result,
    outputPolicy,
    outputFrontier,
    rejected,
  };
}

test('incremental repair preserves an unchanged exact segment byte-for-byte', async (context) => {
  const cxx = await compiler();
  if (!cxx) {
    context.skip('A C++20 compiler is required for incremental repair verification.');
    return;
  }
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-incremental-noop-'));
  try {
    const paths = await buildSmallCertificate(directory, cxx);
    const seedFrontierBytes = await readFile(paths.seedFrontier);
    const rejectFrontier = join(directory, 'reject-6-empty.bin');
    await writeFile(rejectFrontier, subsetFrontier(seedFrontierBytes, []));
    const repaired = await invokeRepair(directory, paths, rejectFrontier, 'noop-repaired');
    assert.equal(repaired.report.status, 'safe');
    assert.equal(repaired.report.inputRoots, 59);
    assert.equal(repaired.report.unaffectedRoots, 59);
    assert.equal(repaired.report.repairRoots, 0);
    assert.equal(repaired.report.fallbackFullRegeneration, false);
    assert.deepEqual(await readFile(repaired.outputPolicy), await readFile(paths.seedPolicy));
    assert.deepEqual(await readFile(repaired.outputFrontier), seedFrontierBytes);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('incremental repair matches full exact regeneration after a partial dependency change', async (context) => {
  const cxx = await compiler();
  if (!cxx) {
    context.skip('A C++20 compiler is required for incremental repair verification.');
    return;
  }
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-incremental-change-'));
  try {
    const paths = await buildSmallCertificate(directory, cxx);
    const seedFrontierBytes = await readFile(paths.seedFrontier);
    const boundaryRecords = frontierRecords(seedFrontierBytes);
    let selected = null;

    for (let index = 0; index < boundaryRecords.length; index += 1) {
      const rejectFrontier = join(directory, `candidate-${index}.reject.bin`);
      await writeFile(rejectFrontier, subsetFrontier(seedFrontierBytes, [boundaryRecords[index]]));
      const unaffected = join(directory, `candidate-${index}.unaffected.bin`);
      const affected = join(directory, `candidate-${index}.affected.bin`);
      const partition = JSON.parse(await run(paths.solver, [
        'partition',
        '--input-frontier', paths.inputFrontier,
        '--policy', paths.seedPolicy,
        '--reference-frontier', paths.seedFrontier,
        '--reject-frontier', rejectFrontier,
        '--unaffected', unaffected,
        '--affected', affected,
      ]));
      if (partition.unaffectedRoots === 0 || partition.affectedRoots === 0) continue;

      const directPolicy = join(directory, `candidate-${index}.direct.policy.bin`);
      const directFrontier = join(directory, `candidate-${index}.direct.frontier.bin`);
      const directRejected = join(directory, `candidate-${index}.direct.rejected.bin`);
      const direct = await runResult(paths.solver, [
        'extend',
        '--input-frontier', paths.inputFrontier,
        '--frontier-pieces', '6',
        '--maximum-states', '2000000',
        '--policy', directPolicy,
        '--frontier', directFrontier,
        '--reject-frontier', rejectFrontier,
        '--rejected', directRejected,
      ]);
      if (direct.code === 0) {
        selected = {
          rejectFrontier,
          partition,
          directPolicy,
          directFrontier,
        };
        break;
      }
      if (index >= 127) break;
    }

    assert.ok(selected, 'Expected a partially affected frontier state with a safe exact repair.');
    const repaired = await invokeRepair(directory, paths, selected.rejectFrontier, 'changed-repaired');
    assert.equal(repaired.report.status, 'safe');
    assert.ok(repaired.report.unaffectedRoots > 0);
    assert.ok(repaired.report.affectedExistingRoots > 0);
    assert.equal(repaired.report.freshRoots, 0);
    assert.equal(repaired.report.fallbackFullRegeneration, false);
    assert.equal(repaired.report.repair.format, 'connect4-chaos-prefix-sharded-certificate-v1');
    assert.equal(repaired.report.repair.shardWorkers, 2);
    assert.equal(
      repaired.report.unaffectedRoots + repaired.report.affectedExistingRoots,
      repaired.report.inputRoots,
    );
    assert.deepEqual(await readFile(repaired.outputPolicy), await readFile(selected.directPolicy));
    assert.deepEqual(await readFile(repaired.outputFrontier), await readFile(selected.directFrontier));
    assert.equal(repaired.report.replay.frontierStates, repaired.report.frontierStates);
    assert.equal(repaired.report.nativeVerification.frontierStates, repaired.report.frontierStates);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
