#!/usr/bin/env node

import { constants as fsConstants } from 'node:fs';
import { access, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const SOURCE = join(ROOT, 'native', 'perfect-chaos.cpp');
const EXPECTED = new Map([
  ['2x2-connect2', { value: 1, states: 6 }],
  ['3x3-connect3', { value: 0, states: 628 }],
  ['6x7-endgame-fixture', {
    value: 1,
    states: 2_585,
    action: { type: 'rotateCW' },
  }],
]);

async function executable(path) {
  if (!path) return false;
  try {
    await access(path, fsConstants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function findCompiler() {
  const explicit = process.env.CXX;
  if (explicit) return explicit;
  for (const candidate of ['/usr/bin/g++', '/usr/bin/clang++']) {
    if (await executable(candidate)) return candidate;
  }
  throw new Error('A C++20 compiler is required (set CXX, or install g++/clang++).');
}

function run(command, args, options = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, {
      cwd: ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
      ...options,
    });
    const stdout = [];
    const stderr = [];
    child.stdout?.on('data', (chunk) => stdout.push(chunk));
    child.stderr?.on('data', (chunk) => stderr.push(chunk));
    child.once('error', reject);
    child.once('close', (code, signal) => {
      const output = Buffer.concat(stdout).toString('utf8');
      const errorOutput = Buffer.concat(stderr).toString('utf8');
      if (code === 0) {
        resolvePromise({ stdout: output, stderr: errorOutput });
        return;
      }
      reject(new Error(
        `${command} exited with ${signal ? `signal ${signal}` : `code ${code}`}\n${errorOutput || output}`,
      ));
    });
  });
}

function validate(records) {
  if (records.length !== EXPECTED.size) {
    throw new Error(`Native verifier returned ${records.length} cases; expected ${EXPECTED.size}.`);
  }
  for (const record of records) {
    const expected = EXPECTED.get(record.name);
    if (!expected) throw new Error(`Unexpected native verification case: ${record.name}`);
    if (record.value !== expected.value || record.states !== expected.states) {
      throw new Error(`Native verification mismatch for ${record.name}: ${JSON.stringify(record)}`);
    }
    if (expected.action && JSON.stringify(record.action) !== JSON.stringify(expected.action)) {
      throw new Error(`Native action mismatch for ${record.name}: ${JSON.stringify(record.action)}`);
    }
  }
}

async function main() {
  const compiler = await findCompiler();
  const temporary = await mkdtemp(join(tmpdir(), 'connect4-perfect-chaos-'));
  const binary = join(temporary, 'perfect-chaos-native');
  try {
    await run(compiler, [
      '-std=c++20',
      '-O3',
      '-Wall',
      '-Wextra',
      '-Wpedantic',
      SOURCE,
      '-o',
      binary,
    ]);
    const { stdout } = await run(binary, ['verify']);
    const records = stdout.trim().split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
    validate(records);
    process.stdout.write(`${JSON.stringify({
      compiler,
      cases: records,
    }, null, 2)}\n`);
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

await main();
