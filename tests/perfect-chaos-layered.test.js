import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const SOURCE = join(ROOT, 'native', 'perfect-chaos-layered.cpp');

// The layered solver must reproduce the monolithic solver's counts exactly.
// These constants are the recorded results of perfect-chaos-complete.cpp,
// embedded in the committed catalog and documented in docs/PERFECT_CHAOS.md.
const EXPECTED = [
  {
    rows: 4, columns: 4, connect: 3,
    states: 31523, wins: 24888, draws: 864, losses: 5771, rootValue: 1,
  },
  {
    rows: 4, columns: 4, connect: 4,
    states: 239230, wins: 97779, draws: 110159, losses: 31292, rootValue: 0,
  },
];

async function executable(path) {
  try {
    await access(path, fsConstants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function findCompiler() {
  if (process.env.CXX) return process.env.CXX;
  for (const candidate of ['/usr/bin/g++', '/usr/bin/clang++']) {
    if (await executable(candidate)) return candidate;
  }
  // Fall back to whatever the PATH offers, so a toolchain installed anywhere
  // other than /usr/bin lets this test run instead of skip.
  for (const candidate of ['g++', 'clang++']) {
    const probe = spawnSync(candidate, ['--version'], { encoding: 'utf8' });
    if (probe.status === 0) return candidate;
  }
  return null;
}

function run(command, args) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] });
    const stdout = [];
    const stderr = [];
    child.stdout.on('data', (chunk) => stdout.push(chunk));
    child.stderr.on('data', (chunk) => stderr.push(chunk));
    child.once('error', reject);
    child.once('close', (code) => resolvePromise({
      code,
      stdout: Buffer.concat(stdout).toString('utf8'),
      stderr: Buffer.concat(stderr).toString('utf8'),
    }));
  });
}

test('the layered solver reproduces the monolithic counts exactly', async (context) => {
  const compiler = await findCompiler();
  if (!compiler) {
    context.skip('no C++ compiler available');
    return;
  }
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-layered-'));
  try {
    const binary = join(directory, process.platform === 'win32' ? 'layered.exe' : 'layered');
    // -static keeps the WinLibs toolchain's iostreams from binding to an
    // older libstdc++ DLL found on PATH (Git for Windows ships one).
    const compiled = await run(compiler, [
      '-O2', '-std=c++20', '-static', '-o', binary, SOURCE,
    ]);
    assert.equal(compiled.code, 0, `compile failed: ${compiled.stderr.slice(0, 2000)}`);

    for (const expected of EXPECTED) {
      const out = join(directory, `${expected.rows}x${expected.columns}-c${expected.connect}`);
      const result = await run(binary, [
        '--rows', String(expected.rows),
        '--columns', String(expected.columns),
        '--connect', String(expected.connect),
        '--threads', '2',
        '--output', out,
      ]);
      assert.equal(result.code, 0, `solve failed: ${result.stderr.slice(0, 2000)}`);
      const line = result.stdout.split('\n').find((entry) => entry.startsWith('{'));
      assert.ok(line, 'no solution line emitted');
      const solution = JSON.parse(line);
      assert.equal(solution.format, 'connect4-chaos-exact-solution-layered-v1');
      for (const field of ['states', 'wins', 'draws', 'losses', 'rootValue']) {
        assert.equal(solution[field], expected[field],
          `${expected.rows}x${expected.columns} c${expected.connect} ${field}`);
      }
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
