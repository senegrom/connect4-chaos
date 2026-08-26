import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { join, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { pythonCommand } from '../scripts/python-command.mjs';

const PYTHON = pythonCommand();

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));

function runPython(source) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(PYTHON.command, [...PYTHON.args, '-c', source], {
      cwd: ROOT,
      env: { ...process.env, PYTHONPATH: join(ROOT, 'scripts') },
      stdio: ['ignore', 'pipe', 'pipe'],
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
      reject(new Error(`python3 exited with ${code ?? signal}.\n${errors || output}`));
    });
  });
}

test('Python policy-table encoding fails closed on conflicting actions', async () => {
  const source = String.raw`
from perfect_chaos_tables import POLICY_MAGIC, POLICY_RECORD_SIZE, encode_table

first = bytearray(POLICY_RECORD_SIZE)
first[16] = 6
first[17] = 7
first[18] = 1
first[19] = 0

second = bytearray(first)
second[18] = 2

encode_table(POLICY_MAGIC, 1, 2, POLICY_RECORD_SIZE, [bytes(first), bytes(second)])
`;
  await assert.rejects(
    runPython(source),
    /Conflicting Perfect Chaos policy actions/,
  );
});
