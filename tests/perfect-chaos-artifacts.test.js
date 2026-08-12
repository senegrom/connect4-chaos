import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import {
  mkdtemp,
  mkdir,
  readFile,
  rm,
  symlink,
  writeFile,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const SCRIPT = join(ROOT, 'scripts', 'perfect-chaos-artifacts.py');

function run(args) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn('python3', [SCRIPT, ...args], {
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
      const errors = Buffer.concat(stderr).toString('utf8');
      if (code === 0) {
        resolvePromise(output);
        return;
      }
      reject(new Error(errors || output));
    });
  });
}

function digest(value) {
  return createHash('sha256').update(value).digest('hex');
}

test('artifact checksum manifests are relative, sorted, complete, and verified', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-artifacts-'));
  try {
    await mkdir(join(directory, 'nested'));
    await writeFile(join(directory, 'z.bin'), 'z');
    await writeFile(join(directory, 'nested', 'a.bin'), 'a');
    await run(['write', '--directory', directory]);
    assert.equal(
      await readFile(join(directory, 'SHA256SUMS'), 'utf8'),
      `${digest('a')}  nested/a.bin\n${digest('z')}  z.bin\n`,
    );
    await run(['verify', '--directory', directory]);
    await writeFile(join(directory, 'z.bin'), 'changed');
    await assert.rejects(
      run(['verify', '--directory', directory]),
      /Checksum mismatch/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('artifact verification rejects unlisted files and path traversal', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-artifacts-'));
  try {
    await writeFile(join(directory, 'safe.bin'), 'safe');
    await run(['write', '--directory', directory]);
    await writeFile(join(directory, 'extra.bin'), 'extra');
    await assert.rejects(
      run(['verify', '--directory', directory]),
      /unlisted file/,
    );
    await writeFile(
      join(directory, 'SHA256SUMS'),
      `${'0'.repeat(64)}  ../escape\n`,
    );
    await assert.rejects(
      run(['verify', '--directory', directory]),
      /Unsafe manifest entry/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('artifact manifest creation rejects symlinks', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-artifacts-'));
  const external = await mkdtemp(join(tmpdir(), 'connect4-chaos-artifacts-external-'));
  try {
    await writeFile(join(external, 'payload.bin'), 'payload');
    await symlink(join(external, 'payload.bin'), join(directory, 'linked.bin'));
    await assert.rejects(
      run(['write', '--directory', directory]),
      /may not contain symlinks/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
    await rm(external, { recursive: true, force: true });
  }
});
