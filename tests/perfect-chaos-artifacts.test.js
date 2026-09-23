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
import { pythonCommand } from '../scripts/python-command.mjs';

const PYTHON = pythonCommand();

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const SCRIPT = join(ROOT, 'scripts', 'perfect-chaos-artifacts.py');

function run(args) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(PYTHON.command, [...PYTHON.args, SCRIPT, ...args], {
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

// Creating a symlink needs elevation or Developer Mode on Windows. When the
// platform refuses, the fixture cannot exist, so the test is skipped rather
// than reported as a failure of the writer.
async function symlinkOrSkip(context, target, linkPath, type) {
  try {
    await symlink(target, linkPath, type);
    return true;
  } catch (error) {
    if (error?.code === 'EPERM') {
      context.skip('creating symlinks is not permitted on this platform');
      return false;
    }
    throw error;
  }
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

test('artifact identity excludes only boundary-labelled incremental repair scratch space', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-artifacts-'));
  try {
    const scratch = join(directory, 'yellow', '.incremental-repair-10-12');
    await mkdir(scratch, { recursive: true });
    await writeFile(join(directory, 'proof.bin'), 'proof');
    await writeFile(join(scratch, 'affected-existing-input.bin'), 'scratch');

    await run(['write', '--directory', directory]);
    assert.equal(
      await readFile(join(directory, 'SHA256SUMS'), 'utf8'),
      `${digest('proof')}  proof.bin\n`,
    );

    await rm(join(directory, 'yellow'), { recursive: true, force: true });
    await run(['verify', '--directory', directory]);

    await writeFile(
      join(directory, 'SHA256SUMS'),
      `${digest('scratch')}  yellow/.incremental-repair-10-12/affected-existing-input.bin\n`
        + `${digest('proof')}  proof.bin\n`,
    );
    await run(['verify', '--directory', directory]);

    const nearMatch = join(directory, 'yellow', '.incremental-repair-ten-twelve');
    await mkdir(nearMatch, { recursive: true });
    await writeFile(join(nearMatch, 'unlisted.bin'), 'unlisted');
    await assert.rejects(
      run(['verify', '--directory', directory]),
      /unlisted file/,
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

test('artifact manifests may not be written through symlinked directories', async (context) => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-artifacts-'));
  const external = await mkdtemp(join(tmpdir(), 'connect4-chaos-artifacts-external-'));
  try {
    if (!await symlinkOrSkip(context, external, join(directory, 'manifest-dir'), 'dir')) return;
    await assert.rejects(
      run([
        'write', '--directory', directory,
        '--manifest', 'manifest-dir/SHA256SUMS',
      ]),
      /may not traverse symlinks/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
    await rm(external, { recursive: true, force: true });
  }
});

test('artifact manifest creation rejects symlinks', async (context) => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-artifacts-'));
  const external = await mkdtemp(join(tmpdir(), 'connect4-chaos-artifacts-external-'));
  try {
    await writeFile(join(external, 'payload.bin'), 'payload');
    if (!await symlinkOrSkip(context, join(external, 'payload.bin'), join(directory, 'linked.bin'))) return;
    await assert.rejects(
      run(['write', '--directory', directory]),
      /may not contain symlinks/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
    await rm(external, { recursive: true, force: true });
  }
});
