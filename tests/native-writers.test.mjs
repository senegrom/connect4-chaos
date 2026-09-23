import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdtemp, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { buildNative, findCompiler, runProcess } from '../scripts/native-build.mjs';

// A certificate small enough to sit in the stream buffer reaches the disk only
// when the stream flushes on close, so a full disk shows up there and nowhere
// else. /dev/full fails every write with ENOSPC; it exists on Linux only.
async function prepare(context, name) {
  if (!existsSync('/dev/full')) {
    context.skip('/dev/full is Linux-only');
    return null;
  }
  if (!findCompiler()) {
    context.skip('no C++ compiler available');
    return null;
  }
  const source = fileURLToPath(new URL(`../native/${name}.cpp`, import.meta.url));
  const { binary } = await buildNative(source, { name });
  const directory = await mkdtemp(join(tmpdir(), `connect4-${name}-writer-`));
  context.after(() => rm(directory, { recursive: true, force: true }));
  return { binary, directory };
}

test('the classic policy generator fails when its policy cannot reach the disk', async (context) => {
  const built = await prepare(context, 'perfect-classic-policy');
  if (!built) return;
  const result = await runProcess(built.binary, ['generate', '--rows', '2', '--columns', '2',
    '--connect', '2', '--role', '1', '--handoff-remaining', '0', '--table-bits', '16',
    '--output', '/dev/full']);
  assert.equal(result.code, 1, result.stdout);
  assert.match(result.stderr, /could not write complete policy output/);
});

test('the complete solver fails when a certificate cannot reach the disk', async (context) => {
  const built = await prepare(context, 'perfect-chaos-complete');
  if (!built) return;
  const prefix = join(built.directory, 'certificate');
  await symlink('/dev/full', `${prefix}-role1.bin`);
  const result = await runProcess(built.binary, ['--rows', '2', '--columns', '2',
    '--connect', '2', '--emit-policy', prefix]);
  assert.equal(result.code, 1, result.stdout);
  assert.match(result.stderr, /could not write the complete policy/);
});
