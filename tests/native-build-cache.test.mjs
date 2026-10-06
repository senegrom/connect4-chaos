import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, stat, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

import { spawnSync } from 'node:child_process';

import { includedHeaders, publishBinary, underLock } from '../scripts/native-build.mjs';

// A compile killed while it held the lock - the harness kills test runs under
// memory pressure - held every later native build for fifteen minutes.
test('a build lock whose owner has exited is broken at once; a live owner is waited for', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-native-lock-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const lock = join(directory, 'solver.lock');
  const exited = spawnSync(process.execPath, ['-e', '']).pid;
  await writeFile(lock, String(exited));
  let built = 0;
  const started = Date.now();
  await underLock(lock, async () => false, async () => { built += 1; });
  assert.equal(built, 1);
  assert.ok(Date.now() - started < 5_000, 'the dead owner was not waited for');

  await writeFile(lock, String(process.pid));
  let ready = false;
  setTimeout(() => { ready = true; }, 600);
  await underLock(lock, async () => ready, async () => { built += 1; });
  assert.equal(built, 1, 'a live owner keeps its lock until the target exists');
});

// Windows reuses PIDs quickly: a dead owner's PID taken by another process
// held every later build for fifteen minutes.
test('an owner refreshes its lock as it builds, and a lock left unrefreshed is broken', { timeout: 10_000 }, async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-native-heartbeat-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const lock = join(directory, 'solver.lock');
  const timing = { heartbeatMs: 20, staleMs: 1_000 };
  let refreshed = 0;
  await underLock(lock, async () => false, async () => {
    const written = (await stat(lock)).mtimeMs;
    await sleep(200);
    refreshed = (await stat(lock)).mtimeMs - written;
  }, timing);
  assert.ok(refreshed > 0, 'the lock was not refreshed during the build');

  await writeFile(lock, String(process.pid));          // a live PID, as a reused one would be
  const old = new Date(Date.now() - 5_000);
  await utimes(lock, old, old);
  let built = 0;
  await underLock(lock, async () => false, async () => { built += 1; }, timing);
  assert.equal(built, 1, 'an unrefreshed lock was waited for');
});

test('a build that lost the race keeps the binary that won it', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-native-publish-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const binary = join(directory, 'solver');
  const denied = async () => { throw Object.assign(new Error('operation not permitted'), { code: 'EPERM' }); };
  // Renaming onto a binary another process is running fails on Windows.
  await assert.rejects(publishBinary(join(directory, 'partial'), binary, denied), /not permitted/);
  await writeFile(binary, 'won');
  await publishBinary(join(directory, 'partial'), binary, denied);
  assert.equal(await readFile(binary, 'utf8'), 'won');
});

// The build cache is keyed on these digests too: keyed on the .cpp alone, it
// kept serving a binary built before an edit to one of its headers.
test('a build identity covers the headers a source includes, however deep', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-native-headers-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const source = join(directory, 'solver.cpp');
  await writeFile(source, '#include <vector>\n#include "io.hpp"\n  #  include "io.hpp"\nint main() {}\n');
  await writeFile(join(directory, 'io.hpp'), '#pragma once\n#include "crc.hpp"\n');
  await writeFile(join(directory, 'crc.hpp'), 'inline int crc() { return 1; }\n');
  const before = await includedHeaders(source);
  assert.deepEqual(Object.keys(before), ['crc.hpp', 'io.hpp']);
  await writeFile(join(directory, 'crc.hpp'), 'inline int crc() { return 2; }\n');
  const after = await includedHeaders(source);
  assert.notEqual(after['crc.hpp'], before['crc.hpp']);
  assert.equal(after['io.hpp'], before['io.hpp']);
});

test('the solvers are keyed on the headers they share', async () => {
  const headers = async (name) => Object.keys(await includedHeaders(
    fileURLToPath(new URL(`../native/${name}.cpp`, import.meta.url))));
  assert.deepEqual(await headers('perfect-chaos-complete'), ['atomic-load.hpp', 'checkpoint-io.hpp']);
  for (const name of ['perfect-chaos-layered', 'perfect-chaos-paired']) {
    assert.deepEqual(await headers(name), ['atomic-load.hpp', 'chaos-layers.hpp', 'checkpoint-io.hpp'], name);
  }
  for (const name of ['perfect-classic', 'perfect-classic-policy']) {
    assert.deepEqual(await headers(name), ['classic-exact.hpp'], name);
  }
});
