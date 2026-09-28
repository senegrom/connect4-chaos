import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { spawnSync } from 'node:child_process';

import { includedHeaders, underLock } from '../scripts/native-build.mjs';

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

test('the complete solver is keyed on its checkpoint and atomic headers', async () => {
  const source = fileURLToPath(new URL('../native/perfect-chaos-complete.cpp', import.meta.url));
  assert.deepEqual(Object.keys(await includedHeaders(source)), ['atomic-load.hpp', 'checkpoint-io.hpp']);
});
