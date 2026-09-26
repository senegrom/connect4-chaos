import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { includedHeaders } from '../scripts/native-build.mjs';

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
