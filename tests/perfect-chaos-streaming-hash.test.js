import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { hashFile } from '../scripts/perfect-chaos-prefix.mjs';

// Artifacts run to hundreds of megabytes, so hashFile reads them in 1 MiB
// pieces instead of whole. What must hold is that the pieces add up to the
// digest of the whole file, at and around a piece's boundary too.
test('a streamed Perfect Chaos artifact hash equals the hash of the whole file', async (context) => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-streaming-hash-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const piece = 1024 * 1024;
  for (const size of [0, 1, piece - 1, piece, piece + 1, 3 * piece + 17]) {
    const bytes = Buffer.alloc(size);
    for (let index = 0; index < size; index += 1) bytes[index] = (index * 131 + (index >> 12)) & 0xff;
    const path = join(directory, `artifact-${size}.bin`);
    await writeFile(path, bytes);
    assert.deepEqual(await hashFile(path), {
      path: `artifact-${size}.bin`,
      bytes: size,
      sha256: createHash('sha256').update(bytes).digest('hex'),
    }, `${size} bytes`);
  }
});
