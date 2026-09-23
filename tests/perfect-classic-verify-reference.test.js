import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

// The sequential verifier the parallel release gate runs once per policy.
const SCRIPT = fileURLToPath(new URL('../scripts/perfect-classic-policy.mjs', import.meta.url));

function emptyPolicy(role, closureStates) {
  const buffer = Buffer.alloc(24);
  buffer.write('C4VPOL1\0', 0, 'binary');
  buffer[8] = 1;
  buffer[9] = 4;
  buffer[10] = 4;
  buffer[11] = 4;
  buffer[12] = role;
  buffer[13] = 16;
  buffer[14] = 10;
  buffer.writeInt8(0, 15);
  buffer.writeUInt32LE(0, 16);
  buffer.writeUInt32LE(closureStates, 20);
  return buffer;
}

async function catalog(context, entries) {
  const directory = await mkdtemp(join(tmpdir(), 'perfect-classic-reference-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const policies = [];
  for (const { role, closureStates, file = `./role${role}.bin` } of entries) {
    const buffer = emptyPolicy(role, closureStates);
    await writeFile(join(directory, `role${role}.bin`), buffer);
    policies.push({
      rows: 4, columns: 4, connect: 4, role, handoffRemaining: 16, rootValue: 0,
      entryCount: 0, closureStates, file,
      bytes: buffer.length, sha256: createHash('sha256').update(buffer).digest('hex'),
    });
  }
  const manifest = join(directory, 'manifest.json');
  await writeFile(manifest, `${JSON.stringify({ format: 'connect4-perfect-classic-manifest-v1', policies })}\n`);
  return manifest;
}

function verifyReference(manifest) {
  return spawnSync(process.execPath, [SCRIPT, 'verify-reference', '--reference', manifest,
    '--verify-table-bits', '14'], { encoding: 'utf8', timeout: 60_000 });
}

test('sequential verify-reference replays a well-formed catalog', async (context) => {
  const result = verifyReference(await catalog(context, [
    { role: 1, closureStates: 1 },
    { role: 2, closureStates: 3 },
  ]));
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout).replay.map((entry) => entry.rootValue), [0, 0]);
});

test('sequential verify-reference refuses an empty catalog instead of passing it vacuously', async (context) => {
  const result = verifyReference(await catalog(context, []));
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /lists no policies/);
});

test('sequential verify-reference rejects duplicate policy identities', async (context) => {
  const result = verifyReference(await catalog(context, [
    { role: 1, closureStates: 1 },
    { role: 1, closureStates: 1 },
  ]));
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Duplicate perfect classic policy 4x4:c4:r1/);
});

test('sequential verify-reference reads policies only from beside the manifest', async (context) => {
  for (const file of ['../role1.bin', './nested/role1.bin', 'role1.bin']) {
    const result = verifyReference(await catalog(context, [{ role: 1, closureStates: 1, file }]));
    assert.notEqual(result.status, 0, file);
    assert.match(result.stderr, /must name a \.\/<name>\.bin file beside its manifest/, file);
  }
});

test('sequential verify-reference hashes the very bytes it decodes', () => {
  // A second read for the digest would let the file change between the check
  // and the replay; one buffer must serve both.
  const source = readFileSync(SCRIPT, 'utf8');
  const start = source.indexOf('async function verifyPolicyManifest(');
  const body = source.slice(start, source.indexOf('\n}\n', start));
  assert.ok(start >= 0 && body.length > 0);
  assert.equal(body.match(/await readFile\(/g)?.length, 2, 'one read of the manifest, one per policy');
  assert.doesNotMatch(body, /hashFile\(/);
  assert.match(body, /createHash\('sha256'\)\.update\(bytes\)/);
  assert.match(body, /decodePerfectClassicPolicy\(bytes, entry\)/);
});
