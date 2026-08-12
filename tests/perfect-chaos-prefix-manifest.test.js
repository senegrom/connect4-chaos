import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const MANIFEST_URL = new URL('../data/perfect-chaos-prefix/manifest.json', import.meta.url);
const DIRECTORY_URL = new URL('../data/perfect-chaos-prefix/', import.meta.url);
const SOURCE_URL = new URL('../native/perfect-chaos-prefix.cpp', import.meta.url);

function digest(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

test('the committed Perfect Chaos prefix certificate reaches fourteen pieces', async () => {
  const manifest = JSON.parse(await readFile(MANIFEST_URL, 'utf8'));
  assert.equal(manifest.format, 'connect4-chaos-layered-prefix-manifest-v1');
  assert.deepEqual(manifest.boundaries, [8, 10, 12, 14]);
  assert.equal(manifest.sourceSha256, digest(await readFile(SOURCE_URL)));

  const expectedFinal = {
    red: { policyEntries: 91_493, frontierStates: 104_251, closureStates: 219_861 },
    yellow: { policyEntries: 278_371, frontierStates: 334_185, closureStates: 689_361 },
  };

  for (const role of ['red', 'yellow']) {
    const finalSegment = manifest.roles[role].replay.segments.at(-1);
    assert.equal(finalSegment.fromPieces, 12);
    assert.equal(finalSegment.frontierPieces, 14);
    assert.equal(finalSegment.terminalDraws, 0);
    assert.deepEqual(
      {
        policyEntries: finalSegment.policyEntries,
        frontierStates: finalSegment.frontierStates,
        closureStates: finalSegment.closureStates,
      },
      expectedFinal[role],
    );

    for (const artifact of manifest.artifacts[role]) {
      const bytes = await readFile(new URL(`${role}/${artifact.path}`, DIRECTORY_URL));
      assert.equal(bytes.length, artifact.bytes, `${role}/${artifact.path} byte length`);
      assert.equal(digest(bytes), artifact.sha256, `${role}/${artifact.path} digest`);
    }
  }
});
