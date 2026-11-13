import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const MANIFEST_URL = new URL('../data/perfect-chaos-prefix/manifest.json', import.meta.url);
const DIRECTORY_URL = new URL('../data/perfect-chaos-prefix/', import.meta.url);
const SOURCE_URL = new URL('../native/perfect-chaos-prefix.cpp', import.meta.url);
const GENERATOR_SOURCE_URL = new URL(
  '../data/perfect-chaos-prefix/provenance/perfect-chaos-prefix-generator.cpp',
  import.meta.url,
);
const ORIGINAL_MANIFEST_URL = new URL(
  '../data/perfect-chaos-prefix/provenance/original-manifest.json',
  import.meta.url,
);

function digest(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

test('the committed Perfect Chaos prefix certificate reaches sixteen pieces', async () => {
  const manifest = JSON.parse(await readFile(MANIFEST_URL, 'utf8'));
  assert.equal(manifest.format, 'connect4-chaos-layered-prefix-manifest-v1');
  assert.deepEqual(manifest.boundaries, [8, 10, 12, 14, 16]);
  assert.equal(manifest.sourceSha256, digest(await readFile(GENERATOR_SOURCE_URL)));
  assert.equal(manifest.verificationSourceSha256, digest(await readFile(SOURCE_URL)));
  assert.equal(manifest.originalManifestSha256, digest(await readFile(ORIGINAL_MANIFEST_URL)));
  assert.equal(manifest.generatorRef, 'claude/connect4-chaos-ai-c1py3r');
  assert.match(manifest.generatorCommit, /^[0-9a-f]{40}$/);

  const expectedFinal = {
    red: { policyEntries: 326_031, frontierStates: 339_682, closureStates: 747_775 },
    yellow: {
      policyEntries: 1_059_068,
      frontierStates: 1_164_120,
      closureStates: 2_498_257,
    },
  };

  for (const role of ['red', 'yellow']) {
    const finalSegment = manifest.roles[role].replay.segments.at(-1);
    assert.equal(finalSegment.fromPieces, 14);
    assert.equal(finalSegment.frontierPieces, 16);
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

// verificationSourceSha256 is the digest of the solver's .cpp alone. A header
// of this repository that it included would be solver code the digest does
// not cover: an edit to it would change the verifier and leave the manifest
// as it was. Looking for `#include "..."` lines missed a header that a macro
// names, and a directive that follows a comment or runs on to the next line.
// So each line that holds a '#', or its other spellings '%:' and '??=', must
// be a plain include of a system header and nothing else. That leaves the
// solver no way to define a macro, and a '#' in a comment fails this too.
test('the pinned digest covers the whole prefix solver, which includes no repository header', async () => {
  const lines = (await readFile(SOURCE_URL, 'utf8')).split('\n');
  const systemInclude = /^#include <[a-z0-9_]+(?:\/[a-z0-9_]+)*(?:\.h)?>$/;
  assert.deepEqual(lines.filter((line) => /#|%:|\?\?=/.test(line) && !systemInclude.test(line)), []);
});
