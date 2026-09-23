import assert from 'node:assert/strict';
import { copyFile, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  checkPerfectChaosCompleteRolePairs,
  verifyPerfectChaosCompleteReference,
} from '../scripts/perfect-chaos-complete.mjs';

const CATALOG = new URL('../data/perfect-chaos-complete/', import.meta.url);

function record(rows, columns, connect, role, replayedRootValue) {
  return { rows, columns, connect, role, replayedRootValue };
}

test('role pairs pin every board to one exact value', () => {
  checkPerfectChaosCompleteRolePairs([
    record(4, 4, 3, 1, 1), record(4, 4, 3, 2, -1),
    record(4, 4, 4, 2, 0), record(4, 4, 4, 1, 0),
  ]);
  // Each replay is a sound lower bound, so a weak role-1 policy that only
  // draws a won board passes its own replay; only the pair exposes it.
  assert.throws(
    () => checkPerfectChaosCompleteRolePairs([record(4, 4, 3, 1, 0), record(4, 4, 3, 2, -1)]),
    /4x4:c3 role values do not form a pair: role 1 proves 0, role 2 proves -1/,
  );
  assert.throws(
    () => checkPerfectChaosCompleteRolePairs([record(4, 5, 4, 1, 0)]),
    /4x5:c4 must carry both starting-role certificates/,
  );
  assert.throws(
    () => checkPerfectChaosCompleteRolePairs([record(4, 5, 4, 1, 0), record(4, 5, 4, 3, 0)]),
    /4x5:c4 must carry both starting-role certificates/,
  );
});

async function catalogOf(context, roles) {
  const directory = await mkdtemp(join(tmpdir(), 'perfect-chaos-complete-reference-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const committed = JSON.parse(await readFile(new URL('manifest.json', CATALOG), 'utf8'));
  const policies = committed.policies.filter((entry) => (
    entry.rows === 4 && entry.columns === 4 && entry.connect === 3 && roles.includes(entry.role)
  ));
  assert.equal(policies.length, roles.length);
  for (const entry of policies) {
    await copyFile(fileURLToPath(new URL(entry.file, CATALOG)), join(directory, entry.file));
  }
  const manifest = join(directory, 'manifest.json');
  await writeFile(manifest, `${JSON.stringify({ ...committed, policies }, null, 2)}\n`);
  return manifest;
}

test('verify-reference replays both roles of a committed board and accepts the pair', async (context) => {
  const verified = await verifyPerfectChaosCompleteReference(await catalogOf(context, [1, 2]));
  assert.deepEqual(verified.replay.map((entry) => entry.replayedRootValue), [1, -1]);
});

test('verify-reference refuses a catalog that drops one starting role', async (context) => {
  await assert.rejects(
    verifyPerfectChaosCompleteReference(await catalogOf(context, [1])),
    /4x4:c3 must carry both starting-role certificates/,
  );
});
