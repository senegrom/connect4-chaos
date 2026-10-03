import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  checkPerfectChaosCompleteRolePairs,
  parseArguments,
  verifyPerfectChaosCompleteReference,
} from '../scripts/perfect-chaos-complete.mjs';

const CATALOG = new URL('../data/perfect-chaos-complete/', import.meta.url);

function record(rows, columns, connect, role, replayedRootValue) {
  return { rows, columns, connect, role, replayedRootValue };
}

test('an option its command does not read is refused, not ignored', () => {
  // A misspelt --reference used to verify the committed catalog and exit 0.
  assert.throws(() => parseArguments(['verify-reference', '--refrence', 'candidate.json']), /has no option --refrence/);
  assert.throws(() => parseArguments(['generate', '--threads', '4']), /has no option --threads/);
  assert.throws(() => parseArguments(['verify-referenc']), /Unknown command/);
  assert.deepEqual(parseArguments(['verify-reference', '--reference', 'candidate.json']),
    { command: 'verify-reference', reference: 'candidate.json' });
  assert.deepEqual(parseArguments(['merge-manifests', '--input', 'a.json', '--input', 'b.json', '--output', 'm.json']),
    { command: 'merge-manifests', inputs: ['a.json', 'b.json'], output: 'm.json' });
  assert.deepEqual(parseArguments(['generate', '--rows', '4', '--solver-threads', '16']),
    { command: 'generate', rows: '4', solver_threads: '16' });
  assert.deepEqual(parseArguments([]), { command: 'verify-reference' });
});

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

async function catalogOf(context, roles, { board = [4, 4, 3], mutate } = {}) {
  const directory = await mkdtemp(join(tmpdir(), 'perfect-chaos-complete-reference-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const committed = JSON.parse(await readFile(new URL('manifest.json', CATALOG), 'utf8'));
  const [rows, columns, connect] = board;
  const policies = [];
  for (const entry of committed.policies) {
    if (entry.rows !== rows || entry.columns !== columns || entry.connect !== connect
        || !roles.includes(entry.role)) continue;
    let bytes = await readFile(fileURLToPath(new URL(entry.file, CATALOG)));
    if (mutate && entry.role === 1) {
      // The changed certificate - changed in place, or returned anew when it
      // grows - with an entry that describes its bytes and header: only the
      // replay can tell.
      const changed = mutate(bytes);
      if (Buffer.isBuffer(changed)) bytes = changed;
      policies.push({
        ...entry, rootValue: bytes.readInt8(13), entryCount: bytes.readUInt32LE(16),
        closureStates: bytes.readUInt32LE(20),
        bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex'),
      });
    } else policies.push(entry);
    await writeFile(join(directory, entry.file), bytes);
  }
  assert.equal(policies.length, roles.length);
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

test('verify-reference rejects a certificate that is wrong in any one respect', async (context) => {
  // Records follow a 24-byte header and take 24 bytes each: the action at
  // byte 18, its column at 19, the stored value at 20. The header's byte 13
  // is the root value and bytes 20-23 the closure size.
  const record = (index) => 24 + index * 24;
  const cases = [
    [(bytes) => {
      assert.equal(bytes.readInt8(record(10) + 20), 1);
      bytes.writeInt8(0, record(10) + 20);
    }, /stored value 0 but the policy forces 1/],
    [(bytes) => {
      bytes[record(10) + 18] = bytes[record(10) + 18] === 1 ? 2 : 1;
      bytes[record(10) + 19] = 0;
    }, /missing a reachable position/],
    [(bytes) => bytes.writeInt8(0, 13), /replayed root value 1 but the header claims 0/],
    [(bytes) => bytes.writeUInt32LE(bytes.readUInt32LE(20) + 1, 20), /closure size 174 does not match the header's 175/],
    [(bytes) => {
      // One record more, for a position no game reaches: a lone stone at the
      // top of column 0. Bytes 16-19 of the header count the records.
      const surplus = Buffer.alloc(24);
      surplus.writeBigUInt64LE(1n << 3n, 0);
      surplus[16] = 4;
      surplus[17] = 4;
      const grown = Buffer.concat([bytes, surplus]);
      grown.writeUInt32LE(bytes.readUInt32LE(16) + 1, 16);
      return grown;
    }, /1 unreachable record\(s\)/],
  ];
  for (const [mutate, message] of cases) {
    await assert.rejects(verifyPerfectChaosCompleteReference(await catalogOf(context, [1, 2], { mutate })), message);
  }
  // 4x4 connect 4 is drawn. Its first drawn record relabelled a win is one
  // the opponent escapes by repeating the position.
  const relabel = (bytes) => {
    let index = 0;
    while (bytes.readInt8(record(index) + 20) !== 0) index += 1;
    bytes.writeInt8(1, record(index) + 20);
  };
  await assert.rejects(
    verifyPerfectChaosCompleteReference(await catalogOf(context, [1, 2], { board: [4, 4, 4], mutate: relabel })),
    /a win is claimed at \S+ but the line can repeat forever/,
  );
});
