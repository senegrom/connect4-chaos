import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { verifyPerfectClassicCatalogParallel } from '../scripts/verify-perfect-classic-parallel.mjs';

function policyBytes({
  rows, columns, connect, role, handoffRemaining, rootValue, closureStates, records = [],
}) {
  const buffer = Buffer.alloc(24 + records.length * 10);
  buffer.write('C4VPOL1\0', 0, 'binary');
  buffer[8] = 1;
  buffer[9] = rows;
  buffer[10] = columns;
  buffer[11] = connect;
  buffer[12] = role;
  buffer[13] = handoffRemaining;
  buffer[14] = 10;
  buffer.writeInt8(rootValue, 15);
  buffer.writeUInt32LE(records.length, 16);
  buffer.writeUInt32LE(closureStates, 20);
  records.forEach((record, index) => {
    const offset = 24 + index * 10;
    buffer.writeBigUInt64LE(record.key, offset);
    buffer[offset + 8] = record.moveMask;
    buffer.writeInt8(record.outcome, offset + 9);
  });
  return buffer;
}

// Writes each policy beside a manifest and returns the manifest path. A policy
// may override the manifest's file entry to test what the runner accepts.
async function writeCatalog(directory, policies) {
  const entries = [];
  for (const [index, { file, records = [], ...metadata }] of policies.entries()) {
    const buffer = policyBytes({ ...metadata, records });
    const name = `policy-${index}.bin`;
    await writeFile(join(directory, name), buffer);
    entries.push({
      ...metadata,
      entryCount: records.length,
      file: file ?? `./${name}`,
      bytes: buffer.length,
      sha256: createHash('sha256').update(buffer).digest('hex'),
    });
  }
  const manifestPath = join(directory, 'manifest.json');
  await writeFile(manifestPath, `${JSON.stringify({
    format: 'connect4-perfect-classic-manifest-v1',
    policies: entries,
  }, null, 2)}\n`);
  return manifestPath;
}

async function writeRootValues(directory, connect, boards) {
  const path = join(directory, 'root-values.json');
  await writeFile(path, `${JSON.stringify({
    format: 'connect4-classic-root-values-v1',
    rules: { connect, gravity: true, chaosMode: false },
    boards,
  })}\n`);
  return path;
}

async function temporary(context, prefix) {
  const directory = await mkdtemp(join(tmpdir(), prefix));
  context.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}

// Empty 4x4 policies: the whole game is handed to the exact solver, so both
// roles replay to the published 4x4 draw.
const DRAWN_4X4 = [
  { rows: 4, columns: 4, connect: 4, role: 1, handoffRemaining: 16, rootValue: 0, closureStates: 1 },
  { rows: 4, columns: 4, connect: 4, role: 2, handoffRemaining: 16, rootValue: 0, closureStates: 3 },
];

// 1x3 connect 2 is a first-player win: the centre drop wins either end. This
// role-1 policy opens at the edge instead, so it proves only a draw - a sound
// lower bound that every replay accepts - while the exact role-2 policy proves
// the true -1 for the second player.
const WEAK_1X3 = [
  {
    rows: 1, columns: 3, connect: 2, role: 1, handoffRemaining: 2, rootValue: 0, closureStates: 4,
    records: [{ key: 0n, moveMask: 1, outcome: 0 }],
  },
  { rows: 1, columns: 3, connect: 2, role: 2, handoffRemaining: 2, rootValue: -1, closureStates: 3 },
];

test('parallel catalog verification independently replays each role', async (context) => {
  const directory = await temporary(context, 'perfect-classic-parallel-test-');
  const result = await verifyPerfectClassicCatalogParallel({
    reference: await writeCatalog(directory, DRAWN_4X4),
    workers: 2,
    verify_table_bits: 14,
  });
  assert.equal(result.policyCount, 2);
  assert.equal(result.boardCount, 1);
  assert.ok(result.exactNodes > 0);
  assert.deepEqual(result.replay.map((record) => record.role), [1, 2]);
  assert.ok(result.replay.every((record) => record.rootValue === 0));
});

test('parallel catalog verification rejects duplicate policy identities', async (context) => {
  const directory = await temporary(context, 'perfect-classic-parallel-duplicate-');
  await assert.rejects(
    verifyPerfectClassicCatalogParallel({
      reference: await writeCatalog(directory, [DRAWN_4X4[0], DRAWN_4X4[0]]),
      workers: 2,
    }),
    /Duplicate perfect classic policy/,
  );
});

test('two passing replays whose root values do not negate each other are rejected', async (context) => {
  const directory = await temporary(context, 'perfect-classic-parallel-pair-');
  await assert.rejects(
    verifyPerfectClassicCatalogParallel({
      reference: await writeCatalog(directory, WEAK_1X3),
      root_values: await writeRootValues(directory, 2, [{ rows: 1, columns: 3, value: 1 }]),
      workers: 2,
      verify_table_bits: 12,
    }),
    /1x3:c2 role values do not form a pair: role 1 proves 0, role 2 proves -1/,
  );
});

test('a proved value that contradicts the published root value is rejected', async (context) => {
  const directory = await temporary(context, 'perfect-classic-parallel-published-');
  await assert.rejects(
    verifyPerfectClassicCatalogParallel({
      reference: await writeCatalog(directory, DRAWN_4X4),
      root_values: await writeRootValues(directory, 4, [{ rows: 4, columns: 4, value: -1 }]),
      workers: 2,
      verify_table_bits: 14,
    }),
    /4x4:c4 proves 0 but the published root value is -1/,
  );
});

test('a board without a published root value is rejected', async (context) => {
  const directory = await temporary(context, 'perfect-classic-parallel-unpublished-');
  const exact = [
    { ...WEAK_1X3[0], handoffRemaining: 3, rootValue: 1, closureStates: 1, records: [] },
    WEAK_1X3[1],
  ];
  await assert.rejects(
    verifyPerfectClassicCatalogParallel({
      reference: await writeCatalog(directory, exact),
      workers: 2,
      verify_table_bits: 12,
    }),
    /1x3:c2 has no published root value/,
  );
});

test('a board missing one starting role is rejected', async (context) => {
  const directory = await temporary(context, 'perfect-classic-parallel-role-');
  await assert.rejects(
    verifyPerfectClassicCatalogParallel({
      reference: await writeCatalog(directory, [DRAWN_4X4[0]]),
      workers: 1,
      verify_table_bits: 14,
    }),
    /4x4:c4 must carry both starting-role policies/,
  );
});

test('manifest file entries that leave the catalog directory are rejected before any replay', async (context) => {
  const directory = await temporary(context, 'perfect-classic-parallel-escape-');
  for (const file of ['../outside.bin', './nested/policy.bin', 'policy-0.bin', './policy-0.bin.json',
    join(directory, 'policy-0.bin'), '.\\policy-0.bin']) {
    await assert.rejects(
      verifyPerfectClassicCatalogParallel({
        reference: await writeCatalog(directory, [{ ...DRAWN_4X4[0], file }, DRAWN_4X4[1]]),
        workers: 1,
      }),
      /entry 0 must name a \.\/<name>\.bin file beside the manifest/,
      file,
    );
  }
});
