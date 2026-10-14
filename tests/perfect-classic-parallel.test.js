import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { replayPerfectClassicPolicy } from '../scripts/perfect-classic-policy.mjs';
import { verifyPerfectClassicCatalogParallel } from '../scripts/verify-perfect-classic-parallel.mjs';
import { decodePerfectClassicPolicy } from '../src/perfect-classic-policy.js';

const RUNNER = fileURLToPath(new URL('../scripts/verify-perfect-classic-parallel.mjs', import.meta.url));

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

test('the release replay rejects a policy that is wrong in any one respect', () => {
  // The committed catalogs only ever give the replay correct policies; each
  // change here breaks one of its checks and none before it.
  const [weak] = WEAK_1X3;
  const replay = (changes) => replayPerfectClassicPolicy(
    decodePerfectClassicPolicy(policyBytes({ ...weak, ...changes })), { exactTableBits: 12 });
  assert.equal(replay({}).rootValue, 0);
  for (const [changes, message] of [
    [{ records: [{ key: 0n, moveMask: 1, outcome: 1 }] }, /outcome mismatch at key 0: 1 instead of 0/],
    [{ rootValue: 1 }, /root value mismatch: 0 instead of 1/],
    [{ records: [] }, /missing reachable key 0/],
    [{ records: [...weak.records, { key: 5n, moveMask: 1, outcome: 0 }] }, /has 1 unreachable record/],
    [{ closureStates: 5 }, /closure mismatch: 4 instead of 5/],
    // Without a handoff the AI's second move comes from the policy too, and
    // at key 6 column 0 holds its first stone.
    [{ handoffRemaining: 0, records: [...weak.records, { key: 6n, moveMask: 1, outcome: 0 }] },
      /selects an illegal move/],
  ]) {
    assert.throws(() => replay(changes), message);
  }
});

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

// The release workflow made these checks inline before the runner's hours of
// replays; the runner now makes them first. The policy files are deleted, so
// any replay or copy that started would fail with a different message. A
// replay must prove exactly the value its entry declares, so values that do
// not pair, or contradict the published one, are refused here too: they used
// to be found only after every replay had run.
test('catalog checks that need no replay fail before any policy is read', async (context) => {
  const directory = await temporary(context, 'perfect-classic-parallel-preamble-');
  const standard = DRAWN_4X4.map((entry) => ({ ...entry, rows: 6, columns: 7 }));
  for (const [policies, rootValues, message] of [
    [[DRAWN_4X4[0]], [{ rows: 4, columns: 4, value: 0 }], /4x4:c4 must carry both starting-role policies/],
    [[DRAWN_4X4[0], { ...DRAWN_4X4[0], role: 3 }, DRAWN_4X4[1]], [{ rows: 4, columns: 4, value: 0 }],
      /4x4:c4 must carry both starting-role policies/],
    [DRAWN_4X4, [{ rows: 5, columns: 5, value: 0 }], /4x4:c4 has no published root value/],
    [standard, [{ rows: 6, columns: 7, value: 1 }], /6x7:c4 is played from the standard 6x7 strategy/],
    [[DRAWN_4X4[0], DRAWN_4X4[1], DRAWN_4X4[1]], [{ rows: 4, columns: 4, value: 0 }],
      /Duplicate perfect classic policy 4x4:c4:r2/],
    // Both sound lower bounds, so both replays would pass.
    [WEAK_1X3, [{ rows: 1, columns: 3, value: 1 }],
      /1x3:c2 role values do not form a pair: role 1 declares 0, role 2 declares -1/],
    [DRAWN_4X4, [{ rows: 4, columns: 4, value: -1 }], /4x4:c4 declares 0 but the published root value is -1/],
    [[{ ...DRAWN_4X4[0], rootValue: 2 }, DRAWN_4X4[1]], [{ rows: 4, columns: 4, value: 0 }], /entry 0 is invalid/],
  ]) {
    const reference = await writeCatalog(directory, policies);
    for (let index = 0; index < policies.length; index += 1) {
      await rm(join(directory, `policy-${index}.bin`));
    }
    await assert.rejects(verifyPerfectClassicCatalogParallel({
      reference, root_values: await writeRootValues(directory, policies[0].connect, rootValues), workers: 1,
    }), message);
  }
});

// The gate writes its receipt only after this command exits 0, and nothing
// else now stands between a refused catalog and a receipt: run the command
// itself, not just the function behind it.
test('the runner the release gate invokes exits non-zero on a refusal and zero on a pass', async (context) => {
  const directory = await temporary(context, 'perfect-classic-parallel-cli-');
  const run = async (name, policies) => {
    const folder = join(directory, name);
    await mkdir(folder);
    return spawnSync(process.execPath, [RUNNER, '--reference', await writeCatalog(folder, policies),
      '--workers', '1', '--verify-table-bits', '14'], { encoding: 'utf8', timeout: 120_000 });
  };
  const refused = await run('refused', [DRAWN_4X4[0]]);
  assert.notEqual(refused.status, 0);
  assert.equal(refused.stdout, '');
  assert.match(refused.stderr, /4x4:c4 must carry both starting-role policies/);
  const accepted = await run('accepted', DRAWN_4X4);
  assert.equal(accepted.status, 0, accepted.stderr);
  assert.equal(JSON.parse(accepted.stdout).policyCount, 2);
});

// Signal 0 only asks whether the process still runs.
function stillRunning(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error.code !== 'ESRCH';
  }
}

// A busy machine can take tens of seconds to start the two replays below, so
// this only ends a run that hangs, such as one where the runner waits for a
// replay that has already closed.
const STOP_TIMEOUT = 180_000;

// The first failure ends the run, and on Linux and macOS a child outlives its
// parent: a replay still running when another failed went on for hours after
// the runner had exited. This drives a copy of the runner, whose verifier is a
// stand-in beside it: role 1 fails once role 2 has started, and role 2 would
// sleep for a minute. The runner runs in this process, which stays the
// children's parent, so Windows, which ends a runner's children when it
// exits, cannot hide one left running.
test('a failed replay stops the others before the runner gives up', { timeout: STOP_TIMEOUT }, async (context) => {
  const root = await temporary(context, 'perfect-classic-parallel-stop-');
  const scripts = join(root, 'scripts');
  await mkdir(scripts);
  // The runner spawns whichever perfect-classic-policy.mjs sits beside it, so a
  // copy of it, with the two helpers it imports, spawns the stand-in below.
  for (const name of ['verify-perfect-classic-parallel.mjs', 'cli-options.mjs', 'entry-point.mjs']) {
    await copyFile(join(dirname(RUNNER), name), join(scripts, name));
  }
  // On Linux and macOS role 2 takes half a second to exit after SIGTERM, so if
  // the runner gave up without waiting for it to close, the check below would
  // find it still running. Windows ends it outright. Left alone, role 2 writes
  // role-2.slept when its minute is up, so a runner that waits for it instead
  // of stopping it fails on that file rather than on the timeout. Role 1 waits
  // for role 2 however long it takes to start: it gives up only once the test
  // has timed out, so that a hung run cannot leave it polling for ever.
  await writeFile(join(scripts, 'perfect-classic-policy.mjs'), [
    "import { existsSync, readFileSync, renameSync, writeFileSync } from 'node:fs';",
    "const started = new URL('../role-2.pid', import.meta.url);",
    "const reference = process.argv[process.argv.indexOf('--reference') + 1];",
    "const [entry] = JSON.parse(readFileSync(reference, 'utf8')).policies;",
    'if (entry.role === 2) {',
    "  process.on('SIGTERM', () => setTimeout(() => process.exit(143), 500));",
    "  writeFileSync(new URL('../role-2.part', import.meta.url), String(process.pid));",
    "  renameSync(new URL('../role-2.part', import.meta.url), started);",
    "  setTimeout(() => writeFileSync(new URL('../role-2.slept', import.meta.url), ''), 60_000);",
    '} else {',
    `  const deadline = Date.now() + ${STOP_TIMEOUT};`,
    '  const poll = setInterval(() => {',
    '    if (!existsSync(started) && Date.now() < deadline) return;',
    '    clearInterval(poll);',
    "    console.error('stand-in refusal');",
    '    process.exitCode = 1;',
    '  }, 10);',
    '}',
  ].join('\n'));
  const copied = await import(pathToFileURL(join(scripts, 'verify-perfect-classic-parallel.mjs')).href);
  const catalog = join(root, 'catalog');
  await mkdir(catalog);
  await assert.rejects(copied.verifyPerfectClassicCatalogParallel({
    reference: await writeCatalog(catalog, DRAWN_4X4),
    root_values: await writeRootValues(catalog, 4, [{ rows: 4, columns: 4, value: 0 }]),
    workers: 2,
  }), /Independent replay failed for 4x4:c4:r1\.\nstand-in refusal/);
  const pid = Number(await readFile(join(root, 'role-2.pid'), 'utf8'));
  const leftRunning = stillRunning(pid);
  // A child the runner left behind is killed here, so it does not sleep out its minute.
  if (leftRunning) process.kill(pid, 'SIGKILL');
  assert.equal(leftRunning, false, 'role 2 was still replaying when the runner gave up');
  assert.equal(existsSync(join(root, 'role-2.slept')), false,
    'the runner waited for role 2 to finish instead of stopping it');
});

test('an empty, missing or renamed policy list is refused', async (context) => {
  const directory = await temporary(context, 'perfect-classic-parallel-empty-');
  const format = 'connect4-perfect-classic-manifest-v1';
  const pair = DRAWN_4X4.map(({ records, ...entry }) => entry);
  for (const manifest of [{ format, policies: [] }, { format }, { format, catalog: pair }, { policies: pair }]) {
    const reference = join(directory, 'manifest.json');
    await writeFile(reference, JSON.stringify(manifest));
    await assert.rejects(verifyPerfectClassicCatalogParallel({ reference, workers: 1 }),
      /manifest format is invalid or empty/, JSON.stringify(manifest));
  }
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
