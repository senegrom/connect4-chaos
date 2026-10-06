import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { copyFile, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { buildNative, findCompiler } from '../scripts/native-build.mjs';
import { parseArguments as parseClassic } from '../scripts/perfect-classic.mjs';
import {
  generatePerfectChaosComplete,
  mergePerfectChaosCompleteManifests,
  verifyPerfectChaosCompleteReference,
} from '../scripts/perfect-chaos-complete.mjs';
import { parseArguments as parseStrategy } from '../scripts/perfect-strategy.mjs';

const CLASSIC_POLICY = fileURLToPath(new URL('../scripts/perfect-classic-policy.mjs', import.meta.url));
const CLASSIC_GENERATOR = fileURLToPath(new URL('../scripts/perfect-classic-policy-generator.mjs', import.meta.url));
const CLASSIC_CATALOG = new URL('../data/perfect-classic/', import.meta.url);

async function temporary(context, name) {
  const directory = await mkdtemp(join(tmpdir(), `connect4-${name}-`));
  context.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}

const node = (script, ...args) => spawnSync(process.execPath, [script, ...args], { encoding: 'utf8', timeout: 120_000 });
// A generator's cached build, with the digests it was compiled under.
const cachedBuild = (name) => buildNative(fileURLToPath(new URL(`../native/${name}.cpp`, import.meta.url)), { name });

test('the classic solver and strategy scripts refuse options their command does not read', () => {
  assert.deepEqual(parseClassic(['solve', '--rows', '5', '--columns', '6']),
    { command: 'solve', rows: '5', columns: '6' });
  assert.throws(() => parseClassic(['solve', '--colums', '6']), /solve has no option --colums\./);
  assert.throws(() => parseClassic(['verify', '--rows', '5']), /verify has no option --rows\./);
  assert.throws(() => parseClassic(['sovle']), /Unknown command: sovle/);
  assert.deepEqual(parseStrategy(['build', '--oracle-book', '7x6.book', '--handoff-remaining', '24']),
    { command: 'build', oracle_book: '7x6.book', handoff_remaining: '24' });
  assert.deepEqual(parseStrategy(['verify', '--input', 'candidate.bin']), { command: 'verify', input: 'candidate.bin' });
  // A misspelling used to verify the committed strategy and exit 0.
  assert.throws(() => parseStrategy(['verify', '--inptu', 'candidate.bin']), /verify has no option --inptu\./);
  assert.throws(() => parseStrategy(['bulid']), /Usage:/);
});

// Both used to run only when a catalog was regenerated, where a break in the
// solver's output lines or the file naming would surface.
test('a generated complete Chaos board merges into a catalog the replay accepts', async (context) => {
  if (!findCompiler()) {
    context.skip('no C++ compiler available');
    return;
  }
  const directory = await temporary(context, 'complete-catalog');
  const boards = [];
  for (const [rows, columns] of [[2, 2], [2, 3]]) {
    boards.push(await generatePerfectChaosComplete({ rows, columns, connect: 2, output: join(directory, `${rows}x${columns}`) }));
  }
  const catalog = join(directory, 'catalog', 'manifest.json');
  const merged = await mergePerfectChaosCompleteManifests(
    boards.map((board) => join(board.output, 'manifest.json')), catalog);
  assert.deepEqual(merged.coverage, { boards: ['2x2-c2', '2x3-c2'], boardCount: 2, roleCount: 4 });
  assert.deepEqual(merged.policies.map((entry) => entry.file),
    ['./2x2-c2-role1.bin', './2x2-c2-role2.bin', './2x3-c2-role1.bin', './2x3-c2-role2.bin']);
  const verified = await verifyPerfectChaosCompleteReference(catalog);
  assert.equal(verified.policyCount, 4);
  const again = join(boards[0].output, 'manifest.json');
  await assert.rejects(mergePerfectChaosCompleteManifests([again, again], join(directory, 'twice', 'manifest.json')),
    /Duplicate Perfect Chaos policy 2x2:c2:r1/);
});

// The solver used to delete its checkpoints as it exited, before the replay:
// a run killed or out of memory there had to solve the whole board again.
test('complete Chaos generation keeps the solve until its manifest is written', async (context) => {
  if (!findCompiler()) {
    context.skip('no C++ compiler available');
    return;
  }
  const output = await temporary(context, 'complete-resume');
  const checkpoints = async () => (await readdir(output)).filter((name) => name.startsWith('solver-checkpoint')).sort();
  // A directory where the manifest goes fails the run after the replay.
  await mkdir(join(output, 'manifest.json'));
  await assert.rejects(generatePerfectChaosComplete({ rows: 2, columns: 3, connect: 2, output }), /EISDIR/);
  assert.deepEqual(await checkpoints(), ['solver-checkpoint.bitset', 'solver-checkpoint.round']);
  await rm(join(output, 'manifest.json'), { recursive: true });
  const { manifest } = await generatePerfectChaosComplete({ rows: 2, columns: 3, connect: 2, output });
  assert.equal(manifest.policies.length, 2);
  assert.deepEqual(await checkpoints(), []);
  // The manifest names the build that solved the board.
  const build = await cachedBuild('perfect-chaos-complete');
  assert.equal(manifest.sourceSha256, build.sourceSha256);
  assert.deepEqual(manifest.headersSha256, build.headersSha256);
  assert.deepEqual(Object.keys(manifest.headersSha256), ['atomic-load.hpp', 'checkpoint-io.hpp']);
});

test('a generated classic board records its search counters and every source of its generator', async (context) => {
  if (!findCompiler()) {
    context.skip('no C++ compiler available');
    return;
  }
  const output = await temporary(context, 'classic-generate');
  const generated = node(CLASSIC_GENERATOR, 'generate', '--rows', '3', '--columns', '3', '--connect', '3',
    '--handoff-remaining', '0', '--table-bits', '16', '--verify-table-bits', '14', '--output', output);
  assert.equal(generated.status, 0, generated.stderr);
  const manifest = JSON.parse(await readFile(join(output, 'manifest.json'), 'utf8'));
  // The manifest names the build that generated the policies, whose exact
  // search is a header shared with perfect-classic.cpp.
  const build = await cachedBuild('perfect-classic-policy');
  assert.equal(manifest.sourceSha256, build.sourceSha256);
  assert.deepEqual(manifest.headersSha256, build.headersSha256);
  assert.deepEqual(Object.keys(manifest.headersSha256), ['classic-exact.hpp']);
  assert.deepEqual(manifest.policies.map((entry) => entry.role), [1, 2]);
  for (const { generator } of manifest.policies) {
    // All five used to be printed as zeros.
    assert.ok(generator.nodes > 0 && generator.tableStores > 0 && generator.cutoffs > 0, JSON.stringify(generator));
    assert.ok(Number.isInteger(generator.tableHits) && Number.isInteger(generator.tableCollisions));
  }
});

test('classic policy manifests merge into a catalog the replay accepts, and refuse a duplicate', async (context) => {
  const directory = await temporary(context, 'classic-catalog');
  const committed = JSON.parse(await readFile(new URL('manifest.json', CLASSIC_CATALOG), 'utf8'));
  const inputs = [];
  for (const role of [1, 2]) {
    const entry = committed.policies.find((policy) => policy.rows === 4 && policy.columns === 4
      && policy.connect === 4 && policy.role === role);
    const folder = join(directory, `role${role}`);
    await mkdir(folder);
    await copyFile(new URL(entry.file, CLASSIC_CATALOG), join(folder, entry.file));
    await writeFile(join(folder, 'manifest.json'), JSON.stringify({ format: committed.format, policies: [entry] }));
    inputs.push(join(folder, 'manifest.json'));
  }
  const catalog = join(directory, 'catalog', 'manifest.json');
  const merged = node(CLASSIC_GENERATOR, 'merge-manifests', '--input', inputs[0], '--input', inputs[1], '--output', catalog);
  assert.equal(merged.status, 0, merged.stderr);
  const manifest = JSON.parse(await readFile(catalog, 'utf8'));
  assert.deepEqual(manifest.policies.map((entry) => [entry.role, entry.file]),
    [[1, './4x4-c4-role1.bin'], [2, './4x4-c4-role2.bin']]);
  const replayed = node(CLASSIC_POLICY, 'verify-reference', '--reference', catalog, '--verify-table-bits', '14');
  assert.equal(replayed.status, 0, replayed.stderr);
  const duplicate = node(CLASSIC_GENERATOR, 'merge-manifests', '--input', inputs[0], '--input', inputs[0],
    '--output', join(directory, 'twice.json'));
  assert.notEqual(duplicate.status, 0);
  assert.match(duplicate.stderr, /Duplicate perfect classic policy 4x4:c4:r1/);
});
