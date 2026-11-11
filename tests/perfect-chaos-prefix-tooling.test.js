import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, realpathSync } from 'node:fs';
import {
  copyFile, mkdir, mkdtemp, open, readdir, realpath, rename, rm, symlink, writeFile,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join } from 'node:path';
import test from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { buildNative, findCompiler, runProcess } from '../scripts/native-build.mjs';
import {
  createJournal,
  earlierRun,
  encodeFrontier,
  encodePolicy,
  generateReference,
  initializeRejections,
  journalKey,
  journaledSegment,
  overlapping,
  parseArguments,
  replaySegment,
  reproducesReference,
  verifyCommittedReference,
} from '../scripts/perfect-chaos-prefix.mjs';

const SCRIPT = fileURLToPath(new URL('../scripts/perfect-chaos-prefix.mjs', import.meta.url));
const SOURCE = fileURLToPath(new URL('../native/perfect-chaos-prefix.cpp', import.meta.url));
const REFERENCE = JSON.parse(readFileSync(
  new URL('../data/perfect-chaos-prefix/manifest.json', import.meta.url), 'utf8'));
const BUILD = Object.freeze({
  sourceSha256: 'a'.repeat(64), headersSha256: { 'io.hpp': 'd'.repeat(64) }, compiler: 'c++',
  compilerVersion: 'test 1', flags: ['-O3'],
});
const EMPTY = { mover: 0n, opponent: 0n, rows: 6, columns: 7, aiTurn: true };
const MANIFEST = JSON.stringify({ format: REFERENCE.format });
const MARK_FORMAT = 'connect4-chaos-layered-prefix-generating-v1';

async function temporary(context) {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-prefix-tooling-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}

/** The mark a generation keeps in `output`, which must exist, from before it
 * clears anything there until its manifest is in place. It names the
 * directory it was written for. */
async function markFor(output) {
  return JSON.stringify({ format: MARK_FORMAT, output: await realpath(output) });
}

/** A generation of `output` with no solver: it does all it does before the
 * solver is first run, and stops there as a kill would. */
function stopped(output, move) {
  return assert.rejects(generateReference(null, output, 16, 1, null, 1, 14, 2_000_000, 1, null, [], move),
    { name: 'TypeError', message: /'binary'/ });
}

/** Every flush of a file to the disk and every rename a generation is handed
 * `move` for, in order, each with what `output` holds at that moment. */
async function watch(context, output) {
  const steps = [];
  const holds = () => (existsSync(output) ? readdirSync(output).sort() : []);
  const probe = await open(SCRIPT);
  const handle = Object.getPrototypeOf(probe);
  await probe.close();
  const flush = handle.sync;
  context.mock.method(handle, 'sync', function sync() {
    steps.push(['flush', holds()]);
    return flush.call(this);
  });
  const move = async (from, to) => {
    steps.push([`${basename(from)} to ${basename(to)}`, holds()]);
    await rename(from, to);
  };
  return { steps, move };
}

/** A link to a directory: a junction on Windows, which needs no privilege,
 * and a symlink elsewhere. Null where neither can be made. */
async function linked(target, path) {
  try {
    await symlink(target, path, process.platform === 'win32' ? 'junction' : 'dir');
    return path;
  } catch (error) {
    if (!['EPERM', 'EACCES', 'ENOSYS', 'ENOTSUP'].includes(error?.code)) throw error;
    return null;
  }
}

test('a regeneration reproduces the reference when its certificates and summaries match', () => {
  // What the generator writes: no provenance, and the current source's hash.
  const regenerated = {
    format: REFERENCE.format,
    theorem: REFERENCE.theorem,
    board: REFERENCE.board,
    boundaries: REFERENCE.boundaries,
    sourceSha256: '0'.repeat(64),
    roles: structuredClone(REFERENCE.roles),
    artifacts: structuredClone(REFERENCE.artifacts),
  };
  assert.notDeepEqual(regenerated, REFERENCE, 'the whole manifests can never agree');
  assert.equal(reproducesReference(regenerated, REFERENCE), true);
  regenerated.artifacts.red[0].sha256 = 'f'.repeat(64);
  assert.equal(reproducesReference(regenerated, REFERENCE), false);
  regenerated.artifacts = structuredClone(REFERENCE.artifacts);
  regenerated.roles.yellow.rejected.at14 += 1;
  assert.equal(reproducesReference(regenerated, REFERENCE), false);
});

test('an option its command does not read is refused, not ignored', () => {
  // A misspelt --reference used to verify the committed catalog and exit 0.
  assert.throws(() => parseArguments(['verify-reference', '--refrence', 'candidate.json']), /has no option --refrence/);
  assert.throws(() => parseArguments(['verify', '--reference', 'candidate.json']), /has no option --reference/);
  assert.throws(() => parseArguments(['generat']), /Unknown command/);
  // The generation command docs/CHAOS_BOUNDED_PROOF.md gives.
  assert.deepEqual(parseArguments(['generate', '--frontier-pieces', '16', '--seed-rejections', 'seeds',
    '--shards', '8', '--shard-from-pieces', '14', '--output', 'out']), {
    command: 'generate', frontier_pieces: '16', seed_rejections: 'seeds', shards: '8',
    shard_from_pieces: '14', output: 'out',
  });
  assert.deepEqual(parseArguments([]), { command: 'verify' });
});

test('an option given no value, or an empty one, is refused before anything is built', () => {
  // `--output "$OUT"` with OUT unset named the working directory, which the
  // generation set about deleting, and an empty --seed-rejections ran
  // unseeded. A bare option parses as true, even one followed by another.
  for (const [argv, option] of [
    [['generate', '--output'], 'output'],
    [['generate', '--output', ''], 'output'],
    [['generate', '--output', '--frontier-pieces', '16'], 'output'],
    [['generate', '--seed-rejections', ''], 'seed-rejections'],
    [['generate', '--journal'], 'journal'],
    [['generate', '--shards', ''], 'shards'],
    [['reproduce-reference', '--reference', ''], 'reference'],
    [['reproduce-reference', '--seed-rejections'], 'seed-rejections'],
    [['verify-reference', '--reference'], 'reference'],
  ]) {
    assert.throws(() => parseArguments(argv), { name: 'RangeError', message: `--${option} requires a value.` },
      argv.join(' '));
  }
});

test('generation refuses an output that would delete its seeds or a certificate', async (context) => {
  const directory = await temporary(context);
  const seeds = join(directory, 'seeds');
  const certificate = join(directory, 'certificate');
  const committed = fileURLToPath(new URL('../data/perfect-chaos-prefix', import.meta.url));
  assert.equal(overlapping(seeds, join(directory, 'seedsmore')), false);
  // The seeds, inside them, around them, the committed certificate, and the
  // certificate that reproduce-reference is checking.
  for (const [output, seed, keep] of [[seeds, seeds, []], [join(seeds, 'red'), seeds, []],
    [directory, seeds, []], [committed, null, []], [join(certificate, 'out'), null, [certificate]]]) {
    await assert.rejects(generateReference(null, output, 8, 1, seed, 1, 14, 2_000_000, 1, null, keep), /overlaps/,
      output);
  }
  assert.ok(existsSync(join(committed, 'manifest.json')));
});

// The output used to be deleted whole: `--output generated` took every other
// run's journal and certificate with it, and `--output ""` the working
// directory. None of this runs the solver; every refusal comes first.
test('generation deletes nothing in its output but an earlier run of it', async (context) => {
  const directory = await temporary(context);
  const write = async (path, text = '') => {
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, text);
  };
  // An earlier run of the output: killed before its manifest, then finished.
  const output = join(directory, 'perfect-chaos-prefix-16');
  await write(join(output, 'red', '0-8.policy.bin'));
  await write(join(output, 'yellow', 'reject-8.bin'));
  await write(join(output, 'generating.json'), await markFor(output));
  assert.deepEqual((await earlierRun(output)).sort(), ['generating.json', 'red', 'yellow']);
  await write(join(output, 'manifest.json'), MANIFEST);
  await rm(join(output, 'generating.json'));
  assert.deepEqual((await earlierRun(output)).sort(), ['manifest.json', 'red', 'yellow']);
  assert.deepEqual(await earlierRun(join(directory, 'new')), []);

  // A manifest or a mark says that a generation made the role directories
  // beside it, and nothing about anything else there. One entry that no
  // generation leaves refuses the output, and nothing in it is deleted.
  const beside = await temporary(context);
  for (const [evidence, foreign] of [['manifest.json', 'README.md'], ['generating.json', 'notes.txt']]) {
    const mixed = join(beside, foreign);
    await write(join(mixed, 'red', 'reject-8.bin'));
    await write(join(mixed, evidence), evidence === 'manifest.json' ? MANIFEST : await markFor(mixed));
    assert.deepEqual((await earlierRun(mixed)).sort(), [evidence, 'red']);
    await write(join(mixed, foreign));
    await assert.rejects(generateReference(null, mixed, 8, 1, null, 1, 14, 2_000_000, 1, null),
      { name: 'RangeError', message: new RegExp(`holds ${foreign}, which no generation leaves in its output`) });
    assert.deepEqual((await readdir(mixed)).sort(), [foreign, evidence, 'red'].sort());
    assert.deepEqual(await readdir(join(mixed, 'red')), ['reject-8.bin']);
  }
  // So the committed certificate, with its README.md and provenance/, is
  // refused here as well as by the overlap guard.
  await assert.rejects(earlierRun(fileURLToPath(new URL('../data/perfect-chaos-prefix', import.meta.url))),
    /holds (README\.md|provenance), which no generation leaves in its output/);

  // The directory that holds it, its journal and something no run wrote.
  const journal = join(directory, 'perfect-chaos-prefix-16.journal', 'a'.repeat(64), 'meta.json');
  const notes = join(directory, 'notes', 'plan.txt');
  await write(journal, '{}');
  await write(notes);
  await assert.rejects(generateReference(null, directory, 8, 1, null, 1, 14, 2_000_000, 1, null),
    /holds \S+, which no generation leaves in its output/);
  for (const path of [journal, notes, join(output, 'manifest.json'), join(output, 'red', '0-8.policy.bin')]) {
    assert.ok(existsSync(path), path);
  }

  // The output itself holds only that run, so it is accepted, and the run is
  // deleted before the null build is read, leaving the new run's mark. Kept,
  // its manifest would stand until the new one is written, and a deeper run's
  // last boundaries would sit beside the new certificate. Its journal and the
  // notes beside it stay.
  await assert.rejects(generateReference(null, output, 8, 1, null, 1, 14, 2_000_000, 1, null),
    { name: 'TypeError', message: /'binary'/ });
  assert.deepEqual(await readdir(output), ['generating.json']);
  for (const path of [journal, notes]) assert.ok(existsSync(path), path);

  // A manifest another generator wrote, and a file where the output goes.
  const other = join(directory, 'complete');
  await write(join(other, 'manifest.json'), JSON.stringify({ format: 'connect4-perfect-chaos-complete-manifest-v1' }));
  await assert.rejects(earlierRun(other), /holds a manifest\.json that is not a prefix certificate's/);
  await assert.rejects(earlierRun(notes), /is a file/);

  // The working directory, which '' names, its parents and generated/ are
  // refused whatever they hold, before anything in them is read.
  for (const path of ['', process.cwd(), dirname(process.cwd())]) {
    await assert.rejects(earlierRun(path), /is the working directory or holds it/, path);
  }
  await assert.rejects(earlierRun(fileURLToPath(new URL('../generated', import.meta.url))),
    /is generated\/ itself, where other runs keep their journals and certificates/);
});

// Any red/ and yellow/ used to pass, unread, for a run killed before its
// manifest, and the generation deleted both trees. A seed set kept apart from
// any certificate is exactly that: rejection tables under red/ and yellow/.
test('generation refuses role directories that no generation marked as its own', async (context) => {
  const directory = await temporary(context);
  const committed = fileURLToPath(new URL('../data/perfect-chaos-prefix', import.meta.url));
  const seeds = join(directory, 'perfect-chaos-prefix-seeds-18');
  for (const role of ['red', 'yellow']) {
    await mkdir(join(seeds, role), { recursive: true });
    await copyFile(join(committed, role, 'reject-8.bin'), join(seeds, role, 'reject-8.bin'));
  }
  const kept = async (label) => {
    assert.deepEqual((await readdir(seeds)).sort(), ['red', 'yellow'], label);
    for (const role of ['red', 'yellow']) assert.deepEqual(await readdir(join(seeds, role)), ['reject-8.bin'], label);
  };
  const unmarked =
    /holds red and yellow with neither the manifest\.json of a finished generation nor the generating\.json of an/;
  await assert.rejects(earlierRun(seeds), unmarked);
  // The documented generation, taken on to 18 pieces, with the seed set as
  // its output: the seeds it reads are fine, and the null build is never read.
  await assert.rejects(generateReference(null, seeds, 18, 1, committed, 8, 14, 2_000_000, 1, null), unmarked);
  await kept('generateReference');
  // They may be a run from before generations marked their outputs, so the
  // refusal says what to do with one: deleting them leaves its journal.
  await assert.rejects(earlierRun(seeds), (error) => error.message.endsWith('so a generation will not clear it. If an'
    + ' earlier generation of this output left them, delete them and run again: that keeps its journal, and with it'
    + ' the segments it finished.'));
  // One role directory, even an empty one, is no more a generation's.
  const redOnly = join(directory, 'red-only');
  await mkdir(join(redOnly, 'red'), { recursive: true });
  await assert.rejects(earlierRun(redOnly), /holds red with neither the manifest\.json/);
  // Nor does a file say so by a mark's or a manifest's name alone, or a mark
  // that names no directory, or what a kill leaves of either while it is
  // written, whole though it may be.
  for (const [name, text, refusal] of [
    ['generating.json', '', /holds a generating\.json that is not a prefix generation's/],
    ['generating.json', JSON.stringify(REFERENCE), /holds a generating\.json that is not a prefix generation's/],
    ['generating.json', JSON.stringify({ format: MARK_FORMAT }), /holds a generating\.json that is not a prefix/],
    ['manifest.json', '', /holds a manifest\.json that is not a prefix certificate's/],
    ['manifest.json', await markFor(seeds), /holds a manifest\.json that is not a prefix certificate's/],
    ['generating.json.part', await markFor(seeds), unmarked],
    ['manifest.json.part', MANIFEST, unmarked],
  ]) {
    await writeFile(join(seeds, name), text);
    await assert.rejects(earlierRun(seeds), refusal, name);
    await rm(join(seeds, name));
  }
  await kept('earlierRun');
});

// A mark vouched for whatever directory it lay in. A copy of an unfinished
// output, kept for its rejection tables, carries that output's mark, and a
// generation given the copy as its output cleared it.
test('a mark vouches only for the directory it was written for', async (context) => {
  const directory = await temporary(context);
  const output = join(directory, 'perfect-chaos-prefix-16');
  await stopped(output);
  await mkdir(join(output, 'red'));
  await writeFile(join(output, 'red', 'reject-8.bin'), '');
  const copy = join(directory, 'seeds-18');
  await mkdir(join(copy, 'red'), { recursive: true });
  for (const name of ['generating.json', join('red', 'reject-8.bin')]) {
    await copyFile(join(output, name), join(copy, name));
  }
  await assert.rejects(generateReference(null, copy, 16, 1, null, 1, 14, 2_000_000, 1, null), { name: 'RangeError',
    message: /seeds-18 holds the generating\.json of a generation into .+perfect-chaos-prefix-16, as a copy of that/ });
  assert.deepEqual((await readdir(copy)).sort(), ['generating.json', 'red']);
  assert.deepEqual(await readdir(join(copy, 'red')), ['reject-8.bin']);
  // The output itself is its mark's, and a run of it clears it.
  assert.deepEqual((await earlierRun(output)).sort(), ['generating.json', 'red']);
  // Moved, it is no longer where its mark says, and cannot be told from a
  // copy: the refusal says what to do if it is the output after all.
  const moved = join(directory, 'moved');
  await rename(output, moved);
  await assert.rejects(earlierRun(moved), (error) => error.message.endsWith('so a generation will not clear it. If it'
    + ' is that output, moved, delete what it holds and run again, with --journal naming the journal that generation'
    + ' kept to reuse the segments it finished.'));
});

// The mark is in the output, whole, before anything there is cleared, and
// stays until the manifest, so a run stopped part-way - here by the null
// build, as a kill would stop a real one later - leaves an output the same
// command clears and generates into again. It used to be written in place:
// a kill part-way through left an empty generating.json, which every later
// run refused.
test('a generation marks its output before it clears it, and leaves an unfinished run its mark', async (context) => {
  const output = join(await temporary(context), 'perfect-chaos-prefix-16');
  const { steps, move } = await watch(context, output);
  // A new output: the mark is written beside its place, flushed, and renamed
  // into it.
  await stopped(output, move);
  assert.deepEqual(steps.splice(0), [['flush', ['generating.json.part']],
    ['generating.json.part to generating.json', ['generating.json.part']]]);
  assert.deepEqual(await readdir(output), ['generating.json']);
  const mark = readFileSync(join(output, 'generating.json'), 'utf8');
  const { startedAt, ...named } = JSON.parse(mark);
  assert.deepEqual(named, { format: MARK_FORMAT, output: await realpath(output) });
  assert.ok(Date.parse(startedAt) <= Date.now(), startedAt);

  // What the run had written when it was killed, seeded rejections included.
  // Run again, it finds its own mark and leaves it as it is: nothing is
  // written that a second kill could cut short.
  const write = async (role, name) => {
    await mkdir(join(output, role), { recursive: true });
    await writeFile(join(output, role, name), '');
  };
  await write('red', '0-8.policy.bin');
  await write('red', 'reject-8.bin');
  await write('yellow', 'reject-8.bin');
  assert.deepEqual((await earlierRun(output)).sort(), ['generating.json', 'red', 'yellow']);
  await stopped(output, move);
  assert.deepEqual(steps, []);
  assert.equal(readFileSync(join(output, 'generating.json'), 'utf8'), mark);
  assert.deepEqual(await readdir(output), ['generating.json']);

  // A finished run generated again: every file of it is still there when the
  // new run's mark is flushed, and when it is renamed into place. Cleared
  // first, the run would be half gone with nothing to say whose the rest is.
  await rm(join(output, 'generating.json'));
  await writeFile(join(output, 'manifest.json'), MANIFEST);
  await write('red', '0-8.policy.bin');
  await write('yellow', 'reject-8.bin');
  await stopped(output, move);
  const finished = ['generating.json.part', 'manifest.json', 'red', 'yellow'];
  assert.deepEqual(steps, [['flush', finished], ['generating.json.part to generating.json', finished]]);
  assert.deepEqual(await readdir(output), ['generating.json']);
});

// Each state below is one a kill leaves: the mark and the manifest are each
// renamed into place whole, so the worst a kill does to either is leave the
// .part it was being written to.
test('whatever a stopped generation leaves in its output, the same command clears it and runs', async (context) => {
  const directory = await temporary(context);
  const states = [
    ['while its first mark was written', ['generating.json.part']],
    ['while its mark was written over a finished run', ['generating.json.part', 'manifest.json', 'red', 'yellow']],
    ['while it cleared a finished run', ['generating.json', 'yellow']],
    ['while it generated', ['generating.json', 'red']],
    ['while its manifest was written', ['generating.json', 'manifest.json.part', 'red', 'yellow']],
    ['before it removed its mark', ['generating.json', 'manifest.json', 'red', 'yellow']],
    ['once it had finished', ['manifest.json', 'red', 'yellow']],
  ];
  for (const [when, names] of states) {
    const output = join(directory, `stopped ${when}`);
    await mkdir(output);
    for (const name of names) {
      if (name === 'red' || name === 'yellow') {
        await mkdir(join(output, name));
        await writeFile(join(output, name, 'reject-8.bin'), '');
      } else {
        // A .part holds whatever the kill let through: here, half a line.
        const whole = name.startsWith('generating.json') ? await markFor(output) : MANIFEST;
        await writeFile(join(output, name), name.endsWith('.part') ? whole.slice(0, 20) : whole);
      }
    }
    assert.deepEqual((await earlierRun(output)).sort(), names, when);
    await stopped(output);
    // Cleared, and marked as this output's for the run after.
    assert.deepEqual(await earlierRun(output), ['generating.json'], when);
  }
  // Stopped at the rename itself, a run leaves the first of those states.
  const output = join(directory, 'stopped at the rename');
  const halt = async () => { throw new Error('stopped at the rename'); };
  await assert.rejects(generateReference(null, output, 16, 1, null, 1, 14, 2_000_000, 1, null, [], halt),
    /stopped at the rename/);
  assert.deepEqual(await readdir(output), ['generating.json.part']);
  await stopped(output);
  assert.deepEqual(await readdir(output), ['generating.json']);
});

// The overlap guard compared spellings. An output that was a junction or a
// symlink to the seeds in use passed it, and the seeds, a finished
// certificate as the committed one is, were deleted as an earlier run.
test('an output that reaches its seeds or a kept certificate through a link is refused', async (context) => {
  const directory = await temporary(context);
  const certificate = async (name) => {
    for (const role of ['red', 'yellow']) {
      await mkdir(join(directory, name, role), { recursive: true });
      await writeFile(join(directory, name, role, 'reject-8.bin'), encodeFrontier(role === 'red' ? 1 : 2, 8, []));
    }
    await writeFile(join(directory, name, 'manifest.json'), MANIFEST);
    return join(directory, name);
  };
  const seeds = await certificate('seeds');
  const alias = await linked(seeds, join(directory, 'alias'));
  if (!alias) {
    context.skip('neither a junction nor a symlink can be made here');
    return;
  }
  const whole = async (path) => {
    assert.deepEqual((await readdir(path)).sort(), ['manifest.json', 'red', 'yellow'], path);
    for (const role of ['red', 'yellow']) assert.deepEqual(await readdir(join(path, role)), ['reject-8.bin'], path);
  };
  // The link as the output, a directory under it, and the seeds named
  // through the link with the directory itself as the output.
  for (const [output, seed] of [[alias, seeds], [join(alias, 'red'), seeds], [join(alias, 'new'), seeds],
    [seeds, alias]]) {
    await assert.rejects(generateReference(null, output, 10, 1, seed, 1, 14, 2_000_000, 1, null),
      { name: 'RangeError', message: /overlaps .+, which generating would delete\./ }, `${output} seeded from ${seed}`);
    await whole(seeds);
  }
  // The certificate that reproduce-reference is checking, reached the same way.
  const kept = await certificate('kept');
  const other = await linked(kept, join(directory, 'other'));
  await assert.rejects(generateReference(null, join(other, 'out'), 8, 1, null, 1, 14, 2_000_000, 1, null, [kept]),
    /overlaps/);
  await whole(kept);
  // A link to a directory of its own is an output like any other, and the
  // same output however it is reached: its mark names the directory.
  const own = join(directory, 'own');
  await mkdir(own);
  const reached = await linked(own, join(directory, 'reached'));
  await stopped(reached);
  assert.equal(JSON.parse(readFileSync(join(own, 'generating.json'), 'utf8')).output, await realpath(own));
  assert.deepEqual(await earlierRun(own), ['generating.json']);
  assert.deepEqual(await earlierRun(reached), ['generating.json']);

  // The native realpath fails on some drives, a RAM disk on Windows among
  // them. Node's own still follows the links there, as scripts/entry-point.mjs
  // has it do, though it leaves letter case as it was spelt: a generation is
  // not stopped at its first check, and the mark is still its output's.
  context.mock.method(realpathSync, 'native', () => {
    throw Object.assign(new Error('EISDIR: illegal operation on a directory, realpath'), { code: 'EISDIR' });
  });
  await assert.rejects(generateReference(null, alias, 10, 1, seeds, 1, 14, 2_000_000, 1, null), /overlaps/);
  await whole(seeds);
  assert.deepEqual(await earlierRun(reached), ['generating.json']);
  if (process.platform === 'win32') assert.deepEqual(await earlierRun(own.toUpperCase()), ['generating.json']);
});

// The working directory and generated/ are refused whatever they hold, and
// however they are reached. A copy of the script in a directory of its own
// has that directory's generated/, and is run from a working directory
// beside it, so no link made here leads into the checkout.
test('the working directory and generated/ are refused through a link as well', async (context) => {
  const root = await temporary(context);
  for (const name of ['scripts', 'generated', 'work', 'links']) await mkdir(join(root, name));
  for (const name of ['perfect-chaos-prefix.mjs', 'cli-options.mjs', 'entry-point.mjs', 'native-build.mjs',
    'option-values.mjs']) {
    await copyFile(join(dirname(SCRIPT), name), join(root, 'scripts', name));
  }
  const outputs = [];
  for (const name of ['work', 'generated']) outputs.push(await linked(join(root, name), join(root, 'links', name)));
  if (outputs.includes(null)) {
    context.skip('neither a junction nor a symlink can be made here');
    return;
  }
  // The copy's earlierRun, asked from its working directory about each link
  // and about a new directory, which it accepts.
  const asked = spawnSync(process.execPath, ['--input-type=module', '-e', `
    const { earlierRun } = await import(process.argv[1]);
    for (const output of process.argv.slice(2)) {
      console.log(await earlierRun(output).then((names) => JSON.stringify(names), (error) => error.message));
    }`, pathToFileURL(join(root, 'scripts', 'perfect-chaos-prefix.mjs')).href, ...outputs, join(root, 'links', 'new')],
  { cwd: join(root, 'work'), encoding: 'utf8', timeout: 120_000 });
  assert.equal(asked.status, 0, asked.stderr);
  const [work, generated, fresh] = asked.stdout.trim().split(/\r?\n/);
  assert.match(work, /is the working directory or holds it/);
  assert.match(generated, /is generated\/ itself, where other runs keep their journals and certificates/);
  assert.equal(fresh, '[]');
});

// The command line used to check a generation's options only once it had
// built the solver, and reached generateReference's refusals only once it had
// also made the journal directory beside the output, which a refused --output
// left behind. Seeds it could not use were refused later still, as each
// role's generation started. With no compiler to build with, a refusal made
// after the build would be the compiler's.
test('the command line refuses a generation before it builds the solver or makes anything', async (context) => {
  const directory = await temporary(context);
  const taken = join(directory, 'taken');
  await mkdir(taken);
  await writeFile(join(taken, 'plan.txt'), '');
  // A certificate to reproduce, which must be kept even when the seeds come
  // from elsewhere, and seeds with no yellow rejections.
  const certificate = join(directory, 'certificate');
  await mkdir(certificate);
  await writeFile(join(certificate, 'manifest.json'), JSON.stringify(REFERENCE));
  const redOnly = join(directory, 'red-only');
  await mkdir(join(redOnly, 'red'), { recursive: true });
  // A seed set kept for a deeper run, with nothing to say a generation made it.
  const seeds = join(directory, 'seeds-18');
  for (const role of ['red', 'yellow']) {
    await mkdir(join(seeds, role), { recursive: true });
    await writeFile(join(seeds, role, 'reject-8.bin'), '');
  }
  const committed = fileURLToPath(new URL('../data/perfect-chaos-prefix', import.meta.url));
  const run = (argv) => spawnSync(process.execPath, [SCRIPT, ...argv],
    { encoding: 'utf8', env: { ...process.env, CXX: join(directory, 'no-compiler') }, timeout: 60_000 });
  const untouched = async (label) => {
    assert.deepEqual((await readdir(directory)).sort(), ['certificate', 'red-only', 'seeds-18', 'taken'], label);
    assert.deepEqual(await readdir(taken), ['plan.txt'], label);
    assert.deepEqual(await readdir(certificate), ['manifest.json'], label);
    for (const role of ['red', 'yellow']) assert.deepEqual(await readdir(join(seeds, role)), ['reject-8.bin'], label);
  };
  const holdsPlan = /RangeError: The output .+ holds plan\.txt, which no generation leaves in its output/;
  for (const [argv, message] of [
    [['generate', '--output', taken], holdsPlan],
    [['reproduce-reference', '--output', taken], holdsPlan],
    [['generate', '--frontier-pieces', '18', '--seed-rejections', committed, '--output', seeds],
      /RangeError: The output .+seeds-18 holds red and yellow with neither the manifest\.json of a finished/],
    [['generate', '--seed-rejections', taken, '--output', join(taken, 'red')],
      /RangeError: The output .+ overlaps .+, which generating would delete\./],
    [['reproduce-reference', '--reference', join(certificate, 'manifest.json'), '--seed-rejections', committed,
      '--output', join(certificate, 'out')],
      /RangeError: The output .+ overlaps .+certificate, which generating would delete\./],
    [['generate', '--seed-rejections', redOnly, '--output', join(directory, 'new')],
      /Error: No seed rejections at .+red-only.yellow\./],
    [['generate', '--frontier-pieces', '13', '--output', join(directory, 'odd')],
      /RangeError: The frontier must be an even piece count of at least 8\./],
    [['generate', '--output', join(directory, 'out'), '--journal', join(directory, 'out', 'journal')],
      /RangeError: The prefix journal must be outside the generated output directory\./],
    [['reproduce-reference', '--shards', '0', '--output', join(directory, 'out')],
      /RangeError: shards must be an integer from 1 through 256\./],
  ]) {
    const result = run(argv);
    assert.equal(result.status, 1, argv.join(' '));
    assert.match(result.stderr, message, argv.join(' '));
    await untouched(argv.join(' '));
  }
  // New outputs pass, for the documented generation and the reproduction, and
  // each run stops at the build it then starts.
  for (const argv of [
    ['generate', '--frontier-pieces', '16', '--seed-rejections', committed, '--shards', '8',
      '--shard-from-pieces', '14', '--output', join(directory, 'new')],
    ['reproduce-reference', '--output', join(directory, 'new')],
  ]) {
    const accepted = run(argv);
    assert.equal(accepted.status, 1, argv.join(' '));
    assert.match(accepted.stderr, /is CXX a working C\+\+ compiler\?/, argv.join(' '));
    await untouched(argv.join(' '));
  }
});

// The manifest used to hash the source again once both roles were done: an
// edit during a generation of hours named a source the binary never ran,
// while the journal, keyed on the build, named the one it did.
test('a generated reference names the source its solver was built from', async (context) => {
  if (!findCompiler()) {
    context.skip('no C++ compiler available');
    return;
  }
  const build = await buildNative(SOURCE);
  const output = join(await temporary(context), 'reference');
  // A build of a source that has changed since.
  const edited = { ...build, sourceSha256: 'e'.repeat(64) };
  const { steps, move } = await watch(context, output);
  const manifest = await generateReference(edited, output, 8, 20, null, 1, 14, 2_000_000, 1, null, [], move);
  assert.equal(manifest.sourceSha256, edited.sourceSha256);
  assert.equal(JSON.parse(readFileSync(join(output, 'manifest.json'), 'utf8')).sourceSha256, edited.sourceSha256);
  // The manifest takes over from the mark. It is written beside its place,
  // flushed, and renamed into it while the mark is still there, and the mark
  // goes last: written in place, a kill part-way through left half a
  // manifest that every later run refused. Finished, the output holds no more
  // than it did, and the next generation knows it by its manifest.
  const written = ['generating.json', 'manifest.json.part', 'red', 'yellow'];
  assert.deepEqual(steps, [['flush', ['generating.json.part']],
    ['generating.json.part to generating.json', ['generating.json.part']],
    ['flush', written], ['manifest.json.part to manifest.json', written]]);
  assert.deepEqual((await readdir(output)).sort(), ['manifest.json', 'red', 'yellow']);
  assert.deepEqual((await earlierRun(output)).sort(), ['manifest.json', 'red', 'yellow']);
});

// verify-reference used to hash the source file again, after the build: an
// edit in between was checked against the manifest in place of the source
// the solver had been built from.
test('verify-reference checks the manifest against the source its solver was built from', async (context) => {
  const reference = fileURLToPath(new URL('../data/perfect-chaos-prefix/manifest.json', import.meta.url));
  const scratch = await temporary(context);
  // Node stands in for the solver. It has no `verify` to run, so a build that
  // passes the source check fails next, at the solver's self-test.
  const built = (sourceSha256) => verifyCommittedReference(reference,
    { ...BUILD, sourceSha256, binary: process.execPath }, scratch);
  // A solver built from other source than the file now holds, which is the
  // source the manifest names, is refused before it is run.
  await assert.rejects(built(BUILD.sourceSha256),
    /The prefix verifier source does not match the committed manifest\./);
  await assert.rejects(built(REFERENCE.verificationSourceSha256), /Native prefix verification failed\./);
});

test('seed rejections stop at their deepest file; a missing seed or a gap is an error', async (context) => {
  const directory = await temporary(context);
  // The committed certificate seeds reject-8 to reject-14 of a run to 16.
  const committed = fileURLToPath(new URL('../data/perfect-chaos-prefix', import.meta.url));
  const { rejects } = await initializeRejections(join(directory, 'seeded'), 'red', [8, 10, 12, 14, 16], committed);
  assert.deepEqual([...rejects.keys()], [8, 10, 12, 14, 16]);
  assert.deepEqual(readFileSync(rejects.get(14)), readFileSync(join(committed, 'red', 'reject-14.bin')));
  assert.deepEqual(readFileSync(rejects.get(16)), encodeFrontier(1, 16, []));
  // A mistyped --seed-rejections used to run with no seeds at all.
  await assert.rejects(initializeRejections(join(directory, 'typo'), 'red', [8, 10], join(directory, 'no-seeds')),
    /No seed rejections/);
  const gap = join(directory, 'gap');
  await mkdir(join(gap, 'red'), { recursive: true });
  await writeFile(join(gap, 'red', 'reject-8.bin'), encodeFrontier(1, 8, []));
  await writeFile(join(gap, 'red', 'reject-12.bin'), encodeFrontier(1, 12, []));
  await assert.rejects(initializeRejections(join(directory, 'gapped'), 'red', [8, 10, 12], gap),
    /skip boundary 10/);
  // A generation refuses them before it deletes the earlier run in its
  // output, and refuses yellow's before red's generation: each role's seeds
  // used to be read only once its own generation had started.
  const output = join(directory, 'output');
  await mkdir(join(output, 'red'), { recursive: true });
  await writeFile(join(output, 'generating.json'), await markFor(output));
  await mkdir(join(gap, 'yellow'));
  const redOnly = join(directory, 'red-only');
  await mkdir(join(redOnly, 'red'), { recursive: true });
  for (const [seeds, refusal] of [[gap, /skip boundary 10/], [redOnly, /No seed rejections at .+red-only.yellow\./]]) {
    await assert.rejects(generateReference(null, output, 12, 1, seeds, 1, 14, 2_000_000, 1, null), refusal);
    assert.deepEqual((await readdir(output)).sort(), ['generating.json', 'red']);
  }
  // A run to 10 has no rejections at its frontier, so the missing reject-10 is
  // no gap: the seeds are accepted, and the earlier run is deleted before the
  // null build is read.
  await assert.rejects(generateReference(null, output, 10, 1, gap, 1, 14, 2_000_000, 1, null),
    { name: 'TypeError', message: /'binary'/ });
  assert.deepEqual(await readdir(output), ['generating.json']);
});

test('the prefix journal is keyed on the source, headers, compiler and flags, not binary bytes', () => {
  const journal = { format: 'connect4-chaos-prefix-journal-v4', ...BUILD };
  const descriptor = { kind: 'probe', inputSha256: 'b'.repeat(64) };
  const key = journalKey(journal, descriptor);
  assert.equal(journalKey({ ...journal }, descriptor), key, 'rebuilding the same source reuses entries');
  for (const changed of [
    { sourceSha256: 'c'.repeat(64) }, { headersSha256: { 'io.hpp': 'e'.repeat(64) } }, { compiler: 'clang++' },
    { compilerVersion: 'test 2' }, { flags: ['-O2'] },
  ]) {
    assert.notEqual(journalKey({ ...journal, ...changed }, descriptor), key, JSON.stringify(changed));
  }
});

test('the journal records a rejection only for a completed table and a clean exit 1', async (context) => {
  const directory = await temporary(context);
  const journal = await createJournal(join(directory, 'journal'), BUILD);
  const files = {
    policyPath: join(directory, 'segment.policy.bin'),
    frontierPath: join(directory, 'segment.frontier.bin'),
    rejectedPath: join(directory, 'segment.rejected.bin'),
  };
  const descriptor = { kind: 'extend', frontierPieces: 8, inputSha256: 'd'.repeat(64) };
  const table = encodeFrontier(1, 8, [{ ...EMPTY, mover: 1n, opponent: 2n }]);

  // Killed part-way through writing the table: a failure to retry, not a
  // result to replay on every later run.
  let calls = 0;
  const killed = await journaledSegment(journal, descriptor, async () => {
    calls += 1;
    await writeFile(files.rejectedPath, table.subarray(0, table.length - 5));
    return { code: null, signal: 'SIGKILL', stdout: '', stderr: '', records: [] };
  }, files);
  assert.equal(killed.signal, 'SIGKILL');
  assert.equal(journal.stores, 0);

  // Exit 1 with a table that parses is a rejection, and is reused.
  const rejectedRun = async () => {
    calls += 1;
    await writeFile(files.rejectedPath, table);
    return { code: 1, signal: null, stdout: '', stderr: 'losing roots', records: [] };
  };
  assert.equal((await journaledSegment(journal, descriptor, rejectedRun, files)).code, 1);
  assert.equal(journal.stores, 1);
  assert.equal((await journaledSegment(journal, descriptor, rejectedRun, files)).code, 1);
  assert.equal(journal.hits, 1);
  assert.equal(calls, 2, 'the stored rejection must answer without rerunning the segment');

  // A success whose outputs do not parse is refused before it is stored.
  const tornDescriptor = { ...descriptor, inputSha256: 'e'.repeat(64) };
  await assert.rejects(journaledSegment(journal, tornDescriptor, async () => {
    await writeFile(files.policyPath, encodePolicy(1, 8, []).subarray(0, 10));
    await writeFile(files.frontierPath, encodeFrontier(1, 8, []));
    return { code: 0, signal: null, stdout: '', stderr: '', records: [{}] };
  }, files), /invalid magic header|length does not match/);
  assert.equal(journal.stores, 1);
  const entries = await readdir(join(directory, 'journal'));
  assert.equal(entries.length, 1, entries.join(', '));
});

test('a journal entry left without its metadata, or stored by another run first, does not stop a store', async (context) => {
  const directory = await temporary(context);
  const journal = await createJournal(join(directory, 'journal'), BUILD);
  const files = {
    policyPath: join(directory, 'segment.policy.bin'),
    frontierPath: join(directory, 'segment.frontier.bin'),
    rejectedPath: join(directory, 'segment.rejected.bin'),
  };
  const table = encodeFrontier(1, 8, [{ ...EMPTY, mover: 1n, opponent: 2n }]);
  let calls = 0;
  const rejectedRun = async () => {
    calls += 1;
    await writeFile(files.rejectedPath, table);
    return { code: 1, signal: null, stdout: '', stderr: 'losing roots', records: [] };
  };

  // A kill part-way through clearing a corrupt entry leaves its directory
  // without meta.json. The rename onto it failed with EPERM on Windows, ending
  // the run, and with ENOTEMPTY elsewhere, where the segment then silently
  // stopped being journaled.
  const stale = { kind: 'extend', frontierPieces: 8, inputSha256: 'd'.repeat(64) };
  await mkdir(join(directory, 'journal', journalKey(journal, stale)));
  await writeFile(join(directory, 'journal', journalKey(journal, stale), 'rejected.bin'), table.subarray(0, 7));
  assert.equal((await journaledSegment(journal, stale, rejectedRun, files)).code, 1);
  assert.equal(journal.stores, 1);
  assert.equal((await journaledSegment(journal, stale, rejectedRun, files)).code, 1);
  assert.equal(journal.hits, 1);
  assert.equal(calls, 1, 'the cleared entry was stored again and reused');

  // Two runs sharing one journal both miss; the slower one finds the faster
  // one's entry in place and keeps it.
  const shared = { ...stale, inputSha256: 'e'.repeat(64) };
  const faster = await createJournal(join(directory, 'journal'), BUILD);
  const slower = await journaledSegment(journal, shared, async () => {
    await journaledSegment(faster, shared, rejectedRun, files);
    return rejectedRun();
  }, files);
  assert.equal(slower.code, 1);
  assert.equal(faster.stores, 1);
  assert.equal(journal.stores, 1, 'the second store kept the first one');
  assert.equal((await journaledSegment(journal, shared, rejectedRun, files)).code, 1);
  assert.equal(calls, 3);
  assert.deepEqual((await readdir(join(directory, 'journal'))).sort(),
    [journalKey(journal, shared), journalKey(journal, stale)].sort(), 'no temporary entry is left behind');
});

test('a segment replay refuses files whose headers name another boundary', async (context) => {
  const directory = await temporary(context);
  const policyPath = join(directory, '8-10.policy.bin');
  const frontierPath = join(directory, '8-10.frontier.bin');
  await writeFile(policyPath, encodePolicy(1, 12, []));
  await writeFile(frontierPath, encodeFrontier(1, 12, []));
  await assert.rejects(
    replaySegment({ role: 1, inputStates: [EMPTY], policyPath, frontierPath, boundary: 10 }),
    /8-10\.frontier\.bin ends at 12 pieces, but the segment it belongs to ends at 10/,
  );
});

test('a segment replay rejects a certificate that differs from the committed one', async (context) => {
  // The committed red 0-8 segment, then one change at a time: every check
  // below used to be exercised only by correct certificates.
  const directory = await temporary(context);
  const committed = new URL('../data/perfect-chaos-prefix/red/', import.meta.url);
  const policy = readFileSync(new URL('0-8.policy.bin', committed));
  const frontier = readFileSync(new URL('0-8.frontier.bin', committed));
  const policyPath = join(directory, '0-8.policy.bin');
  const frontierPath = join(directory, '0-8.frontier.bin');
  const replay = async (policyBytes, frontierBytes = frontier) => {
    await writeFile(policyPath, policyBytes);
    await writeFile(frontierPath, frontierBytes);
    return replaySegment({ role: 1, inputStates: [EMPTY], policyPath, frontierPath, boundary: 8 });
  };
  assert.equal((await replay(policy)).closureStates, 3161);
  // Records follow a 16-byte header whose bytes 12-15 count them; a policy
  // record's action is its byte 18 (0 = drop) and the column its byte 19.
  const moved = Buffer.from(policy);
  if (moved[16 + 18] === 0) moved[16 + 19] = (moved[16 + 19] + 1) % 7;
  else moved.set([0, 3], 16 + 18);
  await assert.rejects(replay(moved), /Replay is missing policy state/);
  const surplus = Buffer.concat([policy, Buffer.alloc(20)]);
  surplus.writeUInt32LE(policy.readUInt32LE(12) + 1, 12);
  // States sort by rows, columns, then mover: a rotated board with every
  // mover bit set sorts last and is never reached.
  surplus.writeBigUInt64LE(0xffffffffffffffffn, policy.length);
  surplus.set([7, 6, 0, 3], policy.length + 16);
  await assert.rejects(replay(surplus), /0-8\.policy\.bin has 1 unreachable records/);
  const short = Buffer.from(frontier.subarray(0, frontier.length - 19));
  short.writeUInt32LE(frontier.readUInt32LE(12) - 1, 12);
  await assert.rejects(replay(policy, short), /Replay frontier mismatch for 0-8\.frontier\.bin/);
  // A drop off the board, from the first record (the empty board), and one
  // into a full column. A record's bytes 0-15 hold the two players' stones,
  // the top of column c at bit c * (rows + 1) + rows - 1, and its bytes 16
  // and 17 the board's rows and columns.
  const outside = Buffer.from(policy);
  outside.set([0, 7], 16 + 18);
  await assert.rejects(replay(outside), /Policy drop column is outside the board/);
  const full = Buffer.from(policy);
  const records = policy.readUInt32LE(12);
  let dropped = false;
  for (let index = 0; index < records && !dropped; index += 1) {
    const at = 16 + index * 20;
    const stones = policy.readBigUInt64LE(at) | policy.readBigUInt64LE(at + 8);
    const rows = policy[at + 16];
    const column = [...Array(policy[at + 17]).keys()]
      .find((candidate) => (stones >> BigInt(candidate * (rows + 1) + rows - 1)) & 1n);
    if (column === undefined) continue;
    full.set([0, column], at + 18);
    dropped = true;
  }
  assert.ok(dropped, 'some record has a full column');
  await assert.rejects(replay(full), /Policy attempts to drop in a full column/);
  // Headers name the role at byte 9 and the boundary at byte 10; the two
  // files must agree with each other and with the role replayed.
  const policyRole = Buffer.from(policy);
  policyRole[9] = 2;
  await assert.rejects(replay(policyRole), /Policy\/frontier role or boundary mismatch/);
  const frontierRole = Buffer.from(frontier);
  frontierRole[9] = 2;
  await assert.rejects(replay(policy, frontierRole), /Policy\/frontier role or boundary mismatch/);
  await assert.rejects(replay(policyRole, frontierRole), /Policy\/frontier role or boundary mismatch/);
  const policyBoundary = Buffer.from(policy);
  policyBoundary[10] = 10;
  await assert.rejects(replay(policyBoundary), /Policy\/frontier role or boundary mismatch/);
});

test('a segment replay refuses a closure in which the AI can lose', async (context) => {
  // The check that the certified policy never loses. Bit column*7 + row
  // (from the bottom) is a cell of the 6x7 board: the human, to move, holds
  // the bottom of columns 0-2 and wins by dropping into column 3.
  const directory = await temporary(context);
  const policyPath = join(directory, '0-8.policy.bin');
  const frontierPath = join(directory, '0-8.frontier.bin');
  await writeFile(policyPath, encodePolicy(1, 8, []));
  await writeFile(frontierPath, encodeFrontier(1, 8, []));
  const humanToMove = {
    mover: 1n | (1n << 7n) | (1n << 14n), opponent: (1n << 42n) | (1n << 43n) | (1n << 44n),
    rows: 6, columns: 7, aiTurn: false,
  };
  await assert.rejects(
    replaySegment({ role: 1, inputStates: [humanToMove], policyPath, frontierPath, boundary: 8 }),
    /Replay reaches an AI-loss terminal/,
  );
});

test('the native prefix solver validates its arguments and keeps scratch files in a given directory', async (context) => {
  if (!findCompiler()) {
    context.skip('no C++ compiler available');
    return;
  }
  const { binary } = await buildNative(SOURCE);
  const directory = await temporary(context);

  // The self-test used to write fixed /tmp paths, which do not exist on Windows.
  const verified = await runProcess(binary, ['verify', '--directory', directory]);
  assert.equal(verified.code, 0, verified.stderr);
  assert.equal(verified.stdout.trim().split(/\r?\n/).length, 4);
  assert.deepEqual(await readdir(directory), [], 'the self-test removes its scratch files');
  const undirected = await runProcess(binary, ['verify']);
  assert.equal(undirected.code, 1);
  assert.match(undirected.stderr, /verify requires --directory/);

  // A piece count is stored in a byte: 264 used to become a silent 8.
  for (const pieces of ['264', '0', '43']) {
    const result = await runProcess(binary, ['generate', '--role', 'red', '--frontier-pieces', pieces,
      '--policy', join(directory, 'p.bin'), '--frontier', join(directory, 'f.bin')]);
    assert.equal(result.code, 1, pieces);
    assert.match(result.stderr, /frontier-pieces must be from 1 through 42/, pieces);
  }

  // A full disk surfaces when the last block is flushed on close; the table
  // must not be reported as written. /dev/full exists on Linux only.
  if (existsSync('/dev/full')) {
    const full = await runProcess(binary, ['generate', '--role', 'red', '--frontier-pieces', '2',
      '--policy', '/dev/full', '--frontier', join(directory, 'f.bin')]);
    assert.equal(full.code, 1, full.stdout);
    assert.match(full.stderr, /Policy write failed/);
  }
});
