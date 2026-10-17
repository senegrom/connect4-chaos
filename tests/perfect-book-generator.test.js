import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const generator = fileURLToPath(new URL('../scripts/perfect-book.mjs', import.meta.url));

function enumerateShard(index) {
  const result = spawnSync(process.execPath, [
    generator,
    'enumerate',
    '--depth', '6',
    '--shard-count', '8',
    '--shard-index', String(index),
  ], {
    encoding: 'utf8',
    maxBuffer: 2 * 1024 * 1024,
    timeout: 15_000,
  });
  assert.equal(result.status, 0, result.stderr || result.error?.message);
  assert.match(result.stderr, /Enumerated 11094 canonical positions/);
  assert.ok(result.stdout.endsWith('\n'));
  return result.stdout.slice(0, -1).split('\n').length;
}

test('perfect-book refuses an option its command does not read', () => {
  // A misspelt --depth enumerated the default depth, and a misspelt --output
  // would have packed over the committed book.
  for (const [argv, message] of [
    [['enumerate', '--dpeth', '0'], /enumerate has no option --dpeth\./],
    [['pack', '--input', 'scored.txt', '--ouput', 'new.bin'], /pack has no option --ouput\./],
    [['enumerat'], /Usage:/],
  ]) {
    const result = spawnSync(process.execPath, [generator, ...argv], { encoding: 'utf8', timeout: 15_000 });
    assert.equal(result.status, 1, argv.join(' '));
    assert.match(result.stderr, message);
  }
});

test('perfect-book pack refuses an option given no value, or an empty one, before it writes anything', async (context) => {
  // A bare --manifest let pack write the book, by default over the committed
  // one, and only then fail; a bare --source was recorded as the source.
  const directory = await mkdtemp(join(tmpdir(), 'connect4-book-pack-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  // The empty board's scores, as the solver prints them.
  await writeFile(join(directory, 'scored.txt'), '-2 -1 0 1 0 -1 -2\n');
  for (const [argv, option] of [
    [['--output', 'book.bin', '--manifest'], 'manifest'],
    [['--output', 'book.bin', '--manifest', ''], 'manifest'],
    [['--output', '--manifest', 'book.json'], 'output'],
    [['--output', 'book.bin', '--manifest', 'book.json', '--source'], 'source'],
  ]) {
    const result = spawnSync(process.execPath, [generator, 'pack', '--input', 'scored.txt', ...argv],
      { cwd: directory, encoding: 'utf8', timeout: 15_000 });
    assert.equal(result.status, 1, argv.join(' '));
    assert.match(result.stderr, new RegExp(`--${option} requires a value`), argv.join(' '));
    assert.deepEqual(await readdir(directory), ['scored.txt'], argv.join(' '));
  }
});

test('perfect-book pack refuses a missing --input with its own message, and a bare one first', async (context) => {
  // A bare --input used to reach pack's own check, which refused it as
  // missing. requireValues now refuses it first, so that check is left only
  // an absent --input to name. Pack runs in a directory of its own, where its
  // default output is not the committed book.
  const directory = await mkdtemp(join(tmpdir(), 'connect4-book-input-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  for (const [argv, message] of [
    [['pack'], /--input is required\./],
    [['pack', '--input'], /--input requires a value\./],
  ]) {
    const result = spawnSync(process.execPath, [generator, ...argv],
      { cwd: directory, encoding: 'utf8', timeout: 15_000 });
    assert.equal(result.status, 1, argv.join(' '));
    assert.match(result.stderr, message, argv.join(' '));
  }
});

test('perfect-book shards are complete, disjoint, and evenly mixed', () => {
  const counts = Array.from({ length: 8 }, (_, index) => enumerateShard(index));
  assert.equal(counts.reduce((sum, count) => sum + count, 0), 11_094);
  assert.ok(Math.max(...counts) - Math.min(...counts) <= 100, counts.join(', '));
});
