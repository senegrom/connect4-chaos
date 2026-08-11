import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
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

test('perfect-book shards are complete, disjoint, and evenly mixed', () => {
  const counts = Array.from({ length: 8 }, (_, index) => enumerateShard(index));
  assert.equal(counts.reduce((sum, count) => sum + count, 0), 11_094);
  assert.ok(Math.max(...counts) - Math.min(...counts) <= 100, counts.join(', '));
});
