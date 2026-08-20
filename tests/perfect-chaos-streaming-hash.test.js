import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(new URL('../scripts/perfect-chaos-prefix.mjs', import.meta.url), 'utf8');

test('Perfect Chaos artifact hashes use bounded streaming reads', () => {
  const start = source.indexOf('async function hashFile(path) {');
  const end = source.indexOf('\n}\n', start) + 3;
  assert.ok(start >= 0 && end > start);
  const helper = source.slice(start, end);
  assert.match(helper, /await open\(path, 'r'\)/);
  assert.match(helper, /Buffer\.allocUnsafe\(1024 \* 1024\)/);
  assert.match(helper, /await handle\.read/);
  assert.match(helper, /await handle\.close\(\)/);
  assert.doesNotMatch(helper, /await readFile\(path\)/);
});
