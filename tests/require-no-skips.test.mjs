import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { main, skippedTests } from '../scripts/require-no-skips.mjs';

const REPORT = `TAP version 13
# Subtest: serves files
ok 1 - serves files
  ---
  duration_ms: 3.1
  ...
# Subtest: rejects symlinks
ok 2 - rejects symlinks # SKIP Windows symlink privileges are unavailable
  ---
  duration_ms: 0.2
  ...
1..2
# tests 2
# suites 0
# pass 1
# fail 0
# cancelled 0
# skipped 1
# todo 0
# duration_ms 12.5
`;

test('the skip count and names come from the TAP report', () => {
  assert.deepEqual(skippedTests(REPORT), {
    count: 1, names: ['rejects symlinks (Windows symlink privileges are unavailable)'],
  });
  assert.equal(skippedTests(REPORT.replace('# skipped 1', '# skipped 0')).count, 0);
  assert.throws(() => skippedTests('not a report'), /No "# skipped" summary/);
});

test('a skip fails the run only under CI', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'no-skips-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const path = join(directory, 'report.tap');
  await writeFile(path, REPORT);
  const stderr = t.mock.method(process.stderr, 'write', () => true);
  assert.equal(await main([path], { CI: 'true' }), 1);
  assert.equal(await main([path], {}), 0, 'a laptop without symlink rights may skip');
  assert.equal(await main([path], { CI: 'false' }), 0);
  await writeFile(path, REPORT.replace('# skipped 1', '# skipped 0'));
  assert.equal(await main([path], { CI: 'true' }), 0);
  assert.ok(stderr.mock.calls.some(({ arguments: [text] }) => /rejects symlinks/.test(text)));
});
