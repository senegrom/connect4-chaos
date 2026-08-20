import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const PATH = '.github/workflows/reinstall-perfect-chaos-journal-cache.yml';
const SOURCE = readFileSync(new URL(`../${PATH}`, import.meta.url), 'utf8');

const ORDERED_STEPS = [
  'Rebuild the journal integration from structural anchors',
  'Write permanent executable integration tests',
  'Execute the selector and all release gates',
  'Publish the idempotent journal integration',
];

test('journal installer executor remains valid YAML', () => {
  const parsed = spawnSync(
    'ruby',
    ['-e', 'require "yaml"; YAML.safe_load(File.read(ARGV.fetch(0)), aliases: true)', PATH],
    { cwd: ROOT, encoding: 'utf8' },
  );
  assert.equal(parsed.status, 0, parsed.stderr);
});

test('executor binds one exact reviewed parent blob and script sequence', () => {
  assert.match(SOURCE, /REVIEWED_INSTALLER_BLOB: 3389c7894587ff1571a7a7320ea05e40a01ec8e0/);
  assert.match(SOURCE, /git rev-list --parents -n 1 HEAD/);
  assert.match(SOURCE, /test "\$\(wc -w <<< "\$parents"\)" = 2/);
  assert.match(SOURCE, /git rev-parse "\$parent:\$installer"/);
  assert.match(SOURCE, /test "\$actual" = "\$REVIEWED_INSTALLER_BLOB"/);
  let previous = -1;
  for (const name of ORDERED_STEPS) {
    const index = SOURCE.indexOf(`'${name}'`);
    assert.ok(index > previous, `${name} is missing or out of order`);
    previous = index;
  }
  assert.match(SOURCE, /Expected one reviewed step/);
  assert.match(SOURCE, /has no shell script/);
});

test('executor preserves strict failures and self-cleans its transient test', () => {
  assert.match(SOURCE, /set -euo pipefail/);
  assert.doesNotMatch(SOURCE, /continue-on-error:/);
  assert.doesNotMatch(SOURCE, /\|\| true/);
  assert.match(SOURCE, /rm tests\/journal-cache-installer-executor\.test\.js/);
  assert.match(SOURCE, /\/tmp\/reviewed-installer-04\.sh/);
});
