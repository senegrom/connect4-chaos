import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const PATH = '.github/workflows/install-bounded-chaos-preparation.yml';
const SOURCE = readFileSync(new URL(`../${PATH}`, import.meta.url), 'utf8');

test('bounded preparation installer remains valid YAML', () => {
  const result = spawnSync(
    'ruby',
    ['-e', 'require "yaml"; YAML.safe_load(File.read(ARGV.fetch(0)), aliases: true)', PATH],
    { cwd: ROOT, encoding: 'utf8' },
  );
  assert.equal(result.status, 0, result.stderr);
});

test('installer requires journal persistence before adding the timeout', () => {
  assert.match(SOURCE, /Restore exact preparation journal from the committed source round/);
  assert.match(SOURCE, /Restore exact preparation journal from a prior attempt/);
  assert.match(SOURCE, /Preserve exact preparation journal for a rerun/);
  assert.match(SOURCE, /--journal "\$journal"/);
  assert.match(SOURCE, /Journal integration marker/);
});

test('installer preserves failure semantics and runs all release gates', () => {
  assert.match(SOURCE, /timeout --signal=TERM --kill-after=60s 300m/);
  assert.match(SOURCE, /The bounded preparation masks an exact failure/);
  assert.match(SOURCE, /if: \$\{\{ always\(\) \}\}/);
  assert.match(SOURCE, /npm run ci/);
  assert.match(SOURCE, /npm run test:browser/);
  assert.match(SOURCE, /npm run chaos:prefix:verify-reference/);
  assert.match(SOURCE, /git reset --hard origin\/main/);
  assert.match(SOURCE, /git rm "\$self" "\$transient"/);
  assert.doesNotMatch(SOURCE, /continue-on-error:/);
});
