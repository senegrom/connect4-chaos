import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const PATH = '.github/workflows/install-supervisor-cancelled-ownership.yml';
const SOURCE = readFileSync(new URL(`../${PATH}`, import.meta.url), 'utf8');

test('exclusive cancellation-ownership installer remains valid YAML', () => {
  const result = spawnSync(
    'ruby',
    ['-e', 'require "yaml"; YAML.safe_load(File.read(ARGV.fetch(0)), aliases: true)', PATH],
    { cwd: ROOT, encoding: 'utf8' },
  );
  assert.equal(result.status, 0, result.stderr);
});

test('installer performs one anchored failure-and-cancellation ownership patch', () => {
  assert.match(SOURCE, /workflow_run\.conclusion != 'failure'/);
  assert.match(SOURCE, /workflow_run\.conclusion != 'cancelled'/);
  assert.match(SOURCE, /Expected one supervisor ownership condition/);
  assert.match(SOURCE, /neither reviewed old nor reviewed new form/);
  assert.match(SOURCE, /perfect-chaos-supervisor-recovery-ownership\.test\.js/);
  assert.match(SOURCE, /recover-perfect-chaos-main-preparation\.yml/);
  assert.match(SOURCE, /recover-cancelled-perfect-chaos-main-preparation\.yml/);
});

test('installer reruns release gates on latest main and self-cleans', () => {
  assert.match(SOURCE, /git reset --hard origin\/main/);
  assert.match(SOURCE, /npm run ci/);
  assert.match(SOURCE, /npm run test:browser/);
  assert.match(SOURCE, /npm run chaos:prefix:verify-reference/);
  assert.match(SOURCE, /git rm "\$self" "\$transient"/);
  assert.match(SOURCE, /Could not publish exclusive cancellation ownership/);
  assert.doesNotMatch(SOURCE, /continue-on-error:/);
  assert.doesNotMatch(SOURCE, /\|\| true/);
});
