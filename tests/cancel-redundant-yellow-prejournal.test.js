import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const PATH = '.github/workflows/cancel-redundant-yellow-prejournal.yml';
const SOURCE = readFileSync(new URL(`../${PATH}`, import.meta.url), 'utf8');

test('redundant Yellow cancellation workflow remains valid YAML', () => {
  const result = spawnSync(
    'ruby',
    ['-e', 'require "yaml"; YAML.safe_load(File.read(ARGV.fetch(0)), aliases: true)', PATH],
    { cwd: ROOT, encoding: 'utf8' },
  );
  assert.equal(result.status, 0, result.stderr);
});

test('the gate binds one exact run, source commit, and adapted state', () => {
  assert.match(SOURCE, /REDUNDANT_RUN: '32394581373'/);
  assert.match(SOURCE, /REDUNDANT_SHA: c8ae854d5415ca218ad7c1e284ddf08c743a3e6e/);
  assert.match(SOURCE, /ADAPTED_STATE_SHA256: 2f21c6b7c31f54a04ca0505d1c8421de8d98514a39ef439dc6966935c4e31b85/);
  assert.match(SOURCE, /runId == 32351498111/);
  assert.match(SOURCE, /oldProfile == \{prepareShards:64, prepareWorkers:4\}/);
  assert.match(SOURCE, /newProfile == \{prepareShards:128, prepareWorkers:4\}/);
});

test('cancellation requires production safeguards and refuses non-cancelled results', () => {
  assert.match(SOURCE, /Restore exact preparation journal from the committed source round/);
  assert.match(SOURCE, /perfect-chaos-journal-cache-workflow\.test\.js/);
  assert.match(SOURCE, /workflow_run\.conclusion != 'cancelled'/);
  assert.match(SOURCE, /perfect-chaos-supervisor-recovery-ownership\.test\.js/);
  assert.match(SOURCE, /actions\/runs\/\$REDUNDANT_RUN\/cancel/);
  assert.match(SOURCE, /Refusing to discard a redundant run concluded as/);
  assert.doesNotMatch(SOURCE, /continue-on-error:/);
  assert.doesNotMatch(SOURCE, /\|\| true/);
});

test('the gate self-cleans, audits, and dispatches only after a second active check', () => {
  assert.match(SOURCE, /perfect-chaos-main-redundant-yellow-cancellation\.json/);
  assert.match(SOURCE, /git rm "\$SELF" "\$transient"/);
  assert.match(SOURCE, /\.status != "completed"/);
  assert.match(SOURCE, /An active Yellow continuation already covers the adapted state/);
  assert.match(SOURCE, /gh workflow run "\$CONTINUE_WORKFLOW"/);
  assert.match(SOURCE, /Ambiguous adapted Yellow dispatch/);
});
