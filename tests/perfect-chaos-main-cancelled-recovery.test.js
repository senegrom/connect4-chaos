import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { pythonCommand } from '../scripts/python-command.mjs';

const PYTHON = pythonCommand();

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const WORKFLOW = readFileSync(new URL(
  '../.github/workflows/recover-cancelled-perfect-chaos-main-preparation.yml',
  import.meta.url,
), 'utf8');

test('cancelled Perfect Chaos preparation recovery remains fail-closed', () => {
  const result = spawnSync(
    PYTHON.command,
    [...PYTHON.args, 'scripts/test-perfect-chaos-main-cancelled-recovery.py'],
    { cwd: ROOT, encoding: 'utf8' },
  );
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
  assert.match(result.stdout, /'tests': 7/);
  assert.match(result.stdout, /'status': 'pass'/);
});

test('the recovery workflow owns only completed cancelled preparation runs', () => {
  assert.match(WORKFLOW, /workflow_run:/);
  assert.match(WORKFLOW, /github\.event\.workflow_run\.conclusion == 'cancelled'/);
  assert.match(WORKFLOW, /perfect-chaos-main-cancelled-recovery\.py/);
  assert.match(WORKFLOW, /scripts\/test-perfect-chaos-main-cancelled-recovery\.py/);
  assert.match(WORKFLOW, /^  actions: write$/m);
  assert.match(WORKFLOW, /^  contents: write$/m);
  assert.match(WORKFLOW, /steps\.decision\.outputs\.action == 'adapted'/);
  assert.doesNotMatch(
    WORKFLOW,
    /workflow_run\.conclusion == 'failure'/,
  );
});

test('cancelled recovery does not rerun incomplete jobs or mask solver output', () => {
  assert.doesNotMatch(WORKFLOW, /rerun-failed-jobs/);
  assert.doesNotMatch(WORKFLOW, /\|\| true/);
  assert.doesNotMatch(WORKFLOW, /continue-on-error: true/);
  assert.match(WORKFLOW, /EXPECTED_STATE_SHA256/);
  assert.match(WORKFLOW, /UPDATED_STATE_SHA256/);
  assert.match(WORKFLOW, /An active \$ROLE continuation already covers the adapted state/);
});
