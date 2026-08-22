import test from 'node:test';
import assert from 'node:assert/strict';
import {
  mkdtempSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const REPOSITORY_ROOT = fileURLToPath(new URL('..', import.meta.url));
const SCRIPT = join(REPOSITORY_ROOT, 'scripts', 'perfect-classic-7x7-failed-run-recovery.py');
const SOURCE_SHA = 'a'.repeat(40);

function launch(overrides = {}) {
  return {
    format: 'connect4-perfect-classic-7x7-launch-v1',
    sourceSha: SOURCE_SHA,
    launchedAt: '2026-08-18T00:00:00Z',
    roles: {
      role1: {
        id: 101,
        head_sha: SOURCE_SHA,
        status: 'in_progress',
        conclusion: null,
      },
      role2: {
        id: 202,
        head_sha: SOURCE_SHA,
        status: 'in_progress',
        conclusion: null,
      },
    },
    ...overrides,
  };
}

function workflowRun(role, overrides = {}) {
  const number = role === 'role1' ? 1 : 2;
  return {
    workflow_run: {
      id: role === 'role1' ? 101 : 202,
      name: `Compute perfect classic 7x7 role ${number} certificate`,
      path: `.github/workflows/compute-perfect-classic-7x7-role${number}.yml`,
      head_branch: 'main',
      head_sha: SOURCE_SHA,
      status: 'completed',
      conclusion: 'failure',
      run_attempt: 1,
      ...overrides,
    },
  };
}

function run(launchValue, eventValue, ...extra) {
  const root = mkdtempSync(join(tmpdir(), 'c4-classic-recovery-'));
  try {
    const launchPath = join(root, 'launch.json');
    const eventPath = join(root, 'event.json');
    writeFileSync(launchPath, `${JSON.stringify(launchValue, null, 2)}\n`);
    writeFileSync(eventPath, `${JSON.stringify(eventValue, null, 2)}\n`);
    return spawnSync('python3', [
      SCRIPT,
      '--launch', launchPath,
      '--event', eventPath,
      ...extra,
    ], { cwd: REPOSITORY_ROOT, encoding: 'utf8' });
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

function parsed(result) {
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}

function expectFailure(result, pattern) {
  assert.notEqual(result.status, 0, 'recovery validator unexpectedly succeeded');
  assert.match(`${result.stdout}\n${result.stderr}`, pattern);
}

test('role 1 and role 2 failures bind to their pinned exact runs', () => {
  for (const role of ['role1', 'role2']) {
    const decision = parsed(run(launch(), workflowRun(role)));
    assert.equal(decision.action, 'rerun-failed-jobs');
    assert.equal(decision.role, role);
    assert.equal(decision.runId, role === 'role1' ? 101 : 202);
    assert.equal(decision.expectedNextAttempt, 2);
  }
});

test('a run id outside the immutable launch record fails closed', () => {
  expectFailure(
    run(launch(), workflowRun('role1', { id: 999 })),
    /does not match pinned run 101/i,
  );
});

test('the event head SHA must equal the launch SHA', () => {
  expectFailure(
    run(launch(), workflowRun('role2', { head_sha: 'b'.repeat(40) })),
    /does not match the pinned launch SHA/i,
  );
});

test('workflow name and path cannot disagree about the role', () => {
  expectFailure(
    run(launch(), workflowRun('role1', {
      path: '.github/workflows/compute-perfect-classic-7x7-role2.yml',
    })),
    /identity is ambiguous|do not identify the same/i,
  );
});

test('successful, cancelled, unfinished, non-main and unrelated runs are ignored', () => {
  const cases = [
    [workflowRun('role1', { conclusion: 'success' }), 'conclusion-success'],
    [workflowRun('role1', { conclusion: 'cancelled' }), 'conclusion-cancelled'],
    [workflowRun('role1', { status: 'in_progress', conclusion: null }), 'run-not-completed'],
    [workflowRun('role1', { head_branch: 'feature' }), 'not-main'],
    [{ workflow_run: {
      id: 303,
      name: 'Other workflow',
      path: '.github/workflows/other.yml',
    } }, 'unsupported-workflow'],
  ];
  for (const [event, reason] of cases) {
    const decision = parsed(run(launch(), event));
    assert.equal(decision.action, 'ignore');
    assert.equal(decision.reason, reason);
  }
});

test('the exact retry budget is bounded', () => {
  const decision = parsed(run(launch(), workflowRun('role2', { run_attempt: 3 })));
  assert.equal(decision.action, 'ignore');
  assert.equal(decision.reason, 'retry-budget-exhausted');
  assert.equal(decision.maximumAttempts, 3);
});

test('a larger explicit retry budget is deterministic', () => {
  const decision = parsed(run(
    launch(),
    workflowRun('role1', { run_attempt: 3 }),
    '--maximum-attempts', '4',
  ));
  assert.equal(decision.action, 'rerun-failed-jobs');
  assert.equal(decision.expectedNextAttempt, 4);
});

test('launch schema and role identities fail closed on drift', () => {
  const missingRole = launch();
  delete missingRole.roles.role2;
  expectFailure(run(missingRole, workflowRun('role1')), /exactly role1 and role2/i);

  const duplicate = launch();
  duplicate.roles.role2.id = duplicate.roles.role1.id;
  expectFailure(run(duplicate, workflowRun('role1')), /cannot share one workflow run id/i);

  const badSha = launch({ sourceSha: 'not-a-sha' });
  expectFailure(run(badSha, workflowRun('role1')), /40-character commit SHA/i);
});
