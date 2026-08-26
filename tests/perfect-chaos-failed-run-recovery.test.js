import test from 'node:test';
import assert from 'node:assert/strict';
import {
  mkdtempSync,
  mkdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { pythonCommand } from '../scripts/python-command.mjs';

const PYTHON = pythonCommand();

const REPOSITORY_ROOT = fileURLToPath(new URL('..', import.meta.url));
const SCRIPT = join(REPOSITORY_ROOT, 'scripts', 'perfect-chaos-failed-run-recovery.py');
const CAMPAIGN = '.campaign/perfect-chaos-main-18';
const WORKFLOW = '.github/workflows/continue-perfect-chaos-18-main.yml';

function command(cwd, executable, ...args) {
  const result = spawnSync(executable, args, { cwd, encoding: 'utf8' });
  assert.equal(result.status, 0, `${executable} ${args.join(' ')}\n${result.stderr}`);
  return result.stdout.trim();
}

function writeJson(path, value) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`);
}

function state(role, existing, cumulative) {
  return {
    role,
    sourceRun: role === 'red' ? 101 : 202,
    sourceSha: role === 'red' ? 'a'.repeat(40) : 'b'.repeat(40),
    sourceArtifact: `perfect-chaos-${role}-18-source-round`,
    existingRejections: existing,
    cumulativeRejections: cumulative,
    prepareShards: 64,
    prepareWorkers: 4,
    shardCount: 256,
  };
}

function commit(root, message) {
  command(root, 'git', 'add', '.');
  command(root, 'git', 'commit', '-m', message);
  return command(root, 'git', 'rev-parse', 'HEAD');
}

function repository() {
  const root = mkdtempSync(join(tmpdir(), 'c4-failed-run-recovery-'));
  command(root, 'git', 'init');
  command(root, 'git', 'config', 'user.name', 'Test');
  command(root, 'git', 'config', 'user.email', 'test@example.invalid');
  writeJson(join(root, CAMPAIGN, 'red.json'), state('red', 0, 10));
  writeJson(join(root, CAMPAIGN, 'yellow.json'), state('yellow', 0, 20));
  commit(root, 'initial states');
  return root;
}

function failedRun({
  id = 500,
  headSha,
  event = 'push',
  title = 'Continue Perfect Chaos 18-piece refinement',
  attempt = 1,
  path = WORKFLOW,
  conclusion = 'failure',
  status = 'completed',
  branch = 'main',
} = {}) {
  return {
    workflow_run: {
      id,
      path,
      status,
      conclusion,
      head_branch: branch,
      head_sha: headSha,
      event,
      display_title: title,
      run_attempt: attempt,
    },
  };
}

function run(root, event, ...extra) {
  const eventPath = join(root, 'event.json');
  writeJson(eventPath, event);
  return spawnSync(PYTHON.command, [
    ...PYTHON.args,
    SCRIPT,
    '--root', root,
    '--event', eventPath,
    ...extra,
  ], { cwd: REPOSITORY_ROOT, encoding: 'utf8' });
}

function parsed(result) {
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}

function expectFailure(result, pattern) {
  assert.notEqual(result.status, 0, 'recovery decision unexpectedly succeeded');
  assert.match(`${result.stdout}\n${result.stderr}`, pattern);
}

function withRepository(callback) {
  const root = repository();
  try {
    callback(root);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

test('a failed push run retries only the role changed by its exact head commit', () => {
  withRepository((root) => {
    writeJson(join(root, CAMPAIGN, 'red.json'), state('red', 10, 15));
    const headSha = commit(root, 'advance red state');
    const decision = parsed(run(root, failedRun({ headSha })));
    assert.equal(decision.action, 'rerun-failed-jobs');
    assert.equal(decision.role, 'red');
    assert.equal(decision.runAttempt, 1);
    assert.equal(decision.expectedNextAttempt, 2);
    assert.equal(decision.cumulativeRejectedRoots, 15);
  });
});

test('a failed workflow dispatch binds through its exact role-state run name', () => {
  withRepository((root) => {
    const headSha = command(root, 'git', 'rev-parse', 'HEAD');
    const title = `Continue Perfect Chaos 18-piece — ${CAMPAIGN}/yellow.json`;
    const decision = parsed(run(root, failedRun({
      headSha,
      event: 'workflow_dispatch',
      title,
    })));
    assert.equal(decision.action, 'rerun-failed-jobs');
    assert.equal(decision.role, 'yellow');
    assert.equal(decision.cumulativeRejectedRoots, 20);
  });
});

test('a legacy generic dispatch is ambiguous and fails closed', () => {
  withRepository((root) => {
    const headSha = command(root, 'git', 'rev-parse', 'HEAD');
    expectFailure(
      run(root, failedRun({ headSha, event: 'workflow_dispatch' })),
      /must name exactly one role-state path/i,
    );
  });
});

test('a state that advanced after the failed run is never retried', () => {
  withRepository((root) => {
    writeJson(join(root, CAMPAIGN, 'red.json'), state('red', 10, 15));
    const failedHead = commit(root, 'advance red for failed run');
    writeJson(join(root, CAMPAIGN, 'red.json'), state('red', 15, 18));
    commit(root, 'advance red again');
    const decision = parsed(run(root, failedRun({ headSha: failedHead })));
    assert.equal(decision.action, 'ignore');
    assert.equal(decision.reason, 'state-advanced');
    assert.equal(decision.role, 'red');
  });
});

test('the retry budget is bounded deterministically', () => {
  withRepository((root) => {
    writeJson(join(root, CAMPAIGN, 'red.json'), state('red', 10, 15));
    const headSha = commit(root, 'advance red state');
    const decision = parsed(run(root, failedRun({ headSha, attempt: 3 })));
    assert.equal(decision.action, 'ignore');
    assert.equal(decision.reason, 'retry-budget-exhausted');
    assert.equal(decision.maximumAttempts, 3);
  });
});

test('a role with a closure candidate is not relaunched', () => {
  withRepository((root) => {
    writeJson(join(root, CAMPAIGN, 'red.json'), state('red', 10, 15));
    const headSha = commit(root, 'advance red state');
    writeJson(join(root, CAMPAIGN, 'closure-candidates', 'red-15.json'), {
      closedCandidate: true,
    });
    const decision = parsed(run(root, failedRun({ headSha })));
    assert.equal(decision.action, 'ignore');
    assert.equal(decision.reason, 'closure-candidate-present');
  });
});

test('a push commit changing both role states is rejected as ambiguous', () => {
  withRepository((root) => {
    writeJson(join(root, CAMPAIGN, 'red.json'), state('red', 10, 15));
    writeJson(join(root, CAMPAIGN, 'yellow.json'), state('yellow', 20, 25));
    const headSha = commit(root, 'advance both states');
    expectFailure(
      run(root, failedRun({ headSha })),
      /must change exactly one role state/i,
    );
  });
});

test('a push commit changing no role state is rejected', () => {
  withRepository((root) => {
    writeFileSync(join(root, 'unrelated.txt'), 'unrelated\n');
    const headSha = commit(root, 'unrelated change');
    expectFailure(
      run(root, failedRun({ headSha })),
      /must change exactly one role state/i,
    );
  });
});

test('successful, cancelled, unfinished, non-main and unrelated runs are ignored', () => {
  withRepository((root) => {
    const headSha = command(root, 'git', 'rev-parse', 'HEAD');
    const cases = [
      [failedRun({ headSha, conclusion: 'success' }), 'conclusion-success'],
      [failedRun({ headSha, conclusion: 'cancelled' }), 'conclusion-cancelled'],
      [failedRun({ headSha, status: 'in_progress', conclusion: null }), 'run-not-completed'],
      [failedRun({ headSha, branch: 'feature' }), 'not-main'],
      [failedRun({ headSha, path: '.github/workflows/other.yml' }), 'unsupported-workflow'],
    ];
    for (const [event, reason] of cases) {
      const decision = parsed(run(root, event));
      assert.equal(decision.action, 'ignore');
      assert.equal(decision.reason, reason);
    }
  });
});

test('malformed run and state identities fail closed', () => {
  withRepository((root) => {
    const headSha = command(root, 'git', 'rev-parse', 'HEAD');
    expectFailure(
      run(root, failedRun({ headSha: 'not-a-sha', event: 'workflow_dispatch', title: `${CAMPAIGN}/red.json` })),
      /head_sha must be a lowercase 40-character commit SHA/i,
    );

    const invalid = state('red', 0, 10);
    invalid.unreviewed = true;
    writeJson(join(root, CAMPAIGN, 'red.json'), invalid);
    const stateHead = commit(root, 'invalid red schema');
    expectFailure(
      run(root, failedRun({ headSha: stateHead })),
      /unknown=\['unreviewed'\]/,
    );
  });
});
