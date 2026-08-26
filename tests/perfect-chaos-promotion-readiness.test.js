import test from 'node:test';
import assert from 'node:assert/strict';
import {
  mkdtempSync,
  mkdirSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { pythonCommand } from '../scripts/python-command.mjs';

const PYTHON = pythonCommand();

const REPOSITORY_ROOT = fileURLToPath(new URL('..', import.meta.url));
const SCRIPT = join(REPOSITORY_ROOT, 'scripts', 'perfect-chaos-promotion-readiness.py');

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

function candidate(role, cumulative, overrides = {}) {
  return {
    format: 'connect4-chaos-auto-advance-decision-v1',
    role,
    fromPieces: 16,
    targetPieces: 18,
    run: role === 'red' ? 303 : 404,
    runSha: role === 'red' ? 'c'.repeat(40) : 'd'.repeat(40),
    resultArtifact: `perfect-chaos-${role}-18-${cumulative}-round`,
    existingRejectedRoots: cumulative,
    newRejectedRoots: 0,
    cumulativeRejectedRoots: cumulative,
    closedCandidate: true,
    nextState: null,
    checksums: {
      roundManifest: '1'.repeat(64),
      evidenceManifest: '2'.repeat(64),
      closureReplay: '3'.repeat(64),
    },
    ...overrides,
  };
}

function fixture({ red = 100, yellow = 200 } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'c4-promotion-readiness-'));
  writeJson(join(root, 'red.json'), state('red', red - 1, red));
  writeJson(join(root, 'yellow.json'), state('yellow', yellow - 1, yellow));
  return root;
}

function run(root, ...extraArguments) {
  const result = spawnSync(PYTHON.command, [
    ...PYTHON.args,
    SCRIPT,
    '--campaign-root', root,
    '--from-pieces', '16',
    '--target-pieces', '18',
    ...extraArguments,
  ], {
    cwd: REPOSITORY_ROOT,
    encoding: 'utf8',
  });
  return result;
}

function parsed(result) {
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}

function expectFailure(result, pattern) {
  assert.notEqual(result.status, 0, 'validator unexpectedly succeeded');
  assert.match(`${result.stdout}\n${result.stderr}`, pattern);
}

function withFixture(callback, options) {
  const root = fixture(options);
  try {
    callback(root);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

test('readiness is false while both exact closure candidates are absent', () => {
  withFixture((root) => {
    const report = parsed(run(root));
    assert.equal(report.ready, false);
    assert.equal(report.roles.red.present, false);
    assert.equal(report.roles.yellow.present, false);
    assert.equal(report.roles.red.cumulativeRejectedRoots, 100);
    assert.equal(report.roles.yellow.cumulativeRejectedRoots, 200);
  });
});

test('one valid role candidate remains a sound non-ready state', () => {
  withFixture((root) => {
    writeJson(join(root, 'closure-candidates', 'red-100.json'), candidate('red', 100));
    const report = parsed(run(root));
    assert.equal(report.ready, false);
    assert.equal(report.roles.red.present, true);
    assert.equal(report.roles.yellow.present, false);
    assert.equal(report.roles.red.evidenceArtifact, 'perfect-chaos-red-18-100-evidence');
  });
});

test('two exact and state-bound candidates produce a deterministic ready report', () => {
  withFixture((root) => {
    writeJson(join(root, 'closure-candidates', 'red-100.json'), candidate('red', 100));
    writeJson(join(root, 'closure-candidates', 'yellow-200.json'), candidate('yellow', 200));
    const first = run(root);
    const second = run(root);
    const report = parsed(first);
    assert.equal(report.ready, true);
    assert.equal(report.roles.red.resultArtifact, 'perfect-chaos-red-18-100-round');
    assert.equal(report.roles.yellow.resultArtifact, 'perfect-chaos-yellow-18-200-round');
    assert.equal(first.stdout, second.stdout);
  });
});

test('--require-ready exits nonzero without both candidates', () => {
  withFixture((root) => {
    writeJson(join(root, 'closure-candidates', 'red-100.json'), candidate('red', 100));
    expectFailure(run(root, '--require-ready'), /not present yet/i);
  });
});

test('duplicate candidates for one role fail closed', () => {
  withFixture((root) => {
    writeJson(join(root, 'closure-candidates', 'red-100.json'), candidate('red', 100));
    writeJson(join(root, 'closure-candidates', 'red-101.json'), candidate('red', 101));
    expectFailure(run(root), /at most one red closure candidate/i);
  });
});

test('a candidate count that differs from committed state fails closed', () => {
  withFixture((root) => {
    writeJson(join(root, 'closure-candidates', 'red-99.json'), candidate('red', 99));
    expectFailure(run(root), /does not match the committed red state count/i);
  });
});

test('a zero-round decision cannot change its rejection count', () => {
  withFixture((root) => {
    writeJson(join(root, 'closure-candidates', 'red-100.json'), candidate('red', 100, {
      existingRejectedRoots: 99,
    }));
    expectFailure(run(root), /must preserve its cumulative count/i);
  });
});

test('malformed proof checksum identity fails closed', () => {
  withFixture((root) => {
    const value = candidate('red', 100);
    value.checksums.closureReplay = 'not-a-digest';
    writeJson(join(root, 'closure-candidates', 'red-100.json'), value);
    expectFailure(run(root), /closureReplay must be a lowercase SHA-256 digest/i);
  });
});

test('wrong boundary or artifact identity fails closed', () => {
  withFixture((root) => {
    writeJson(join(root, 'closure-candidates', 'red-100.json'), candidate('red', 100, {
      targetPieces: 20,
    }));
    expectFailure(run(root), /targetPieces must be 18/);
  });
  withFixture((root) => {
    writeJson(join(root, 'closure-candidates', 'red-100.json'), candidate('red', 100, {
      resultArtifact: 'unbound-artifact',
    }));
    expectFailure(run(root), /resultArtifact must be/);
  });
});

test('unexpected candidate entries fail closed instead of being ignored', () => {
  withFixture((root) => {
    writeJson(join(root, 'closure-candidates', 'notes.json'), { status: 'looks-good' });
    expectFailure(run(root), /Unexpected closure candidate entry/);
  });
});

test('candidate symlinks are rejected', (context) => {
  withFixture((root) => {
    const real = join(root, 'real-candidate.json');
    writeJson(real, candidate('red', 100));
    const directory = join(root, 'closure-candidates');
    mkdirSync(directory, { recursive: true });
    try {
      symlinkSync(real, join(directory, 'red-100.json'));
    } catch (error) {
      if (error.code === 'EPERM') {
        // Windows only grants symlink creation to elevated or developer-mode
        // sessions; the rejection path itself is covered on Linux CI.
        context.skip('symlink creation is not permitted here');
        return;
      }
      throw error;
    }
    expectFailure(run(root), /must be a regular file/);
  });
});

test('state schema drift fails closed', () => {
  withFixture((root) => {
    const invalid = state('red', 99, 100);
    invalid.unreviewed = true;
    writeJson(join(root, 'red.json'), invalid);
    expectFailure(run(root), /unknown=\['unreviewed'\]/);
  });
});
