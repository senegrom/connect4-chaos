import test from 'node:test';
import assert from 'node:assert/strict';
import {
  cpSync,
  mkdtempSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const REPOSITORY_ROOT = fileURLToPath(new URL('..', import.meta.url));
const PATCHER = join(REPOSITORY_ROOT, 'scripts', 'patch-perfect-classic-fragment-downloads.py');
const WORKFLOWS = [
  '.github/workflows/compute-perfect-classic-7x7-role1.yml',
  '.github/workflows/compute-perfect-classic-7x7-role2.yml',
];

function fixture() {
  const root = mkdtempSync(join(tmpdir(), 'c4-classic-workflow-patch-'));
  for (const relative of WORKFLOWS) {
    const source = join(REPOSITORY_ROOT, relative);
    const destination = join(root, relative);
    mkdirSync(dirname(destination), { recursive: true });
    cpSync(source, destination);
  }
  return root;
}

function run(root) {
  return spawnSync('python3', [PATCHER, '--root', root], {
    cwd: REPOSITORY_ROOT,
    encoding: 'utf8',
  });
}

function parsed(result) {
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}

function expectFailure(result, pattern) {
  assert.notEqual(result.status, 0, 'workflow patch unexpectedly succeeded');
  assert.match(`${result.stdout}\n${result.stderr}`, pattern);
}

function withFixture(callback) {
  const root = fixture();
  try {
    callback(root);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

test('both 7×7 workflows contain exactly three verified fragment downloads', () => {
  withFixture((root) => {
    const report = parsed(run(root));
    assert.equal(report.format, 'connect4-perfect-classic-fragment-workflow-patch-v1');
    assert.equal(typeof report.changed, 'boolean');
    assert.equal(report.workflows.length, 2);
    for (const relative of WORKFLOWS) {
      const source = readFileSync(join(root, relative), 'utf8');
      assert.equal((source.match(/scripts\/perfect-classic-download-fragments\.py/g) ?? []).length, 3);
      assert.equal((source.match(/actions\/download-artifact@v4/g) ?? []).length, 1);
      assert.equal((source.match(/^  actions: read$/gm) ?? []).length, 1);
      assert.match(source, /--metadata downloaded\/artifact-download-audit\.json/);
      assert.match(source, /fragment-artifact-audit\.json/);
    }
  });
});

test('the patch is byte-idempotent', () => {
  withFixture((root) => {
    parsed(run(root));
    const before = Object.fromEntries(
      WORKFLOWS.map((relative) => [relative, readFileSync(join(root, relative))]),
    );
    const second = parsed(run(root));
    assert.equal(second.changed, false);
    for (const relative of WORKFLOWS) {
      assert.deepEqual(readFileSync(join(root, relative)), before[relative]);
    }
  });
});

test('the patched workflows remain valid YAML', () => {
  withFixture((root) => {
    parsed(run(root));
    const result = spawnSync('ruby', [
      '-e',
      'require "yaml"; ARGV.each { |p| YAML.safe_load(File.read(p), aliases: true) }',
      ...WORKFLOWS.map((relative) => join(root, relative)),
    ], { encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
  });
});

test('an unexpected artifact pattern fails closed', () => {
  withFixture((root) => {
    const path = join(root, WORKFLOWS[0]);
    const source = readFileSync(path, 'utf8').replace(
      "pattern: 'c4cert-7x7-r1-prefix1-*'",
      "pattern: 'unreviewed-*'",
    );
    writeFileSync(path, source);
    expectFailure(run(root), /artifact pattern sequence/i);
  });
});

test('a missing wildcard stage fails closed instead of silently weakening coverage', () => {
  withFixture((root) => {
    const path = join(root, WORKFLOWS[1]);
    const source = readFileSync(path, 'utf8').replace(
      /      - uses: actions\/download-artifact@v4\n        with: \{ pattern: 'c4cert-7x7-r2-prefix2-\*', path: downloaded \}\n/,
      '',
    );
    writeFileSync(path, source);
    expectFailure(run(root), /artifact pattern sequence|patched workflow must contain/i);
  });
});

test('permission-schema drift fails closed', () => {
  withFixture((root) => {
    const path = join(root, WORKFLOWS[0]);
    const source = readFileSync(path, 'utf8').replace(
      'permissions:\n  contents: read\n',
      'permissions:\n  contents: write\n',
    );
    writeFileSync(path, source);
    expectFailure(run(root), /permissions block/i);
  });
});

test('assembly-command drift fails closed before dropping the audit record', () => {
  withFixture((root) => {
    const path = join(root, WORKFLOWS[0]);
    const source = readFileSync(path, 'utf8').replace(
      'node scripts/perfect-classic-shards.mjs assemble',
      'node scripts/unreviewed-assembler.mjs assemble',
    );
    writeFileSync(path, source);
    expectFailure(run(root), /assembly command/i);
  });
});
