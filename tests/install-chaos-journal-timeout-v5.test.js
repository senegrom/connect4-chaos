import test from 'node:test';
import assert from 'node:assert/strict';
import {
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const PATH = '.github/workflows/install-chaos-journal-timeout-v5.yml';
const source = readFileSync(new URL(`../${PATH}`, import.meta.url), 'utf8');

function embeddedPatcher() {
  const match = source.match(
    /cat > \/tmp\/install-chaos-journal-timeout-v5\.py <<'PY'\n([\s\S]*?)\n\s+PY\n/,
  );
  assert.ok(match, 'embedded Python patcher is missing');
  const lines = match[1].split('\n');
  const nonempty = lines.filter((line) => line.trim());
  const indent = Math.min(...nonempty.map((line) => line.match(/^\s*/)[0].length));
  return lines.map((line) => line.slice(Math.min(indent, line.length))).join('\n');
}

test('installer and embedded patcher are syntactically valid', () => {
  const yaml = spawnSync(
    'ruby',
    ['-e', 'require "yaml"; YAML.safe_load(File.read(ARGV.fetch(0)), aliases: true)', PATH],
    { cwd: ROOT, encoding: 'utf8' },
  );
  assert.equal(yaml.status, 0, yaml.stderr);

  const python = spawnSync(
    'python3',
    ['-c', 'import ast,sys; ast.parse(sys.stdin.read())'],
    { cwd: ROOT, input: embeddedPatcher(), encoding: 'utf8' },
  );
  assert.equal(python.status, 0, python.stderr);
});

test('patcher executes against the exact current workflow structure', () => {
  const sandbox = mkdtempSync(join(tmpdir(), 'connect4-chaos-journal-v5-'));
  try {
    mkdirSync(join(sandbox, '.github', 'workflows'), { recursive: true });
    mkdirSync(join(sandbox, 'tests'), { recursive: true });
    for (const name of [
      'reusable-perfect-chaos-18-round.yml',
      'supervise-perfect-chaos-main.yml',
    ]) {
      copyFileSync(
        join(ROOT, '.github', 'workflows', name),
        join(sandbox, '.github', 'workflows', name),
      );
    }
    const script = join(sandbox, 'install.py');
    writeFileSync(script, embeddedPatcher());
    const applied = spawnSync('python3', [script], { cwd: sandbox, encoding: 'utf8' });
    assert.equal(applied.status, 0, `${applied.stdout}\n${applied.stderr}`);

    const round = readFileSync(
      join(sandbox, '.github', 'workflows', 'reusable-perfect-chaos-18-round.yml'),
      'utf8',
    );
    const supervisor = readFileSync(
      join(sandbox, '.github', 'workflows', 'supervise-perfect-chaos-main.yml'),
      'utf8',
    );
    const restore = round.indexOf('Restore the content-addressed preparation journal');
    const rebuild = round.indexOf('Rebuild the exact 16-piece prefix with bounded deterministic workers');
    const upload = round.indexOf('Preserve the content-addressed preparation journal');
    const publish = round.indexOf('Publish deterministic classification matrix');
    assert.ok(restore >= 0 && restore < rebuild && rebuild < upload && upload < publish);
    assert.equal((round.match(/--journal "\$journal"/g) ?? []).length, 1);
    assert.equal((round.match(/timeout --signal=TERM --kill-after=60s 300m/g) ?? []).length, 1);
    assert.match(round, /previous-attempt/);
    assert.match(round, /adapted-recovery/);
    assert.match(round, /source-round/);
    assert.match(round, /updatedStateSha256 == \$current/);
    assert.match(supervisor, /workflow_run\.conclusion != 'cancelled'/);
    assert.ok(readFileSync(join(sandbox, 'tests', 'perfect-chaos-journal-workflow.test.js'), 'utf8'));
    assert.ok(readFileSync(join(sandbox, 'tests', 'perfect-chaos-supervisor-recovery-ownership.test.js'), 'utf8'));

    const parsed = spawnSync(
      'ruby',
      [
        '-e',
        'require "yaml"; ARGV.each { |path| YAML.safe_load(File.read(path), aliases: true) }',
        '.github/workflows/reusable-perfect-chaos-18-round.yml',
        '.github/workflows/supervise-perfect-chaos-main.yml',
      ],
      { cwd: sandbox, encoding: 'utf8' },
    );
    assert.equal(parsed.status, 0, parsed.stderr);
  } finally {
    rmSync(sandbox, { recursive: true, force: true });
  }
});

test('publication is race-safe, fully gated and self-cleaning', () => {
  assert.match(source, /paths:\n\s+- \.github\/workflows\/install-chaos-journal-timeout-v5\.yml/);
  assert.match(source, /group: install-audit-bound-resumable-perfect-chaos-preparation/);
  assert.match(source, /for publication in 1 2 3 4 5/);
  const loop = source.slice(source.indexOf('for publication in 1 2 3 4 5'));
  const reset = loop.indexOf('git reset --hard origin/main');
  const patch = loop.indexOf('python3 /tmp/install-chaos-journal-timeout-v5.py');
  const exact = loop.indexOf('npm run ci');
  const browser = loop.indexOf('npm run test:browser');
  const replay = loop.indexOf('npm run chaos:prefix:verify-reference');
  const commit = loop.indexOf("git commit -m 'Persist and bound exact Chaos preparation'");
  const push = loop.indexOf('git push origin HEAD:main');
  assert.ok(reset >= 0 && reset < patch && patch < exact && exact < browser);
  assert.ok(browser < replay && replay < commit && commit < push);
  assert.doesNotMatch(source, /continue-on-error:/);
  assert.doesNotMatch(source, /set \+e/);
  assert.doesNotMatch(source, /\|\| true/);
  assert.match(source, /git rm --ignore-unmatch/);
  assert.match(source, /Unexpected production path/);
  assert.match(source, /Main advanced during publication; rebuilding and retesting latest main/);
});
