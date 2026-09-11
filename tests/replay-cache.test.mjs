import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';

const workflow = (name) => readFileSync(new URL(`../.github/workflows/${name}`, import.meta.url), 'utf8');
const ci = workflow('ci.yml');
const classic = workflow('verify-perfect-classic-policies.yml');

function job(source, name) {
  const lines = source.split('\n');
  const start = lines.indexOf(`  ${name}:`);
  assert.ok(start >= 0, `missing ${name} job`);
  let end = start + 1;
  while (end < lines.length && !/^  [\w-]+:$/.test(lines[end])) end += 1;
  return lines.slice(start, end).join('\n');
}

// Deliberately layout-specific: fail rather than silently ignore a renamed or
// restructured gate. Shell bodies come from the workflow, not duplicate scripts.
function steps(source) {
  const matches = [...source.matchAll(/^      - name: (.+)$/gm)];
  const result = new Map(matches.map((match, at) => [match[1],
    source.slice(match.index, matches[at + 1]?.index ?? source.length)]));
  assert.equal(result.size, matches.length, 'step names must be unique');
  return result;
}

function field(source, name, indent = 8) {
  assert.equal(typeof source, 'string', `missing step for ${name}`);
  return source.match(new RegExp(`^${' '.repeat(indent)}${name}: (.+)$`, 'm'))?.[1];
}

function shell(source) {
  assert.equal(field(source, 'shell'), 'bash');
  assert.equal(field(source, 'run'), '|');
  return source.split('        run: |\n')[1].split('\n').map((line) => {
    assert.ok(!line.trim() || line.startsWith('          '), 'unexpected shell indentation');
    return line.slice(10);
  }).join('\n');
}

const FINGERPRINT = 'Fingerprint the catalog and everything that judges it';
const RESTORE = 'Look for a finished replay of these bytes';
const REPLAY = 'Independently replay every committed policy';
const SAVE = 'Record that these bytes were replayed';
const REPORT = 'Report a skipped replay';
const CACHE_KEY = '${{ steps.catalog.outputs.key }}';
const MISS = "steps.replayed.outputs.cache-hit != 'true'";
const HIT = "steps.replayed.outputs.cache-hit == 'true'";

function assertReplayGate(source) {
  const verify = job(source, 'verify');
  assert.doesNotMatch(verify, /^    (if|continue-on-error):/m);
  assert.doesNotMatch(verify, /^\s+continue-on-error:|\|\| true|always\(|!cancelled\(/m);
  const all = steps(verify);
  assert.equal(field(all.get('Check out repository'), 'ref', 10), '${{ github.sha }}');
  assert.equal(field(all.get('Verify source and small reference games'), 'run'), 'npm run classic:policy:verify');
  assert.ok(all.has('Validate catalog coverage metadata'));
  // These conditions retain GitHub's implicit success() check. In particular,
  // cache publication must never acquire always() or continue-on-error.
  assert.deepEqual([...all].filter(([, step]) => field(step, 'if') !== undefined)
    .map(([name, step]) => [name, field(step, 'if')]), [[REPLAY, MISS], [SAVE, MISS], [REPORT, HIT]]);
  assert.deepEqual([...all.keys()].filter((name) => [FINGERPRINT, RESTORE, REPLAY, SAVE, REPORT].includes(name)),
    [FINGERPRINT, RESTORE, REPLAY, SAVE, REPORT]);
  assert.equal(field(all.get(FINGERPRINT), 'id'), 'catalog');
  assert.equal(field(all.get(RESTORE), 'id'), 'replayed');
  assert.equal(field(all.get(RESTORE), 'lookup-only', 10), 'true');
  for (const [name, action] of [[RESTORE, 'restore'], [SAVE, 'save']]) {
    const step = all.get(name);
    assert.match(field(step, 'uses'), new RegExp(`^actions/cache/${action}@[0-9a-f]{40}(?: #.*)?$`));
    assert.equal(field(step, 'key', 10), CACHE_KEY);
    assert.equal(field(step, 'path', 10), '.perfect-classic-replayed');
    assert.doesNotMatch(step, /restore-keys:|cache-primary-key/);
  }
  assert.equal(field(all.get(REPLAY), 'REPLAY_KEY', 10), CACHE_KEY);
  assert.match(shell(all.get(FINGERPRINT)), /git ls-tree -r HEAD --/);
  assert.match(shell(all.get(FINGERPRINT)), /sha256sum/);
  assert.match(shell(all.get(REPLAY)), /set -euo pipefail/);
  assert.match(shell(all.get(REPLAY)), /node scripts\/perfect-classic-policy\.mjs verify-reference\s*\\\s*--reference data\/perfect-classic\/manifest\.json/);
  return all;
}

test('replay-cache wiring requires exact hits and success-gated publication', () => {
  const triggers = classic.slice(classic.indexOf('\non:\n'), classic.indexOf('\npermissions:'));
  assert.match(triggers, /  workflow_call:/);
  assert.match(triggers, /  workflow_dispatch:/);
  assert.doesNotMatch(triggers, /  (push|pull_request):/);
  assertReplayGate(classic);
});

test('release guards reject changed-path skips, partial cache keys and failure bypasses', () => {
  const changes = [
    [MISS, "steps.changed.outputs.policies == 'true'"],
    [MISS, "always() && steps.replayed.outputs.cache-hit != 'true'"],
    ['    steps:\n', '    if: false\n    steps:\n'],
    ['        id: replayed\n', '        id: replayed\n        continue-on-error: true\n'],
    ['          lookup-only: true', '          lookup-only: true\n          restore-keys: perfect-classic-replay-'],
    [CACHE_KEY, 'perfect-classic-replay-latest'],
    ['        id: catalog\n', '        id: catalog\n        if: false\n'],
    ['          ref: ${{ github.sha }}', '          ref: main'],
  ];
  for (const [before, after] of changes) {
    assert.ok(classic.includes(before));
    assert.throws(() => assertReplayGate(classic.replace(before, after)), assert.AssertionError, after);
  }
});

function execute(command, args, cwd, env = {}) {
  return spawnSync(command, args, { cwd, encoding: 'utf8', timeout: 10_000,
    env: { ...process.env, ...env } });
}

function succeeded(result) {
  assert.ifError(result.error);
  assert.equal(result.signal, null, result.stderr);
  assert.equal(result.status, 0, result.stderr);
  return result.stdout;
}

async function temporary(t) {
  const root = await mkdtemp(join(tmpdir(), 'classic-replay-gate-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

test('replay fingerprint tracks every proof input, additions, deletions and content reversions', async (t) => {
  const root = await temporary(t);
  const git = (...args) => succeeded(execute('git', ['-c', 'user.name=Gate test',
    '-c', 'user.email=gate@example.invalid', '-c', 'commit.gpgsign=false', ...args], root));
  const inputs = ['data/perfect-classic/manifest.json', 'data/perfect-classic/role1.bin',
    'native/perfect-classic-policy.cpp', 'scripts/perfect-classic-policy.mjs', 'src/data-loader.js',
    'src/engine.js', 'src/perfect-classic-policy.js', '.github/workflows/verify-perfect-classic-policies.yml'];
  for (const path of inputs) {
    await mkdir(dirname(join(root, path)), { recursive: true });
    await writeFile(join(root, path), `original ${path}\n`);
  }
  git('init', '-q');
  const commit = () => { git('add', '-A'); git('commit', '-qm', 'fixture'); };
  commit();
  const original = git('rev-parse', 'HEAD').trim();
  const script = shell(assertReplayGate(classic).get(FINGERPRINT));
  const fingerprint = async () => {
    const output = join(root, 'github-output');
    await rm(output, { force: true });
    succeeded(execute('bash', ['-c', script], root, { GITHUB_OUTPUT: output }));
    const key = (await readFile(output, 'utf8')).trim();
    await rm(output);
    assert.match(key, /^key=perfect-classic-replay-v1-[0-9a-f]{64}$/);
    return key;
  };
  const key = await fingerprint();
  await writeFile(join(root, 'README.md'), 'documentation-only push\n');
  commit();
  assert.equal(await fingerprint(), key, 'a different commit with identical proof inputs reuses evidence');
  for (const path of [...inputs, 'data/perfect-classic/new-role.bin']) {
    await writeFile(join(root, path), 'changed\n');
    commit();
    assert.notEqual(await fingerprint(), key, `${path} must invalidate the receipt`);
    git('restore', '--source', original, '--staged', '--worktree', '.');
    commit();
    assert.equal(await fingerprint(), key, 'restoring identical bytes restores the fingerprint');
  }
  await rm(join(root, inputs[1]));
  commit();
  assert.notEqual(await fingerprint(), key, 'removing a policy also invalidates the receipt');
});

test('only a completed successful replay writes a cache receipt', async (t) => {
  const script = shell(assertReplayGate(classic).get(REPLAY));
  for (const mode of ['success', 'failure', 'cancel']) {
    const root = await temporary(t);
    const bin = join(root, 'bin');
    await mkdir(bin);
    // Replace only the expensive replay. The actual workflow shell handles
    // its exit status, cancellation and receipt publication.
    await writeFile(join(bin, 'node'), `#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == '-e' ]]; then printf '1'; exit 0; fi
[[ "$*" == 'scripts/perfect-classic-policy.mjs verify-reference --reference data/perfect-classic/manifest.json' ]]
touch "$REPLAY_STARTED"
case "$REPLAY_MODE" in
  failure) exit 17 ;;
  cancel) kill -TERM "$PPID"; exit 143 ;;
esac
`, { mode: 0o755 });
    const marker = join(root, 'started');
    const result = execute('bash', ['-c', script], root, { PATH: `${bin}:${process.env.PATH}`,
      REPLAY_KEY: 'test-replay-key', REPLAY_MODE: mode, REPLAY_STARTED: marker });
    assert.ifError(result.error);
    assert.ok(existsSync(marker), 'the expensive replay must actually be invoked on a miss');
    const receipt = join(root, '.perfect-classic-replayed');
    if (mode === 'success') {
      succeeded(result);
      assert.equal(await readFile(receipt, 'utf8'), 'test-replay-key\n');
    } else {
      assert.ok(result.status !== 0 || result.signal, mode);
      assert.equal(existsSync(receipt), false, `${mode} must not be reusable evidence`);
    }
  }
});

test('training CI runs the arena wrapper and CLI regression tests', () => {
  assert.match(job(ci, 'training-regressions'), /python -m neural\.test_modal_arena/);
  const needs = job(ci, 'pages').match(/^    needs: \[([^\]]+)\]/m)?.[1].split(',').map((name) => name.trim());
  assert.ok(needs?.includes('test') && needs.includes('training-regressions'));
});
