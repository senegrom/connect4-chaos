import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

const workflow = (name) => readFileSync(new URL(`../.github/workflows/${name}`, import.meta.url), 'utf8');
const helper = 'scripts/native-toolchain.mjs';

function job(source, name) {
  const found = source.match(new RegExp(`^  ${name}:\\n([\\s\\S]*?)(?=^  [\\w-]+:|$(?![\\s\\S]))`, 'm'));
  assert.ok(found, `missing ${name} job`);
  return found[0];
}

test('Darwin builds remain an unconditional same-commit Pages dependency', () => {
  const ci = workflow('ci.yml');
  const needs = job(ci, 'pages').match(/^    needs: \[([^\]]+)\]/m)?.[1].split(',').map((s) => s.trim());
  assert.ok(needs?.includes('native-portability'));
  assert.doesNotMatch(job(ci, 'pages'), /always\(|!cancelled\(|continue-on-error:/);
  const call = job(ci, 'native-portability');
  assert.match(call, /uses: \.\/\.github\/workflows\/native-portability\.yml/);
  assert.doesNotMatch(call, /^    (if|continue-on-error):/m);
  const portability = workflow('native-portability.yml');
  assert.match(portability, /  workflow_call:/);
  const darwin = job(portability, 'darwin');
  assert.match(darwin, /runs-on: macos-/);
  assert.doesNotMatch(darwin, /^\s+(if|continue-on-error):|\|\| true/m);
  for (const command of ['tests/native-toolchain.test.mjs', 'tests/perfect-chaos-layered.test.js',
    'tests/perfect-chaos-paired.test.js', 'npm run classic:verify', 'npm run classic:policy:verify',
    'npm run chaos:prefix:verify', 'node scripts/perfect-chaos-native.mjs']) {
    assert.ok(darwin.includes(command), `Darwin gate must exercise ${command}`);
  }
});

test('shared linker policy changes trigger Chaos prefix verification for pushes and PRs', () => {
  const source = workflow('verify-perfect-chaos-prefix.yml');
  for (const event of ['push', 'pull_request']) {
    const section = job(source, event);
    assert.match(section, /    paths:/);
    assert.ok(section.split('\n').some((line) => line.trim() === `- ${helper}`), event);
  }
});

test('the real classic replay fingerprint changes with linker policy bytes and deletion', async (t) => {
  const source = workflow('verify-perfect-classic-policies.yml');
  const start = source.indexOf('      - name: Fingerprint the catalog and everything that judges it\n');
  assert.ok(start >= 0);
  const end = source.indexOf('      - name:', start + 1);
  assert.ok(end > start);
  const step = source.slice(start, end);
  assert.match(step, /        shell: bash\n/);
  const body = step.split('        run: |\n')[1];
  assert.ok(body);
  const script = body.split('\n').map((line) => {
    assert.ok(!line.trim() || line.startsWith('          '));
    return line.slice(10);
  }).join('\n');
  const root = await mkdtemp(join(tmpdir(), 'native-replay-key-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const run = (command, args, env = {}) => {
    const result = spawnSync(command, args, { cwd: root, encoding: 'utf8',
      timeout: 20_000, env: { ...process.env, ...env } });
    assert.ifError(result.error);
    assert.equal(result.status, 0, result.stderr);
    return result.stdout;
  };
  const git = (...args) => run('git', ['-c', 'user.name=Gate test',
    '-c', 'user.email=gate@example.invalid', '-c', 'commit.gpgsign=false', ...args]);
  const commit = () => { git('add', '-A'); git('commit', '-qm', 'fixture'); };
  await mkdir(join(root, 'scripts'));
  await writeFile(join(root, helper), 'original linker policy\n');
  git('init', '-q');
  commit();
  const fingerprint = async () => {
    const output = join(root, 'github-output');
    await rm(output, { force: true });
    run('bash', ['-c', script], { GITHUB_OUTPUT: output });
    const key = (await readFile(output, 'utf8')).trim();
    await rm(output);
    assert.match(key, /^key=perfect-classic-replay-v1-[0-9a-f]{64}$/);
    return key;
  };
  const first = await fingerprint();
  await writeFile(join(root, helper), 'changed linker policy\n');
  commit();
  assert.notEqual(await fingerprint(), first, 'changed linker flags must invalidate the replay');
  await writeFile(join(root, helper), 'original linker policy\n');
  commit();
  assert.equal(await fingerprint(), first, 'identical content can reuse a completed replay');
  await rm(join(root, helper));
  commit();
  assert.notEqual(await fingerprint(), first, 'a deleted dependency must invalidate the replay');
});
