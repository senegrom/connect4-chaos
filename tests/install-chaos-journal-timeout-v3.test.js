import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const PATH = '.github/workflows/install-chaos-journal-timeout-v3.yml';
const source = readFileSync(new URL(`../${PATH}`, import.meta.url), 'utf8');

test('v3 production installer remains valid YAML and has one exact trigger', () => {
  const parsed = spawnSync(
    'ruby',
    ['-e', 'require "yaml"; YAML.safe_load(File.read(ARGV.fetch(0)), aliases: true)', PATH],
    { cwd: ROOT, encoding: 'utf8' },
  );
  assert.equal(parsed.status, 0, parsed.stderr);
  assert.match(source, /paths:\n\s+- \.github\/workflows\/install-chaos-journal-timeout-v3\.yml/);
  assert.match(source, /group: install-bounded-resumable-perfect-chaos-preparation-v3/);
  assert.match(source, /permissions:\n\s+actions: read\n\s+contents: write/);
});

test('installer binds all three operational fixes in one current-main patch', () => {
  assert.match(source, /Restore the content-addressed preparation journal/);
  assert.match(source, /Preserve the content-addressed preparation journal/);
  assert.match(source, /--journal \"\$journal\"/);
  assert.match(source, /timeout --signal=TERM --kill-after=60s 300m/);
  assert.match(source, /workflow_run\.conclusion != 'cancelled'/);
  assert.match(source, /connect4-chaos-journal-restore-v1/);
  assert.match(source, /previous-attempt/);
  assert.match(source, /source-round/);
});

test('every publication race reapplies and reruns the complete release gates', () => {
  assert.match(source, /for publication in 1 2 3 4 5/);
  const loop = source.slice(source.indexOf('for publication in 1 2 3 4 5'));
  const reset = loop.indexOf('git reset --hard origin/main');
  const patch = loop.indexOf('python3 /tmp/install-chaos-journal-timeout-v3.py');
  const exact = loop.indexOf('npm run ci');
  const browser = loop.indexOf('npm run test:browser');
  const replay = loop.indexOf('npm run chaos:prefix:verify-reference');
  const commit = loop.indexOf("git commit -m 'Persist and bound exact Chaos preparation'");
  const push = loop.indexOf('git push origin HEAD:main');
  assert.ok(reset >= 0 && reset < patch && patch < exact && exact < browser);
  assert.ok(browser < replay && replay < commit && commit < push);
  assert.match(loop, /Main advanced during publication; rebuilding and retesting latest main/);
});

test('installer is fail-closed and self-cleans every known scaffold', () => {
  assert.doesNotMatch(source, /continue-on-error:/);
  assert.doesNotMatch(source, /set \+e/);
  assert.doesNotMatch(source, /\|\| true/);
  assert.match(source, /git rm --ignore-unmatch/);
  assert.match(source, /install-chaos-journal-timeout-v3\.yml/);
  assert.match(source, /install-chaos-journal-timeout-v3\.test\.js/);
  assert.match(source, /reinstall-perfect-chaos-journal-cache\.yml/);
  assert.match(source, /install-supervisor-cancelled-ownership\.yml/);
  assert.match(source, /Unexpected production path/);
});
