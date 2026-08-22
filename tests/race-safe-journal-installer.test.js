import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const PATH = '.github/workflows/reinstall-perfect-chaos-journal-cache.yml';
const SOURCE = readFileSync(new URL(`../${PATH}`, import.meta.url), 'utf8');

test('race-safe journal installer remains valid YAML', () => {
  const result = spawnSync(
    'ruby',
    ['-e', 'require "yaml"; YAML.safe_load(File.read(ARGV.fetch(0)), aliases: true)', PATH],
    { cwd: ROOT, encoding: 'utf8' },
  );
  assert.equal(result.status, 0, result.stderr);
});

test('installer binds the exact reviewed source independently of moving main', () => {
  assert.match(SOURCE, /REVIEWED_INSTALLER_COMMIT: 189216be7956855f08b1bb54131b8220437baa8e/);
  assert.match(SOURCE, /REVIEWED_INSTALLER_BLOB: 3389c7894587ff1571a7a7320ea05e40a01ec8e0/);
  assert.match(SOURCE, /git rev-parse "\$REVIEWED_INSTALLER_COMMIT:\$installer"/);
  assert.match(SOURCE, /test "\$actual" = "\$REVIEWED_INSTALLER_BLOB"/);
  assert.match(SOURCE, /Expected one reviewed step/);
  assert.match(SOURCE, /has no shell script/);
});

test('every publication race rebuilds and reruns all release gates', () => {
  assert.match(SOURCE, /for attempt in 1 2 3 4 5/);
  assert.match(SOURCE, /git reset --hard origin\/main/);
  assert.match(SOURCE, /reviewed-installer-01\.sh/);
  assert.match(SOURCE, /reviewed-installer-02\.sh/);
  assert.match(SOURCE, /reviewed-installer-03\.sh/);
  assert.match(SOURCE, /reviewed-installer-04\.sh/);
  assert.match(SOURCE, /Publication raced with main/);
  assert.match(SOURCE, /rebuilding and retesting latest main/);
  assert.match(SOURCE, /rm -f "\$transient"/);
  assert.doesNotMatch(SOURCE, /continue-on-error:/);
  assert.doesNotMatch(SOURCE, /\|\| true/);
});

test('concurrent successful installation is accepted only with production markers', () => {
  assert.match(SOURCE, /Journal installer is already absent from main/);
  assert.match(SOURCE, /perfect-chaos-journal-cache-workflow\.test\.js/);
  assert.match(SOURCE, /Restore exact preparation journal from the committed source round/);
});
