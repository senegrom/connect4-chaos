import assert from 'node:assert/strict';
import { readdirSync } from 'node:fs';
import test from 'node:test';

import { job, needs, workflow } from './workflows.mjs';

// How CI gates Pages. The classic replay workflow's own steps, receipt and
// fingerprint are pinned in tests/replay-cache.test.mjs.
const ci = workflow('ci.yml');
const browser = workflow('browser-regressions.yml');

test('Pages requires same-commit committed classic-policy verification', () => {
  assert.ok(needs(ci, 'pages')?.includes('classic-policies'), 'separate workflows do not gate deployment');
  assert.doesNotMatch(job(ci, 'pages'), /always\(|!cancelled\(|continue-on-error:/);
  const gate = job(ci, 'classic-policies');
  assert.match(gate, /uses: \.\/\.github\/workflows\/verify-perfect-classic-policies\.yml/);
  assert.doesNotMatch(gate, /^    (if|continue-on-error):/m);
});

test('Pages requires the same-commit Chaos prefix certificate replay', () => {
  assert.ok(needs(ci, 'pages')?.includes('chaos-prefix'), 'the prefix replay must gate deployment');
  const gate = job(ci, 'chaos-prefix');
  assert.match(gate, /uses: \.\/\.github\/workflows\/verify-perfect-chaos-prefix\.yml/);
  assert.doesNotMatch(gate, /^    (if|continue-on-error):/m);

  // Called unconditionally from CI rather than on a path filter, so no change
  // the replay depends on can slip past it.
  const prefix = workflow('verify-perfect-chaos-prefix.yml');
  const triggers = prefix.slice(prefix.indexOf('\non:\n'), prefix.indexOf('\npermissions:'));
  assert.match(triggers, /  workflow_call:/);
  assert.match(triggers, /  workflow_dispatch:/);
  assert.doesNotMatch(triggers, /  (push|pull_request):|paths:/);
  const verify = job(prefix, 'verify');
  assert.match(verify, /ref: \$\{\{ github\.sha \}\}/);
  assert.match(verify, /npm run chaos:prefix:verify-reference/);
  assert.doesNotMatch(verify, /^\s+(if|continue-on-error):|\|\| true/m);
});

test('Pages requires the neural strength benchmark, run on the real network', () => {
  assert.ok(needs(ci, 'pages')?.includes('neural-strength'), 'a network or search that plays worse must not deploy');
  const gate = job(ci, 'neural-strength');
  // A benchmark that skipped would pass having measured nothing.
  assert.match(gate, /tests\/strength\/neural-strength\.mjs && node scripts\/require-no-skips\.mjs node-test\.tap/);
  assert.match(gate, /NEURAL_MODEL_DOWNLOAD: '1'/);
  assert.doesNotMatch(gate, /^\s+(if|continue-on-error):|\|\| true/m);
});

test('Pages requires the same-commit Darwin native builds, unconditionally', () => {
  assert.ok(needs(ci, 'pages')?.includes('native-portability'));
  assert.doesNotMatch(job(ci, 'pages'), /always\(|!cancelled\(|continue-on-error:/);
  const call = job(ci, 'native-portability');
  assert.match(call, /uses: \.\/\.github\/workflows\/native-portability\.yml/);
  assert.doesNotMatch(call, /^    (if|continue-on-error):/m);
  const portability = workflow('native-portability.yml');
  assert.match(portability, /  workflow_call:/);
  const darwin = job(portability, 'darwin');
  assert.match(darwin, /runs-on: macos-/);
  // The gate exists to build with clang; CXX is what selects it.
  assert.match(darwin, /^ +CXX: clang\+\+$/m);
  assert.doesNotMatch(darwin, /^\s+(if|continue-on-error):|\|\| true/m);
  // A runner without a compiler turns these tests into skips; the TAP report
  // and require-no-skips turn any skip back into a failure.
  for (const command of ['tests/native-toolchain.test.mjs', 'tests/perfect-chaos-layered.test.js',
    'tests/perfect-chaos-paired.test.js', '--test-reporter-destination=node-test.tap',
    'node scripts/require-no-skips.mjs node-test.tap', 'npm run classic:verify', 'npm run classic:policy:verify',
    'npm run chaos:prefix:verify']) {
    assert.ok(darwin.includes(command), `Darwin gate must exercise ${command}`);
  }
});

test('every browser scenario suite uses the shared pre-teardown evidence runner', () => {
  // Every scripts/*-regressions.py, so a new suite cannot run outside it.
  const suites = readdirSync(new URL('../scripts/', import.meta.url))
    .filter((name) => name.endsWith('-regressions.py')).map((name) => name.slice(0, -'.py'.length));
  assert.ok(suites.includes('model-cache-browser-regressions') && suites.length >= 8, suites.join(', '));
  for (const suite of suites) {
    assert.ok(browser.includes(`python scripts/browser_evidence.py scripts/${suite}.py --browser`), suite);
  }
  assert.match(browser, /python scripts\/test-browser-evidence\.py --browser/);
  assert.match(browser, /path: browser-results\//);
  assert.doesNotMatch(browser, /if-no-files-found: ignore/);
});
