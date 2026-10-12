import assert from 'node:assert/strict';
import { readdirSync } from 'node:fs';
import test from 'node:test';

import { job, needs, workflow } from './workflows.mjs';

// How CI gates Pages. The classic replay workflow's own steps, receipt and
// fingerprint are pinned in tests/replay-cache.test.mjs.
const ci = workflow('ci.yml');
const browser = workflow('browser-regressions.yml');

// The README promises that Pages deploys only after every CI job passes on
// the same commit, and pages.needs is all that holds it to that: main has no
// branch protection. A job dropped from the list, or a new one never added,
// could fail and the site would still deploy.
test('Pages needs every other CI job, and deploys only when CI calls it', () => {
  const jobs = [...ci.slice(ci.indexOf('\njobs:\n')).matchAll(/^  ([\w-]+):$/gm)].map(([, name]) => name);
  assert.deepEqual(needs(ci, 'pages')?.sort(), jobs.filter((name) => name !== 'pages').sort());
  // A trigger of its own, as pages.yml had until 2026-09-08, deploys with no CI at all.
  const pages = workflow('pages.yml');
  assert.equal(pages.slice(pages.indexOf('\non:\n'), pages.indexOf('\npermissions:')).trim(), 'on:\n  workflow_call:');
  // The smoke loads the assembled artifact, so a file the site needs but the
  // artifact omits, or a module URL left unstamped, fails in a browser here.
  assert.match(job(ci, 'test'), /bash scripts\/build-site\.sh _site\n +BROWSER_SMOKE_ROOT=_site npm run test:browser\n/);
});

// A job Pages needs gates nothing once it cannot fail. In that job or in a
// workflow it calls, continue-on-error or `||` on a flaky browser leg or smoke
// would let a commit whose checks fail deploy. always() or !cancelled() would
// run a later step after a check failed: the classic replay's receipt, saved
// after a failed replay, would let the next push skip it. Any other condition
// could skip a check. The only conditions are the evidence upload after a
// failure and the classic replay's receipt (tests/replay-cache.test.mjs).
test('every job Pages needs fails when its checks fail', () => {
  const gates = [];
  const visit = (name, text) => {
    gates.push([name, text]);
    // Every job-level uses: line. A called workflow this test cannot follow,
    // a quoted path or another repository's, would fall out of every check
    // below, so it fails here instead.
    for (const [, value] of text.matchAll(/^    uses:(.*)$/gm)) {
      const target = value.trim();
      const file = target.match(/^\.\/\.github\/workflows\/([\w.-]+\.yml)(?:\s+#.*)?$/)?.[1];
      assert.ok(file, `${name} calls ${JSON.stringify(target)}, which this test cannot follow`);
      visit(file, workflow(file));
    }
  };
  for (const name of needs(ci, 'pages')) visit(name, job(ci, name));
  const conditions = [];
  for (const [name, text] of gates) {
    assert.doesNotMatch(text, /continue-on-error:|\|\||always\(|!cancelled\(/, `${name} lets a failed check pass`);
    for (const [, condition] of text.matchAll(/^[ -]*if:(.*)$/gm)) conditions.push(condition.trim());
  }
  assert.deepEqual(conditions.sort(), ['failure()', "steps.replayed.outputs.cache-hit != 'true'",
    "steps.replayed.outputs.cache-hit != 'true'", "steps.replayed.outputs.cache-hit == 'true'"]);
});

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
