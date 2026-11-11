import assert from 'node:assert/strict';
import { readdirSync } from 'node:fs';
import test from 'node:test';

import { job, needs, scripts, workflow } from './workflows.mjs';

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

/** The ways of passing a failed check that the test below looks for, as it
 * finds them in the text of a job or of a workflow. continue-on-error or
 * `||` on a flaky browser leg or smoke would let a commit whose checks fail
 * deploy. always() or !cancelled() would run a later step after a check
 * failed: the classic replay's receipt, saved after a failed replay, would
 * let the next push skip it. */
function masks(text) {
  const found = [];
  for (const [mask, pattern] of [
    ['continue-on-error', /continue-on-error:/], ['||', /\|\|/], ['always()', /always\(/],
    ['!cancelled()', /!cancelled\(/],
    // scripts/browser_evidence.py then lists a suite's unexpected page and
    // console errors and passes it. The name alone is refused, wherever it
    // is set: an env block, the script itself, a line written to GITHUB_ENV.
    ['C4_BROWSER_ERRORS', /C4_BROWSER_ERRORS/],
    // The script runs on past a failed command and exits as its last one
    // does, or a pipeline exits as its last command does.
    ['set +e', /\bset +\+(?:[a-z]*e[a-z]*|o +errexit)\b/], ['set +o pipefail', /\bset +\+o +pipefail\b/],
  ]) {
    if (pattern.test(text)) found.push(mask);
  }
  // `shell: bash` is the one named shell that is known to stop at a failed
  // command, and at one that fails inside a pipeline.
  for (const [, shell] of text.matchAll(/^[ -]*shell:(.*)$/gm)) {
    if (shell.trim() !== 'bash') found.push(`shell: ${shell.trim()}`);
  }
  // A step left on the default shell runs under `bash -e`, without pipefail:
  // `check | tee log` exits as tee does. Comments may hold a `|`.
  for (const { script, shell } of scripts(text)) {
    if (shell !== 'bash' && /(?<!\|)\|(?!\|)/.test(script.replace(/^\s*#.*$/gm, ''))) {
      found.push('a pipe without pipefail');
    }
  }
  return found;
}

// What every job Pages needs runs: the text of each of those jobs and, read
// whole, of each workflow one of them calls. A workflow of its own, such as
// codeql.yml, is no job of CI's, is not waited for by Pages and is not read.
function gates() {
  // The settings above ci.yml's jobs apply to each of them: an env block
  // there reaches every script, though not a workflow that a job calls.
  const found = [['ci.yml, above its jobs', ci.slice(0, ci.indexOf('\njobs:\n'))]];
  const visit = (name, text) => {
    found.push([name, text]);
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
  return found;
}

// A job Pages needs gates nothing once it cannot fail, so none of them, and
// no workflow one of them calls, may hold a mask that masks() knows. Any
// condition could skip a check: the only ones are the evidence upload after a
// failure and the classic replay's receipt (tests/replay-cache.test.mjs).
//
// This is a tripwire for those masks, not proof that a failed check fails
// its job. It reads the workflows' text and runs nothing, so it cannot see a
// check that hides its own failure: `! check`, a check inside an `if`, one
// before `&&` on a line that is not its script's last, a script or action
// that exits 0 whatever happened. Nor can it see a check that was taken out,
// or a setting kept outside these files. The tests below, and those of the
// scripts the jobs run, pin the checks that must be there.
test('no job Pages needs passes a failed check in a way this test knows', () => {
  const conditions = [];
  for (const [name, text] of gates()) {
    assert.deepEqual(masks(text), [], `${name} lets a failed check pass`);
    for (const [, condition] of text.matchAll(/^[ -]*if:(.*)$/gm)) conditions.push(condition.trim());
  }
  assert.deepEqual(conditions.sort(), ['failure()', "steps.replayed.outputs.cache-hit != 'true'",
    "steps.replayed.outputs.cache-hit != 'true'", "steps.replayed.outputs.cache-hit == 'true'"]);
});

// The test above passes when masks() finds nothing, as it also would if
// masks() could find nothing. Each mask here is in a step such as a later
// edit could add to a job Pages needs. All but the last four used to pass
// that test, which looked for those four alone.
test('each known way of passing a failed check is found in a step that uses it', () => {
  const step = (...keys) => ['    steps:', '      - name: Check', ...keys.map((key) => `        ${key}`)].join('\n');
  for (const [text, found] of [
    [step('run: python scripts/browser_evidence.py scripts/ui-browser-regressions.py', 'env:',
      '  C4_BROWSER_ERRORS: report'), ['C4_BROWSER_ERRORS']],
    ['env:\n  C4_BROWSER_ERRORS: report\n', ['C4_BROWSER_ERRORS']],
    [step('run: echo "C4_BROWSER_ERRORS=report" >> "$GITHUB_ENV"'), ['C4_BROWSER_ERRORS']],
    [step('run: npm run ci | tee ci.log'), ['a pipe without pipefail']],
    [step('run: |', '  npm run ci 2>&1 | tee ci.log', '  echo done'), ['a pipe without pipefail']],
    [step('run: |', '  set +e', '  npm run ci', '  echo done'), ['set +e']],
    [step('run: |', '  set +o pipefail', '  npm run ci | tee ci.log', 'shell: bash'), ['set +o pipefail']],
    [step('run: npm run ci', 'shell: bash {0}'), ['shell: bash {0}']],
    ['defaults:\n  run:\n    shell: sh\n', ['shell: sh']],
    [step('run: npm run ci', 'continue-on-error: true'), ['continue-on-error']],
    [step('run: npm run ci || true'), ['||']],
    [step('run: node scripts/require-no-skips.mjs node-test.tap', 'if: always()'), ['always()']],
    [step('run: node scripts/require-no-skips.mjs node-test.tap', 'if: ${{ !cancelled() }}'), ['!cancelled()']],
  ]) {
    assert.deepEqual(masks(text), found, text);
  }
  // What the jobs do today is none of them: a block scalar's `|`, a comment
  // that holds one, and a pipeline in a step that names bash, which sets
  // pipefail for it.
  for (const text of [
    step('run: |', '  # tee | nothing', '  npm run ci', '  set -x'),
    step('shell: bash', 'run: |', '  set -euo pipefail', '  git ls-tree -r HEAD | sha256sum | cut -d" " -f1'),
  ]) {
    assert.deepEqual(masks(text), [], text);
  }
  // The steps are read in the layout the workflows have: the prefix replay
  // is one bash step, and every browser step runs on the default shell.
  assert.deepEqual(scripts(workflow('verify-perfect-chaos-prefix.yml')), [{
    script: 'set -euo pipefail\nnpm run chaos:prefix:verify\nnpm run chaos:prefix:verify-reference', shell: 'bash',
  }]);
  const suites = scripts(browser);
  assert.equal(suites.length, browser.match(/^ +run:/gm).length);
  assert.ok(suites.every(({ shell }) => shell === undefined));
  assert.ok(suites.some(({ script }) => script.startsWith('python scripts/browser_evidence.py scripts/ui-browser-')));
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
