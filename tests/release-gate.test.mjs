import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const workflow = (name) => readFileSync(new URL(`../.github/workflows/${name}`, import.meta.url), 'utf8');
const ci = workflow('ci.yml');
const classic = workflow('verify-perfect-classic-policies.yml');
const browser = workflow('browser-regressions.yml');

function job(source, name) {
  const lines = source.split('\n');
  const start = lines.indexOf(`  ${name}:`);
  assert.ok(start >= 0, `missing ${name} job`);
  let end = start + 1;
  while (end < lines.length && !/^  [\w-]+:$/.test(lines[end])) end += 1;
  return lines.slice(start, end).join('\n');
}

test('Pages requires same-commit committed classic-policy verification', () => {
  const pages = job(ci, 'pages');
  const needs = pages.match(/^    needs: \[([^\]]+)\]/m)?.[1].split(',').map((name) => name.trim());
  assert.ok(needs?.includes('classic-policies'), 'separate workflows do not gate deployment');
  assert.doesNotMatch(pages, /always\(|!cancelled\(|continue-on-error:/);
  const gate = job(ci, 'classic-policies');
  assert.match(gate, /uses: \.\/\.github\/workflows\/verify-perfect-classic-policies\.yml/);
  assert.doesNotMatch(gate, /^    (if|continue-on-error):/m);
});

test('classic replay is reusable, pinned to the caller commit, and skipped only by a finished replay', () => {
  const triggers = classic.slice(classic.indexOf('\non:\n'), classic.indexOf('\npermissions:'));
  assert.match(triggers, /  workflow_call:/);
  assert.match(triggers, /  workflow_dispatch:/);
  assert.doesNotMatch(triggers, /  (push|pull_request):/);
  const verify = job(classic, 'verify');
  assert.match(verify, /ref: \$\{\{ github.sha \}\}/);
  assert.match(verify, /node scripts\/perfect-classic-policy\.mjs verify-reference\s*\\\s*--reference data\/perfect-classic\/manifest\.json/);
  assert.doesNotMatch(verify, /continue-on-error:|\|\| true/);

  // The replay costs five hours and proves a property of the committed bytes,
  // so it may be skipped - but only by a run that already finished one over
  // exactly those bytes. This job used to forbid every condition, which is a
  // blunter rule than the one that matters: what a condition is allowed to
  // depend on. A changed-path or event test would let a documentation push
  // publish a catalog nothing had replayed, because CI cancels runs in
  // progress and a cancelled replay leaves main unverified.
  const conditions = [...verify.matchAll(/^\s+if: (.+)$/gm)].map((match) => match[1].trim());
  assert.ok(conditions.length > 0, 'the replay is expected to be skippable');
  for (const condition of conditions) {
    assert.match(condition, /^steps\.replayed\.outputs\.cache-hit [!=]= 'true'$/,
      `the replay may only turn on a finished replay, not on ${condition}`);
  }

  // That cache hit has to be keyed on content, and on everything the verdict
  // depends on: the catalog, the code that reads it, the independent replayer
  // it is checked against, and this workflow. Narrow the fingerprint and a
  // change to the checker would inherit an older run's receipt.
  assert.match(verify, /git ls-tree -r HEAD --/);
  assert.match(verify, /key: \$\{\{ steps\.catalog\.outputs\.key \}\}/);
  for (const path of ['data/perfect-classic', 'native/perfect-classic-policy.cpp',
    'scripts/perfect-classic-policy.mjs', 'src/perfect-classic-policy.js', 'src/engine.js',
    'src/data-loader.js', '.github/workflows/verify-perfect-classic-policies.yml']) {
    assert.ok(verify.includes(path), `the fingerprint must cover ${path}`);
  }

  // And the receipt is written after the replay returns, never before, so a
  // cancelled or failing run leaves none behind.
  const replay = verify.slice(verify.indexOf('Independently replay every committed policy'));
  assert.ok(replay.indexOf('verify-reference') < replay.lastIndexOf('.perfect-classic-replayed'),
    'the receipt must be written after the replay, not before it');
});

test('every browser scenario suite uses the shared pre-teardown evidence runner', () => {
  const suites = ['browser-regressions', 'neural-worker-regressions', 'review-browser-regressions',
    'rereview-browser-regressions', 'failure-browser-regressions', 'handoff-browser-regressions', 'ui-browser-regressions'];
  for (const suite of suites) {
    assert.ok(browser.includes(`python scripts/browser_evidence.py scripts/${suite}.py --browser`), suite);
  }
  assert.match(browser, /python scripts\/test-browser-evidence\.py --browser/);
  assert.match(browser, /path: browser-results\//);
  assert.doesNotMatch(browser, /if-no-files-found: ignore/);
});
