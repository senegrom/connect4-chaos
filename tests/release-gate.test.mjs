import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

// How CI gates Pages. The classic replay workflow's own steps, receipt and
// fingerprint are pinned in tests/replay-cache.test.mjs.
const workflow = (name) => readFileSync(new URL(`../.github/workflows/${name}`, import.meta.url), 'utf8');
const ci = workflow('ci.yml');
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

test('Pages requires the same-commit Chaos prefix certificate replay', () => {
  const pages = job(ci, 'pages');
  const needs = pages.match(/^    needs: \[([^\]]+)\]/m)?.[1].split(',').map((name) => name.trim());
  assert.ok(needs?.includes('chaos-prefix'), 'the prefix replay must gate deployment');
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
  const pages = job(ci, 'pages');
  const needs = pages.match(/^    needs: \[([^\]]+)\]/m)?.[1].split(',').map((name) => name.trim());
  assert.ok(needs?.includes('neural-strength'), 'a network or search that plays worse must not deploy');
  const gate = job(ci, 'neural-strength');
  // A benchmark that skipped would pass having measured nothing.
  assert.match(gate, /tests\/strength\/neural-strength\.mjs && node scripts\/require-no-skips\.mjs node-test\.tap/);
  assert.match(gate, /NEURAL_MODEL_DOWNLOAD: '1'/);
  assert.doesNotMatch(gate, /^\s+(if|continue-on-error):|\|\| true/m);
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
