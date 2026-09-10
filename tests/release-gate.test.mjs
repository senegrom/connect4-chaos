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

test('classic replay is reusable, unconditional, and pinned to the caller commit', () => {
  const triggers = classic.slice(classic.indexOf('\non:\n'), classic.indexOf('\npermissions:'));
  assert.match(triggers, /  workflow_call:/);
  assert.match(triggers, /  workflow_dispatch:/);
  assert.doesNotMatch(triggers, /  (push|pull_request):/);
  const verify = job(classic, 'verify');
  assert.match(verify, /ref: \$\{\{ github.sha \}\}/);
  assert.match(verify, /node scripts\/perfect-classic-policy\.mjs verify-reference\s*\\\s*--reference data\/perfect-classic\/manifest\.json/);
  assert.doesNotMatch(verify, /^\s+(if|continue-on-error):|\|\| true/gm);
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
