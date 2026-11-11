import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

// docs/browser-testing.md and docs/WEBKIT_CI.md each give a recipe to run the
// Playwright checks "locally, as CI does". The first fell behind the workflow
// once: CI pinned WebKit in a requirements file of its own and put two CSP
// checks ahead of the suites, and the recipe showed neither, so a local run
// tested another WebKit than CI's without the guard that would have said so.
const read = (path) => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');
const workflow = read('.github/workflows/browser-regressions.yml');
/** A document's recipe: its first shell block. */
const recipe = (path) => read(path).match(/^```sh\n([\s\S]*?)^```$/m)?.[1] ?? '';
/** Every requirements file a text names, the workflow's matrix entries included. */
const requirements = (text) => [...new Set(text.match(/scripts\/[\w-]+requirements\.txt/g))].sort();

test("the local recipe runs the Browser regressions workflow's checks, in its order", () => {
  const local = recipe('docs/browser-testing.md');
  assert.deepEqual(requirements(local), requirements(workflow));
  // Every script a step runs: the checks CI starts with, then the suites
  // behind browser_evidence.py, which the recipe loops over by name.
  const scripts = [...new Set([...workflow.matchAll(/scripts\/([\w-]+)\.py/g)].map(([, name]) => name))]
    .filter((name) => name !== 'browser_evidence');
  assert.ok(scripts.includes('test-browser-csp') && scripts.length >= 12, scripts.join(', '));
  let rest = local;
  for (const name of scripts) {
    const found = rest.match(new RegExp(`(?<![\\w-])${name}(?![\\w-])`));
    assert.ok(found, `${name}.py is missing from the recipe, or not where CI runs it`);
    rest = rest.slice(found.index + name.length);
  }
  assert.match(local, /python scripts\/browser_evidence\.py scripts\/\$suite\.py --browser/);
});

test('the WebKit guard recipe installs what CI installs and runs a check CI runs', () => {
  const local = recipe('docs/WEBKIT_CI.md');
  assert.deepEqual(requirements(local), requirements(workflow));
  const guard = local.match(/python (scripts\/[\w-]+\.py --browser) webkit$/m)?.[1];
  assert.ok(guard && workflow.includes(`python ${guard} \${{ matrix.browser }}`), guard);
});
