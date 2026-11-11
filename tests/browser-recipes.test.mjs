import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import test from 'node:test';

import { workflow } from './workflows.mjs';

// docs/browser-testing.md and docs/WEBKIT_CI.md each give a recipe to run the
// Playwright checks "locally, as CI does". The first fell behind the workflow
// once: CI pinned WebKit in a requirements file of its own and put two CSP
// checks ahead of the suites, and the recipe showed neither, so a local run
// tested another WebKit than CI's without the guard that would have said so.
const browser = workflow('browser-regressions.yml');
const read = (path) => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');

/** A document's recipe: its one shell block that installs the browsers. */
function recipe(path) {
  const blocks = [...read(path).matchAll(/^```sh\n([\s\S]*?)^```$/gm)].map(([, block]) => block)
    .filter((block) => block.includes('playwright install'));
  assert.equal(blocks.length, 1, `${path} should hold one \`\`\`sh block that runs "playwright install", its recipe`);
  return blocks[0];
}

/** The shell lines of a workflow's steps: each `run:` value, on its own line
 * or as the block under it. A comment is not a step, whatever it names. */
function commands(text) {
  const lines = [];
  let column = null; // of the run: key whose block is being read
  for (const line of text.split('\n')) {
    if (column !== null && (line.trim() === '' || line.search(/\S/) > column)) {
      lines.push(line.trim());
      continue;
    }
    column = null;
    const run = line.match(/^( *(?:- )?)run:(.*)$/);
    if (!run) continue;
    if (/^[|>][-+\d]*$/.test(run[2].trim())) column = run[1].length;
    else lines.push(run[2].trim());
  }
  return lines.filter((line) => line !== '' && !line.startsWith('#'));
}

/** What the job runs for one browser: its commands with each
 * ${{ matrix.<key> }} written out. The matrix lists the browsers on one line
 * or, with one pinned as #135 pinned WebKit, gives each an entry of its own. */
function leg(text, name) {
  const entry = text.match(new RegExp(`^( +)- browser: ${name}\\n((?:\\1  .*\\n)*)`, 'm'))?.[2] ?? '';
  return commands(text).map((line) => line.replace(/\$\{\{ *matrix\.([\w-]+) *\}\}/g, (expression, key) => {
    const value = key === 'browser' ? name : entry.match(new RegExp(`^ +${key}: *(.+)$`, 'm'))?.[1];
    assert.ok(value, `"${line}": the matrix gives ${name} no ${key} that this test can read`);
    return value;
  }));
}

/** A recipe's shell lines, with its `for name in a b; do ... done` loop
 * written out and its comments left out. */
const unrolled = (local) => local.replace(/\\\n */g, '')
  .replace(/^for (\w+) in (.+); do\n((?: +.*\n)+)done$/m, (_, name, values, body) => values.trim().split(/ +/)
    .map((value) => body.replaceAll(`$${name}`, value)).join(''))
  .split('\n').map((line) => line.trim()).filter((line) => line !== '' && !line.startsWith('#'));

/** The requirements files a text names. */
const requirements = (text) => [...new Set(text.match(/scripts\/[\w-]*requirements\.txt/g))].sort();

/** The Python scripts some shell lines run, line by line: a suite's line
 * names the evidence runner before the suite. */
const ran = (lines) => lines.map((line) => [...line.matchAll(/scripts\/([\w$-]+)\.py/g)].map(([, name]) => name))
  .filter((names) => names.length > 0);

/** The recipe for both browsers installs what their jobs install, and runs
 * the scripts a job runs, in the job's order. */
function checkRecipe(text, local) {
  const jobs = ['chromium', 'webkit'].map((name) => leg(text, name));
  assert.deepEqual(requirements(local), requirements(jobs.flat().join('\n')));
  assert.deepEqual(ran(unrolled(local)), ran(jobs[0]));
}

/** The WebKit recipe installs what the WebKit job installs, and each check
 * it runs is a line of that job: a virtual environment's python is python. */
function checkWebkitRecipe(text, local) {
  const job = leg(text, 'webkit');
  assert.deepEqual(requirements(local), requirements(job.join('\n')));
  const checks = unrolled(local).filter((line) => ran([line]).length > 0)
    .map((line) => line.replace(/^\S*python[\d.]*(?= )/, 'python'));
  assert.ok(checks.length > 0, 'the recipe runs none of the checks');
  for (const check of checks) assert.ok(job.includes(check), `the WebKit job does not run "${check}"`);
}

test("the local recipe runs the Browser regressions workflow's checks, in its order", () => {
  checkRecipe(browser, recipe('docs/browser-testing.md'));
});

test('the WebKit guard recipe installs what the WebKit job installs and runs a check as that job runs it', () => {
  checkWebkitRecipe(browser, recipe('docs/WEBKIT_CI.md'));
});

// docs/WEBKIT_CI.md says what to do when a browser's release regresses again:
// pin that browser alone, in a requirements file and matrix entry of its own,
// as #135 did. These are the workflow in both layouts, cut down to three
// steps, with comments that name a script and a file no step uses.
const steps = (install) => `    steps:
      # scripts/site-server.py is not needed here: each suite serves the checkout.
      - name: Install test dependencies and browser
        run: |
          python -m pip install -r ${install}
          python -m playwright install --with-deps \${{ matrix.browser }}
      - name: Verify CSP enforcement and strict browser error detection
        run: python scripts/test-browser-csp.py --browser \${{ matrix.browser }}
      - name: Neural worker recovery and real-model smoke
        run: python scripts/browser_evidence.py scripts/neural-worker-regressions.py --browser \${{ matrix.browser }}
`;
const COMMON = `      matrix:
        # scripts/webkit-requirements.txt is gone: both browsers share a release.
        browser: [chromium, webkit]
${steps('scripts/browser-requirements.txt')}`;
const PINNED = `      matrix:
        include:
          - browser: chromium
            requirements: scripts/browser-requirements.txt
          # Unpin once scripts/test-browser-csp.py passes on a newer WebKit.
          - browser: webkit
            requirements: scripts/webkit-requirements.txt
${steps('"${{ matrix.requirements }}"')}`;
const both = (...files) => `${files.map((file) => `python -m pip install -r ${file}`).join('\n')}
python -m playwright install --with-deps chromium webkit
python scripts/test-browser-csp.py --browser chromium
for suite in neural-worker-regressions; do
  python scripts/browser_evidence.py scripts/$suite.py --browser chromium
done
`;
const webkitOnly = (file) => `python3 -m venv .venv
.venv/bin/python -m pip install -r ${file}
.venv/bin/python -m playwright install --with-deps webkit
.venv/bin/python scripts/test-browser-csp.py --browser webkit
`;

test('a browser pinned alone, as #135 pinned WebKit, is what its recipe must install', () => {
  const [common, own] = ['scripts/browser-requirements.txt', 'scripts/webkit-requirements.txt'];
  const refused = { code: 'ERR_ASSERTION', message: /requirements\.txt/ };
  // One release for both browsers, as today.
  checkRecipe(COMMON, both(common));
  checkWebkitRecipe(COMMON, webkitOnly(common));
  assert.throws(() => checkWebkitRecipe(COMMON, webkitOnly(own)), refused, 'a pin the workflow no longer has');
  // WebKit pinned alone: its recipe installs its own file, and the recipe
  // for both browsers installs each browser's.
  checkRecipe(PINNED, both(common, own));
  checkWebkitRecipe(PINNED, webkitOnly(own));
  assert.throws(() => checkWebkitRecipe(PINNED, webkitOnly(common)), refused, 'the file WebKit no longer installs');
  assert.throws(() => checkRecipe(PINNED, both(common)), refused, "without the WebKit job's own file");
  // Either way, a step left out or a suite run without the evidence runner differs from the job.
  const differs = { code: 'ERR_ASSERTION', message: /test-browser-csp|browser_evidence/ };
  assert.throws(() => checkRecipe(COMMON, both(common).replace(/^python scripts\/test-browser-csp.*\n/m, '')), differs);
  assert.throws(() => checkRecipe(COMMON, both(common).replace('scripts/browser_evidence.py ', '')), differs);
  assert.throws(() => checkWebkitRecipe(COMMON, webkitOnly(common).replace(' --browser webkit', '')),
    { code: 'ERR_ASSERTION', message: /the WebKit job does not run/ });
});

// The README's Documentation table lists every document, and the project
// tree under it says how many there are. docs/WEBKIT_CI.md was missing from
// both from #135 on.
const COUNTS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
  'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'];

test('the README lists every document in docs/, and its tree counts them', () => {
  const readme = read('README.md');
  const listed = [...readme.matchAll(/^\| \[([\w-]+)\]\(docs\/\1\.md\) \| .+ \|$/gm)].map(([, name]) => `${name}.md`);
  const documents = readdirSync(new URL('../docs/', import.meta.url)).filter((name) => name.endsWith('.md'));
  assert.deepEqual(listed.sort(), documents.sort());
  assert.equal(readme.match(/ docs\/ +the (\w+) documents listed above$/m)?.[1], COUNTS[documents.length]);
});
