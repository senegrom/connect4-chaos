// Readers for the tests that pin the CI workflows (tests/release-gate.test.mjs
// and tests/replay-cache.test.mjs). They are deliberately layout-specific: a
// renamed or restructured job fails those tests instead of slipping past them.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

/** The text of .github/workflows/<name>. */
export const workflow = (name) => readFileSync(new URL(`../.github/workflows/${name}`, import.meta.url), 'utf8');

/** One job of a workflow, from its `  name:` line up to the next job's. */
export function job(source, name) {
  const lines = source.split('\n');
  const start = lines.indexOf(`  ${name}:`);
  assert.ok(start >= 0, `missing ${name} job`);
  let end = start + 1;
  while (end < lines.length && !/^  [\w-]+:$/.test(lines[end])) end += 1;
  return lines.slice(start, end).join('\n');
}

/** The jobs a job lists in its `needs: [...]`. */
export function needs(source, name) {
  return job(source, name).match(/^    needs: \[([^\]]+)\]/m)?.[1].split(',').map((entry) => entry.trim());
}

/** Every step's `run:` script in a job's or a workflow's text, its lines
 * without their indentation, with the `shell:` its step names: undefined for
 * a step left on the runner's default shell. */
export function scripts(source) {
  const lines = source.split('\n');
  const found = [];
  lines.forEach((line, index) => {
    const run = line.match(/^( *(?:- )?)run:(.*)$/);
    const value = run?.[2].trim();
    // `defaults: run:` has no value on its line: it is a mapping, not a script.
    if (!value) return;
    // A step's keys all start at one column; its first follows the item's dash.
    const column = run[1].length;
    const dash = `${' '.repeat(column - 2)}- `;
    const below = (other, least) => other.trim() === '' || other.search(/\S/) >= least;
    let start = index;
    while (start > 0 && !lines[start].startsWith(dash)
        && (below(lines[start - 1], column) || lines[start - 1].startsWith(dash))) start -= 1;
    let end = index + 1;
    while (end < lines.length && below(lines[end], column)) end += 1;
    const shell = new RegExp(`^(?: {${column}}|${dash})shell:(.*)$`);
    // A block scalar's lines are those indented past its key.
    let last = index + 1;
    while (last < lines.length && below(lines[last], column + 1)) last += 1;
    const block = lines.slice(index + 1, last).map((other) => other.trim()).join('\n').trim();
    found.push({
      script: /^[|>][-+\d]*$/.test(value) ? block : value,
      shell: lines.slice(start, end).map((other) => other.match(shell)?.[1].trim()).find((name) => name !== undefined),
    });
  });
  return found;
}
