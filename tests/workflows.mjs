// Readers for the tests that pin the CI workflows (tests/release-gate.test.mjs,
// replay-cache and browser-recipes). They are deliberately layout-specific: a
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

/** The script of every `run:` step in a job's or a workflow's text, in order,
 * its lines without their indentation, with the `shell:` its step names:
 * undefined for a step left on the runner's default shell. A script is what
 * stands after its key and every line below that is indented past the key:
 * a block scalar, a command that goes on over several lines, or one that
 * starts under a key with nothing after it. `defaults: run:` then comes back
 * too, as a script of its keys. A step written in flow style, between
 * braces, is not read; runKeys counts it. */
export function scripts(source) {
  const lines = source.split('\n');
  // Where a line's first key starts, past the dash of a list item, and the
  // line from there on.
  const keyed = (line) => {
    const column = line.match(/^ *(?:- +)?/)[0].length;
    return [column, line.slice(column)];
  };
  // Whether a line is blank or indented by at least `column`.
  const within = (line, column) => line.trim() === '' || line.search(/\S/) >= column;
  const found = [];
  lines.forEach((line, index) => {
    const [column, key] = keyed(line);
    const value = key.match(/^(?:run|"run"|'run') *:(.*)$/)?.[1].trim();
    if (value === undefined) return;
    // A step's keys all start at one column. Its first follows the dash of
    // its list item, which is where the step begins.
    const first = (other) => /^ *- /.test(other) && keyed(other)[0] === column;
    let start = index;
    while (start > 0 && !first(lines[start]) && (within(lines[start - 1], column) || first(lines[start - 1]))) {
      start -= 1;
    }
    let end = index + 1;
    while (end < lines.length && within(lines[end], column)) end += 1;
    let last = index + 1;
    while (last < lines.length && within(lines[last], column + 1)) last += 1;
    // A block scalar's sign, with any comment after it, is no part of the script.
    const script = [/^(?:[|>][-+\d]*)?(?: *#.*)?$/.test(value) ? '' : value, ...lines.slice(index + 1, last)];
    const shell = lines.slice(start, end).map(keyed)
      .find(([other, text]) => other === column && /^shell *:/.test(text))?.[1];
    found.push({
      script: script.map((other) => other.trim()).join('\n').trim(),
      shell: shell?.slice(shell.indexOf(':') + 1).trim(),
    });
  });
  return found;
}

/** How many `run` keys a text holds, however each is written: on a line of
 * its own, in a flow-style step or between quotes. Comment lines are not
 * counted. Where this is more than scripts() returns, a step is written in a
 * way that scripts() passes over. */
export function runKeys(source) {
  return [...source.replace(/^\s*#.*$/gm, '').matchAll(/(?:^|[\s{,"'])run["']? *:/gm)].length;
}
