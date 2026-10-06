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
