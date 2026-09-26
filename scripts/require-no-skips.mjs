// Fails a CI test run that skipped tests or marked them todo.
//
// Tests that need a compiler, Python, the model or symlink privileges skip
// themselves when those are missing, which is right on a laptop and wrong in
// CI: a runner image that lost g++ or python3 would turn the proof-tool tests
// into skips, `node --test` would still exit 0, and Pages would deploy. CI has
// run with zero skips, so any skip there means lost coverage. A todo test is
// lost coverage too: it runs, but its failure does not fail the run.
//
// Usage: node scripts/require-no-skips.mjs <tap-file>
// Strict when the CI environment variable is set (GitHub sets CI=true);
// elsewhere it lists the skips and exits 0.
import { readFile } from 'node:fs/promises';

import { isEntryPoint } from './entry-point.mjs';

export function skippedTests(tap) {
  const summary = tap.match(/^# skipped (\d+)$/m);
  if (!summary) throw new Error('No "# skipped" summary line: is this a node --test TAP report?');
  const todo = Number(tap.match(/^# todo (\d+)$/m)?.[1] ?? 0);
  const names = [...tap.matchAll(/^\s*(?:not )?ok \d+ - (.*?) # (SKIP|TODO)(?: (.*))?$/gm)]
    .map(([, name, kind, reason]) => (kind === 'TODO' ? `${name} (todo${reason ? `: ${reason}` : ''})`
      : reason ? `${name} (${reason})` : name));
  return { count: Number(summary[1]) + todo, names };
}

export async function main(argv = process.argv.slice(2), env = process.env) {
  const [path] = argv;
  if (!path) throw new Error('Usage: node scripts/require-no-skips.mjs <tap-file>');
  const { count, names } = skippedTests(await readFile(path, 'utf8'));
  if (count === 0) return 0;
  const strict = Boolean(env.CI) && env.CI !== 'false';
  const list = names.map((name) => `  - ${name}`).join('\n');
  process.stderr.write(`${count} test(s) skipped or marked todo${list ? `:\n${list}` : '.'}\n`);
  if (!strict) return 0;
  process.stderr.write('CI must run every test: install what the skipped tests need, and finish '
    + 'or remove todo tests.\n');
  return 1;
}

if (isEntryPoint(import.meta.url)) process.exitCode = await main();
