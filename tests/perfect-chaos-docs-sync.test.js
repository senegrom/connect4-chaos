import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { pythonCommand } from '../scripts/python-command.mjs';

const ROOT = fileURLToPath(new URL('..', import.meta.url));

// The generated sections carry arrows, multiplication signs and dashes. The
// script used to read and write with the platform's default encoding, which
// on Windows is cp1252 and turned every check into a false "stale". Force a
// non-UTF-8 default here - cp1252 on Windows, ASCII elsewhere - so the check
// only passes if the script names its encoding itself.
test('the Perfect Chaos docs check is in sync and independent of the locale encoding', (context) => {
  let python;
  try {
    python = pythonCommand();
  } catch {
    context.skip('Python 3 is required');
    return;
  }
  const result = spawnSync(python.command,
    [...python.args, 'scripts/sync-perfect-chaos-docs.py', '--check'], {
      cwd: ROOT,
      encoding: 'utf8',
      timeout: 60_000,
      env: {
        ...process.env,
        PYTHONUTF8: '0',
        PYTHONCOERCECLOCALE: '0',
        LC_ALL: 'C',
        LANG: 'C',
      },
    });
  assert.ifError(result.error);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(JSON.parse(result.stdout).changed.length, 0);
});
