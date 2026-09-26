import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { CLASSIC_SCRIPTS, PYTHON_CHECK, classicScriptError } from '../scripts/check-syntax.mjs';
import { pythonCommand } from '../scripts/python-command.mjs';

test('the classic service worker is parsed as a classic script', () => {
  // As a module, these passed node --check; as a classic service worker the
  // script would not install, and the site would lose cross-origin isolation.
  for (const source of ['await caches.open("x");', 'export const version = 1;', 'import "./x.js";']) {
    assert.match(classicScriptError(source, 'worker.js'), /SyntaxError/, source);
  }
  assert.equal(classicScriptError('self.addEventListener("fetch", () => {});', 'worker.js'), null);
  for (const file of CLASSIC_SCRIPTS) {
    const url = new URL(`../${file}`, import.meta.url);
    assert.equal(classicScriptError(readFileSync(url, 'utf8'), file), null, file);
  }
});

test('Python sources are compiled, not only parsed', async (context) => {
  let python;
  try {
    python = pythonCommand();
  } catch {
    context.skip('Python 3 is required');
    return;
  }
  const directory = await mkdtemp(join(tmpdir(), 'connect4-python-check-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const check = (path) => spawnSync(python.command, [...python.args, '-c', PYTHON_CHECK, path],
    { encoding: 'utf8' }).status;
  // ast.parse accepts every one of these; importing the module fails.
  for (const [name, source] of [['return.py', 'return 1\n'], ['break.py', 'break\n'],
    ['arguments.py', 'def f(a, a):\n    pass\n'], ['nonlocal.py', 'nonlocal x\n']]) {
    const path = join(directory, name);
    await writeFile(path, source);
    assert.notEqual(check(path), 0, name);
  }
  const valid = join(directory, 'valid.py');
  await writeFile(valid, 'from __future__ import annotations\n\ndef f(a: Missing) -> None:\n    return None\n');
  assert.equal(check(valid), 0);
});
