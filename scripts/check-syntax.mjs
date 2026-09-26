#!/usr/bin/env node
// Parse-checks every tracked JavaScript and Python source file, so a file no
// test imports - a worker, a build or proof script, the model CDN Worker -
// still fails here on a syntax error instead of wherever it next runs. Each
// .js/.mjs file goes through `node --check` (as an ES module, per
// package.json) unless the browser runs it as a classic script, and one
// Python process compiles every .py file.
import { spawn, spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { availableParallelism } from 'node:os';
import process from 'node:process';
import vm from 'node:vm';

import { isEntryPoint } from './entry-point.mjs';
import { pythonCommand } from './python-command.mjs';

// Registered as a classic service worker (src/cross-origin-isolation.js).
// Checked as a module, a top-level await or an export passed here, and the
// worker would then fail to install and the site would lose isolation.
export const CLASSIC_SCRIPTS = new Set(['cross-origin-isolation-worker.js']);

/** The classic-script parse error in source, or null. Nothing is run. */
export function classicScriptError(source, filename) {
  try {
    new vm.Script(source, { filename });
    return null;
  } catch (error) {
    return `${filename}\n${error}`;
  }
}

// compile(), not ast.parse(): ast accepts what only the compiler rejects -
// a return, yield or await outside a function, a break outside a loop,
// duplicate arguments, a misplaced nonlocal or global - and every one of
// them stops the module from importing.
export const PYTHON_CHECK = [
  'import sys',
  'for path in sys.argv[1:]:',
  '    with open(path, encoding="utf-8") as source:',
  '        compile(source.read(), path, "exec", dont_inherit=True)',
].join('\n');

function nodeCheck(file) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, ['--check', file], { stdio: ['ignore', 'ignore', 'pipe'] });
    let stderr = '';
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('close', (code) => resolve(code === 0 ? null : `${file}\n${stderr}`));
  });
}

async function main() {
  const listed = spawnSync('git', ['ls-files', '-z', '--', '*.js', '*.mjs', '*.py'], { encoding: 'utf8' });
  if (listed.status !== 0) throw new Error(`git ls-files failed: ${listed.stderr || listed.error}`);
  const files = listed.stdout.split('\0').filter(Boolean);
  const scripts = files.filter((file) => !file.endsWith('.py'));
  const python = files.filter((file) => file.endsWith('.py'));

  const failures = [];
  const classic = scripts.filter((file) => CLASSIC_SCRIPTS.has(file));
  if (classic.length !== CLASSIC_SCRIPTS.size) failures.push(`Missing classic scripts: ${[...CLASSIC_SCRIPTS]}`);
  for (const file of classic) {
    const failure = classicScriptError(readFileSync(file, 'utf8'), file);
    if (failure) failures.push(failure);
  }
  const queue = scripts.filter((file) => !CLASSIC_SCRIPTS.has(file));
  await Promise.all(Array.from({ length: Math.min(3, availableParallelism()) }, async () => {
    while (queue.length > 0) {
      const failure = await nodeCheck(queue.shift());
      if (failure) failures.push(failure);
    }
  }));

  const { command, args } = pythonCommand();
  const parse = spawnSync(command, [...args, '-c', PYTHON_CHECK, ...python], { encoding: 'utf8' });
  if (parse.status !== 0) failures.push(`Python sources\n${parse.stderr || parse.error}`);

  if (failures.length > 0) {
    process.stderr.write(`${failures.join('\n')}\n`);
    process.exit(1);
  }
  process.stdout.write(`Parsed ${scripts.length} JavaScript and ${python.length} Python files.\n`);
}

if (isEntryPoint(import.meta.url)) await main();
