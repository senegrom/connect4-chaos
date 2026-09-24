#!/usr/bin/env node
// Parse-checks every tracked JavaScript and Python source file, so a file no
// test imports - a worker, a build or proof script, the model CDN Worker -
// still fails here on a syntax error instead of wherever it next runs. Each
// .js/.mjs file goes through `node --check` (as an ES module, per
// package.json), and one Python process parses every .py file with ast.
import { spawn, spawnSync } from 'node:child_process';
import { availableParallelism } from 'node:os';
import process from 'node:process';

import { pythonCommand } from './python-command.mjs';

const listed = spawnSync('git', ['ls-files', '-z', '--', '*.js', '*.mjs', '*.py'], { encoding: 'utf8' });
if (listed.status !== 0) throw new Error(`git ls-files failed: ${listed.stderr || listed.error}`);
const files = listed.stdout.split('\0').filter(Boolean);
const scripts = files.filter((file) => !file.endsWith('.py'));
const python = files.filter((file) => file.endsWith('.py'));

function nodeCheck(file) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, ['--check', file], { stdio: ['ignore', 'ignore', 'pipe'] });
    let stderr = '';
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('close', (code) => resolve(code === 0 ? null : `${file}\n${stderr}`));
  });
}

const failures = [];
const queue = [...scripts];
await Promise.all(Array.from({ length: Math.min(3, availableParallelism()) }, async () => {
  while (queue.length > 0) {
    const failure = await nodeCheck(queue.shift());
    if (failure) failures.push(failure);
  }
}));

const { command, args } = pythonCommand();
const parse = spawnSync(command, [...args, '-c', [
  'import ast, sys',
  'for path in sys.argv[1:]:',
  '    with open(path, encoding="utf-8") as source:',
  '        ast.parse(source.read(), filename=path)',
].join('\n'), ...python], { encoding: 'utf8' });
if (parse.status !== 0) failures.push(`Python sources\n${parse.stderr || parse.error}`);

if (failures.length > 0) {
  process.stderr.write(`${failures.join('\n')}\n`);
  process.exit(1);
}
process.stdout.write(`Parsed ${scripts.length} JavaScript and ${python.length} Python files.\n`);
