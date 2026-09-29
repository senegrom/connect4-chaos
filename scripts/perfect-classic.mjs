#!/usr/bin/env node
import { integerOption, parseArguments as parseCommand } from './cli-options.mjs';
import { isEntryPoint } from './entry-point.mjs';
import { buildNative, parseJsonLines, runProcess } from './native-build.mjs';

import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const SOURCE = join(ROOT, 'native', 'perfect-classic.cpp');
const ROOT_VALUES = join(ROOT, 'data', 'perfect-classic-root-values.json');

// The options each command reads. Anything else is refused rather than
// ignored: `solve --colums 5` solved the default seven columns.
const COMMAND_OPTIONS = Object.freeze({
  verify: [],
  solve: ['rows', 'columns', 'connect', 'table_bits', 'maximum_nodes', 'sequence'],
});

export function parseArguments(argv) {
  return parseCommand(argv, COMMAND_OPTIONS, { defaultCommand: 'verify' });
}

const run = (command, args) => runProcess(command, args, { cwd: ROOT });

// The cached build the native tests and the other proof scripts share: the
// same compiler probe and flags, rebuilt only when the source, its headers,
// the compiler or the flags change.
async function compile() {
  const build = await buildNative(SOURCE, { name: 'perfect-classic' });
  return { compiler: build.compiler, binary: build.binary, warnings: build.warnings };
}

async function verify(binary) {
  const native = await run(binary, ['verify']);
  if (native.code !== 0) throw new Error(`Native classic verification failed.\n${native.stderr}`);
  const records = parseJsonLines(native.stdout);
  const reference = JSON.parse(await readFile(ROOT_VALUES, 'utf8'));
  const published = new Map(
    reference.boards.map((entry) => [`${entry.rows}x${entry.columns}`, entry.value]),
  );
  const expected = [
    [2, 2, 2, 1],
    [3, 3, 3, 0],
    [4, 4, 3, 1],
    [4, 4, 4, published.get('4x4')],
    [4, 5, 4, published.get('4x5')],
    [4, 6, 4, published.get('4x6')],
  ];
  if (records.length !== expected.length) {
    throw new Error('Native classic verification returned the wrong case count.');
  }
  expected.forEach(([rows, columns, connect, value], index) => {
    const record = records[index];
    if (record.rows !== rows || record.columns !== columns
        || record.connect !== connect || record.value !== value
        || !Number.isInteger(record.column) || record.column < 0 || record.column >= columns) {
      throw new Error(`Native classic verification mismatch: ${JSON.stringify(record)}`);
    }
  });
  return records;
}

function solveArguments(options) {
  const rows = integerOption(options.rows, 6, 'rows', 1, 7);
  const columns = integerOption(options.columns, 7, 'columns', 1, 7);
  const connect = integerOption(options.connect, 4, 'connect', 1, Math.max(rows, columns));
  const tableBits = integerOption(options.table_bits, 22, 'table-bits', 8, 27);
  const maximumNodes = integerOption(
    options.maximum_nodes,
    0,
    'maximum-nodes',
    0,
    Number.MAX_SAFE_INTEGER,
  );
  const sequence = options.sequence === undefined ? '' : String(options.sequence);
  return [
    'solve',
    '--rows', String(rows),
    '--columns', String(columns),
    '--connect', String(connect),
    '--table-bits', String(tableBits),
    '--maximum-nodes', String(maximumNodes),
    '--sequence', sequence,
  ];
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const compiled = await compile();
  if (compiled.warnings) process.stderr.write(`${compiled.warnings}\n`);
  if (options.command === 'verify') {
    const records = await verify(compiled.binary);
    process.stdout.write(`${JSON.stringify({
      compiler: compiled.compiler,
      verified: records,
    }, null, 2)}\n`);
    return;
  }
  // solve: parseArguments refused every other command.
  const result = await run(compiled.binary, solveArguments(options));
  if (result.code !== 0) throw new Error(result.stderr || result.stdout);
  process.stdout.write(result.stdout);
}

if (isEntryPoint(import.meta.url)) await main();
