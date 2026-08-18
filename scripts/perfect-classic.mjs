#!/usr/bin/env node

import { constants as fsConstants } from 'node:fs';
import { access, mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const SOURCE = join(ROOT, 'native', 'perfect-classic.cpp');
const ROOT_VALUES = join(ROOT, 'data', 'perfect-classic-root-values.json');

function parseArguments(argv) {
  const options = { command: argv[0] ?? 'verify' };
  for (let index = 1; index < argv.length; index += 1) {
    const argument = argv[index];
    if (!argument.startsWith('--')) throw new RangeError(`Unexpected argument: ${argument}`);
    const name = argument.slice(2).replaceAll('-', '_');
    const value = argv[index + 1];
    if (value === undefined || value.startsWith('--')) options[name] = true;
    else {
      options[name] = value;
      index += 1;
    }
  }
  return options;
}

function integerOption(value, fallback, label, minimum, maximum) {
  const selected = value === undefined ? fallback : Number.parseInt(String(value), 10);
  if (!Number.isInteger(selected) || selected < minimum || selected > maximum) {
    throw new RangeError(`${label} must be an integer from ${minimum} through ${maximum}.`);
  }
  return selected;
}

async function executable(path) {
  if (!path) return false;
  try {
    await access(path, fsConstants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function findCompiler() {
  if (process.env.CXX) return process.env.CXX;
  for (const candidate of ['/usr/bin/g++', '/usr/bin/clang++']) {
    if (await executable(candidate)) return candidate;
  }
  // Fall back to whatever the PATH offers, so a toolchain installed anywhere
  // other than /usr/bin still works without setting CXX by hand.
  for (const candidate of ['g++', 'clang++']) {
    const probe = await run(candidate, ['--version']).catch(() => null);
    if (probe && probe.code === 0) return candidate;
  }
  throw new Error('A C++20 compiler is required (set CXX, or install g++/clang++).');
}

function run(command, args, options = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, {
      cwd: ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
      ...options,
    });
    const stdout = [];
    const stderr = [];
    child.stdout?.on('data', (chunk) => stdout.push(chunk));
    child.stderr?.on('data', (chunk) => stderr.push(chunk));
    child.once('error', reject);
    child.once('close', (code, signal) => resolvePromise({
      code,
      signal,
      stdout: Buffer.concat(stdout).toString('utf8'),
      stderr: Buffer.concat(stderr).toString('utf8'),
    }));
  });
}

async function compile(directory) {
  const compiler = await findCompiler();
  const binary = join(directory, 'perfect-classic');
  const result = await run(compiler, [
    '-std=c++20',
    '-O3',
    '-Wall',
    '-Wextra',
    '-Wpedantic',
    SOURCE,
    '-o',
    binary,
  ]);
  if (result.code !== 0) {
    throw new Error(`Classic solver compilation failed.\n${result.stderr || result.stdout}`);
  }
  return { compiler, binary, warnings: result.stderr.trim() };
}

function parseJsonLines(output) {
  return output.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
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
  const temporary = await mkdtemp(join(tmpdir(), 'connect4-classic-'));
  try {
    const compiled = await compile(temporary);
    if (compiled.warnings) process.stderr.write(`${compiled.warnings}\n`);
    if (options.command === 'verify') {
      const records = await verify(compiled.binary);
      process.stdout.write(`${JSON.stringify({
        compiler: compiled.compiler,
        verified: records,
      }, null, 2)}\n`);
      return;
    }
    if (options.command === 'solve') {
      const result = await run(compiled.binary, solveArguments(options));
      if (result.code !== 0) throw new Error(result.stderr || result.stdout);
      process.stdout.write(result.stdout);
      return;
    }
    throw new RangeError(`Unknown command: ${options.command}`);
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

await main();
