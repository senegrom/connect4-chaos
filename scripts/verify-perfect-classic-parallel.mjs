#!/usr/bin/env node

import { spawn } from 'node:child_process';
import {
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const VERIFIER = join(ROOT, 'scripts', 'perfect-classic-policy.mjs');

function parseArguments(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
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

function run(command, args) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, {
      cwd: ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    const stdout = [];
    const stderr = [];
    child.stdout.on('data', (chunk) => stdout.push(chunk));
    child.stderr.on('data', (chunk) => stderr.push(chunk));
    child.once('error', reject);
    child.once('close', (code, signal) => {
      resolvePromise({
        code,
        signal,
        stdout: Buffer.concat(stdout).toString('utf8'),
        stderr: Buffer.concat(stderr).toString('utf8'),
      });
    });
  });
}

function policyIdentity(entry) {
  return `${entry.rows}x${entry.columns}:c${entry.connect}:r${entry.role}`;
}

function validateEntry(entry, index) {
  if (!entry || !Number.isInteger(entry.rows) || !Number.isInteger(entry.columns)
      || !Number.isInteger(entry.connect) || !Number.isInteger(entry.role)
      || typeof entry.file !== 'string' || entry.file.length === 0) {
    throw new Error(`Perfect classic manifest entry ${index} is invalid.`);
  }
  return entry;
}

async function prepareSinglePolicy(directory, sourceManifest, entry, index) {
  const policyDirectory = join(directory, String(index).padStart(3, '0'));
  await mkdir(policyDirectory, { recursive: true });
  const source = resolve(dirname(sourceManifest), entry.file);
  const filename = basename(entry.file);
  const target = join(policyDirectory, filename);
  await copyFile(source, target);
  const manifestPath = join(policyDirectory, 'manifest.json');
  await writeFile(manifestPath, `${JSON.stringify({
    format: 'connect4-perfect-classic-manifest-v1',
    policies: [{ ...entry, file: `./${filename}` }],
  }, null, 2)}\n`);
  return manifestPath;
}

async function verifyOne(task, options) {
  const args = [
    VERIFIER,
    'verify-reference',
    '--reference', task.manifestPath,
    '--verify-table-bits', String(options.verifyTableBits),
  ];
  if (options.maximumVerifyNodes !== null) {
    args.push('--maximum-verify-nodes', String(options.maximumVerifyNodes));
  }
  const result = await run(process.execPath, args);
  if (result.code !== 0) {
    throw new Error(
      `Independent replay failed for ${task.identity}.\n${result.stderr || result.stdout}`,
    );
  }
  let parsed;
  try {
    parsed = JSON.parse(result.stdout);
  } catch (error) {
    throw new Error(
      `Independent replay returned invalid JSON for ${task.identity}: ${error.message}`,
    );
  }
  const replay = parsed?.replay;
  if (!Array.isArray(replay) || replay.length !== 1) {
    throw new Error(`Independent replay returned the wrong result count for ${task.identity}.`);
  }
  const record = replay[0];
  if (record.rows !== task.entry.rows
      || record.columns !== task.entry.columns
      || record.connect !== task.entry.connect
      || record.role !== task.entry.role
      || record.entryCount !== task.entry.entryCount
      || record.closureStates !== task.entry.closureStates
      || record.rootValue !== task.entry.rootValue) {
    throw new Error(`Independent replay metadata mismatch for ${task.identity}.`);
  }
  return record;
}

async function runPool(tasks, workers, operation) {
  const results = new Array(tasks.length);
  let cursor = 0;
  let stopped = false;

  const worker = async () => {
    while (!stopped) {
      const index = cursor;
      cursor += 1;
      if (index >= tasks.length) return;
      try {
        results[index] = await operation(tasks[index], index);
      } catch (error) {
        stopped = true;
        throw error;
      }
    }
  };

  await Promise.all(Array.from(
    { length: Math.min(workers, tasks.length) },
    () => worker(),
  ));
  return results;
}

export async function verifyPerfectClassicCatalogParallel(rawOptions = {}) {
  if (!rawOptions.reference || rawOptions.reference === true) {
    throw new RangeError('--reference is required.');
  }
  const reference = resolve(String(rawOptions.reference));
  const manifest = JSON.parse(await readFile(reference, 'utf8'));
  if (manifest?.format !== 'connect4-perfect-classic-manifest-v1'
      || !Array.isArray(manifest.policies)
      || manifest.policies.length === 0) {
    throw new Error('Perfect classic policy manifest format is invalid or empty.');
  }

  const workers = integerOption(rawOptions.workers, 2, 'workers', 1, 8);
  const verifyTableBits = integerOption(
    rawOptions.verify_table_bits,
    22,
    'verify-table-bits',
    8,
    25,
  );
  const maximumVerifyNodes = rawOptions.maximum_verify_nodes === undefined
    ? null
    : integerOption(
      rawOptions.maximum_verify_nodes,
      0,
      'maximum-verify-nodes',
      1,
      Number.MAX_SAFE_INTEGER,
    );

  const identities = new Set();
  const temporary = await mkdtemp(join(tmpdir(), 'perfect-classic-parallel-'));
  const start = Date.now();
  try {
    const tasks = [];
    for (let index = 0; index < manifest.policies.length; index += 1) {
      const entry = validateEntry(manifest.policies[index], index);
      const identity = policyIdentity(entry);
      if (identities.has(identity)) throw new Error(`Duplicate perfect classic policy ${identity}.`);
      identities.add(identity);
      tasks.push({
        entry,
        identity,
        manifestPath: await prepareSinglePolicy(temporary, reference, entry, index),
      });
    }

    let completed = 0;
    const replay = await runPool(tasks, workers, async (task) => {
      const record = await verifyOne(task, { verifyTableBits, maximumVerifyNodes });
      completed += 1;
      process.stderr.write(
        `[${completed}/${tasks.length}] verified ${task.identity}: `
        + `${record.exactNodes.toLocaleString()} exact nodes\n`,
      );
      return record;
    });

    const summary = {
      format: 'connect4-perfect-classic-parallel-replay-v1',
      reference,
      workers,
      verifyTableBits,
      policyCount: replay.length,
      boardCount: new Set(replay.map((entry) => `${entry.rows}x${entry.columns}`)).size,
      exactNodes: replay.reduce((total, entry) => total + entry.exactNodes, 0),
      elapsedMs: Date.now() - start,
      replay,
    };
    if (rawOptions.output && rawOptions.output !== true) {
      const output = resolve(String(rawOptions.output));
      await mkdir(dirname(output), { recursive: true });
      await writeFile(output, `${JSON.stringify(summary, null, 2)}\n`);
    }
    return summary;
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

async function main() {
  const summary = await verifyPerfectClassicCatalogParallel(
    parseArguments(process.argv.slice(2)),
  );
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
