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
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { isEntryPoint } from './entry-point.mjs';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const VERIFIER = join(ROOT, 'scripts', 'perfect-classic-policy.mjs');
const ROOT_VALUES = join(ROOT, 'data', 'perfect-classic-root-values.json');
// A catalog names its policies as plain files beside the manifest; anything
// else could make the replay read bytes from outside the hashed catalog.
const POLICY_FILE = /^\.\/[A-Za-z0-9][A-Za-z0-9._-]*\.bin$/;

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

function boardIdentity(entry) {
  return `${entry.rows}x${entry.columns}:c${entry.connect}`;
}

function policyIdentity(entry) {
  return `${boardIdentity(entry)}:r${entry.role}`;
}

function validateEntry(entry, index) {
  if (!entry || !Number.isInteger(entry.rows) || !Number.isInteger(entry.columns)
      || !Number.isInteger(entry.connect) || !Number.isInteger(entry.role)) {
    throw new Error(`Perfect classic manifest entry ${index} is invalid.`);
  }
  if (typeof entry.file !== 'string' || !POLICY_FILE.test(entry.file)) {
    throw new Error(
      `Perfect classic manifest entry ${index} must name a ./<name>.bin file beside the manifest.`,
    );
  }
  return entry;
}

// Published values are first-player values for the connect length the file
// declares; a board the file does not list has no external value to check.
async function publishedRootValues(path) {
  const published = JSON.parse(await readFile(path, 'utf8'));
  const connect = published?.rules?.connect;
  if (published?.format !== 'connect4-classic-root-values-v1'
      || !Number.isInteger(connect) || !Array.isArray(published.boards)) {
    throw new Error('Published classic root values are invalid.');
  }
  const values = new Map();
  for (const board of published.boards) {
    if (!Number.isInteger(board?.rows) || !Number.isInteger(board?.columns)
        || ![-1, 0, 1].includes(board.value)) {
      throw new Error('Published classic root values are invalid.');
    }
    const identity = boardIdentity({ ...board, connect });
    if (values.has(identity)) throw new Error(`Duplicate published root value for ${identity}.`);
    values.set(identity, board.value);
  }
  return values;
}

// Each replay proves a lower bound only: its policy forces at least its root
// value against every opponent. The two roles of one board are the two sides
// of the same game, so for the game value v the first role proves v1 <= v and
// the second proves v2 <= -v. Requiring v1 === -v2 therefore pins both to v
// exactly, and the published value checks that v against an outside solution.
function checkRootValues(replay, published) {
  const boards = new Map();
  for (const record of replay) {
    const identity = boardIdentity(record);
    const roles = boards.get(identity) ?? new Map();
    roles.set(record.role, record.rootValue);
    boards.set(identity, roles);
  }
  for (const [identity, roles] of boards) {
    if (roles.size !== 2 || !roles.has(1) || !roles.has(2)) {
      throw new Error(`${identity} must carry both starting-role policies.`);
    }
    const first = roles.get(1);
    const second = roles.get(2);
    if (first !== -second) {
      throw new Error(
        `${identity} role values do not form a pair: role 1 proves ${first}, role 2 proves ${second}.`,
      );
    }
    if (!published.has(identity)) {
      throw new Error(`${identity} has no published root value to check against.`);
    }
    if (published.get(identity) !== first) {
      throw new Error(
        `${identity} proves ${first} but the published root value is ${published.get(identity)}.`,
      );
    }
  }
}

async function prepareSinglePolicy(directory, sourceManifest, entry, index) {
  const policyDirectory = join(directory, String(index).padStart(3, '0'));
  await mkdir(policyDirectory, { recursive: true });
  // validateEntry confined entry.file to a plain ./<name>.bin, so it names a
  // file beside the source manifest and nowhere else.
  await copyFile(join(dirname(sourceManifest), entry.file), join(policyDirectory, entry.file));
  const manifestPath = join(policyDirectory, 'manifest.json');
  await writeFile(manifestPath, `${JSON.stringify({
    format: 'connect4-perfect-classic-manifest-v1',
    policies: [entry],
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
  // Read before the replays, so an unreadable file fails in seconds, not hours.
  // Only fixtures pass --root-values; the release gate uses the committed file.
  const published = await publishedRootValues(
    rawOptions.root_values && rawOptions.root_values !== true
      ? resolve(String(rawOptions.root_values))
      : ROOT_VALUES,
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
    checkRootValues(replay, published);

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

if (isEntryPoint(import.meta.url)) await main();
