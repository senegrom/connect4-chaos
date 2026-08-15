#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { constants as fsConstants } from 'node:fs';
import {
  access,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  PERFECT_CLASSIC_ROLE_FIRST,
  PERFECT_CLASSIC_ROLE_SECOND,
  decodePerfectClassicPolicy,
} from '../src/perfect-classic-policy.js';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const SOURCE = join(ROOT, 'native', 'perfect-classic-policy.cpp');
const ROOT_VALUES = join(ROOT, 'data', 'perfect-classic-root-values.json');
const AI_TURN_BIT = 1n << 63n;
const WIN = 1;
const DRAW = 0;
const LOSS = -1;

function parseArguments(argv) {
  const options = { command: argv[0] ?? 'verify' };
  for (let index = 1; index < argv.length; index += 1) {
    const argument = argv[index];
    if (!argument.startsWith('--')) throw new RangeError(`Unexpected argument: ${argument}`);
    const name = argument.slice(2).replaceAll('-', '_');
    const value = argv[index + 1];
    if (name === 'input') {
      if (value === undefined || value.startsWith('--')) {
        throw new RangeError('--input requires a manifest path.');
      }
      options.inputs ??= [];
      options.inputs.push(value);
      index += 1;
    } else if (value === undefined || value.startsWith('--')) options[name] = true;
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
  const binary = join(directory, 'perfect-classic-policy');
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
    throw new Error(`Classic policy compiler failed.\n${result.stderr || result.stdout}`);
  }
  return { compiler, binary, warnings: result.stderr.trim() };
}

function parseJsonLines(output) {
  return output.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function createGeometry(rows, columns, connect) {
  const stride = rows + 1;
  const columnBits = (1n << BigInt(rows)) - 1n;
  const columnWithSentinel = (1n << BigInt(stride)) - 1n;
  const bottomMasks = Array.from(
    { length: columns },
    (_, column) => 1n << BigInt(column * stride),
  );
  const columnMasks = bottomMasks.map((bottom) => bottom * columnBits);
  const bottomMask = bottomMasks.reduce((mask, bit) => mask | bit, 0n);
  const boardMask = bottomMask * columnBits;
  const centre = (columns - 1) / 2;
  const columnOrder = Array.from({ length: columns }, (_, column) => column)
    .sort((first, second) => (
      Math.abs(first - centre) - Math.abs(second - centre) || first - second
    ));
  return {
    rows,
    columns,
    connect,
    stride,
    cellCount: rows * columns,
    columnWithSentinel,
    bottomMasks,
    columnMasks,
    bottomMask,
    boardMask,
    columnOrder,
    directions: [1, stride - 1, stride, stride + 1],
  };
}

function possibleMoves(geometry, mask) {
  return (mask + geometry.bottomMask) & geometry.boardMask;
}

function moveForColumn(geometry, mask, column) {
  if (!Number.isInteger(column) || column < 0 || column >= geometry.columns) return 0n;
  return (mask + geometry.bottomMasks[column]) & geometry.columnMasks[column];
}

function play(position, move) {
  return {
    current: position.current ^ position.mask,
    mask: position.mask | move,
    moves: position.moves + 1,
  };
}

function hasAlignment(geometry, bits) {
  for (const direction of geometry.directions) {
    const shift = BigInt(direction);
    let run = bits;
    for (let offset = 1; offset < geometry.connect && run !== 0n; offset += 1) {
      run &= bits >> (BigInt(offset) * shift);
    }
    if (run !== 0n) return true;
  }
  return false;
}

function mirrorBits(geometry, bits) {
  let mirrored = 0n;
  for (let column = 0; column < geometry.columns; column += 1) {
    const group = (bits >> BigInt(column * geometry.stride)) & geometry.columnWithSentinel;
    mirrored |= group << BigInt((geometry.columns - 1 - column) * geometry.stride);
  }
  return mirrored;
}

function canonicalize(geometry, position) {
  const normal = position.current + position.mask;
  const mirroredCurrent = mirrorBits(geometry, position.current);
  const mirroredMask = mirrorBits(geometry, position.mask);
  const mirrored = mirroredCurrent + mirroredMask;
  return normal <= mirrored
    ? { position, key: normal }
    : {
      position: { current: mirroredCurrent, mask: mirroredMask, moves: position.moves },
      key: mirrored,
    };
}

function immediateWinningMoves(geometry, position, pieces = position.current) {
  const possible = possibleMoves(geometry, position.mask);
  let winning = 0n;
  for (const column of geometry.columnOrder) {
    const move = moveForColumn(geometry, position.mask, column);
    if ((possible & move) !== 0n && hasAlignment(geometry, pieces | move)) winning |= move;
  }
  return winning;
}

function nonLosingMoves(geometry, position) {
  let possible = possibleMoves(geometry, position.mask);
  const opponent = position.current ^ position.mask;
  const opponentWins = immediateWinningMoves(geometry, position, opponent);
  if (opponentWins !== 0n) {
    if ((opponentWins & (opponentWins - 1n)) !== 0n) return 0n;
    possible = opponentWins;
  }

  let safe = 0n;
  for (const column of geometry.columnOrder) {
    const move = moveForColumn(geometry, position.mask, column);
    if ((possible & move) === 0n) continue;
    if (hasAlignment(geometry, position.current | move)) {
      safe |= move;
      continue;
    }
    const child = play(position, move);
    if (immediateWinningMoves(geometry, child) === 0n) safe |= move;
  }
  return safe;
}

/**
 * Independently re-solves policy handoff states with a fixed-size direct-mapped
 * table. Replacement collisions can cost work but cannot produce false hits,
 * so memory use is deterministic and an unfinished search fails closed.
 */
class IndependentExactSolver {
  constructor(geometry, options = {}) {
    this.geometry = geometry;
    this.maximumNodes = options.maximumNodes ?? Infinity;
    this.tableBits = options.tableBits ?? 22;
    if (!Number.isInteger(this.tableBits) || this.tableBits < 8 || this.tableBits > 25) {
      throw new RangeError('Independent verification table bits must be from 8 through 25.');
    }
    this.size = 2 ** this.tableBits;
    this.indexMask = BigInt(this.size - 1);
    this.keys = new BigUint64Array(this.size);
    this.lowerBounds = new Int8Array(this.size);
    this.upperBounds = new Int8Array(this.size);
    this.flags = new Uint8Array(this.size);
    this.nodes = 0;
    this.hits = 0;
    this.stores = 0;
    this.collisions = 0;
  }

  index(key) {
    return Number((key ^ (key >> 23n) ^ (key >> 41n)) & this.indexMask);
  }

  probe(key) {
    const index = this.index(key);
    if (this.keys[index] !== key + 1n) return null;
    this.hits += 1;
    return {
      lower: (this.flags[index] & 1) === 0 ? -2 : this.lowerBounds[index],
      upper: (this.flags[index] & 2) === 0 ? 2 : this.upperBounds[index],
    };
  }

  prepare(key) {
    const index = this.index(key);
    const stored = this.keys[index];
    if (stored !== 0n && stored !== key + 1n) this.collisions += 1;
    if (stored !== key + 1n) {
      this.keys[index] = key + 1n;
      this.flags[index] = 0;
    }
    return index;
  }

  storeLower(key, score) {
    const index = this.prepare(key);
    if ((this.flags[index] & 1) !== 0 && score <= this.lowerBounds[index]) return;
    this.lowerBounds[index] = score;
    this.flags[index] |= 1;
    this.stores += 1;
  }

  storeUpper(key, score) {
    const index = this.prepare(key);
    if ((this.flags[index] & 2) !== 0 && score >= this.upperBounds[index]) return;
    this.upperBounds[index] = score;
    this.flags[index] |= 2;
    this.stores += 1;
  }

  visit() {
    this.nodes += 1;
    if (this.nodes > this.maximumNodes) {
      const error = new RangeError('Independent classic policy replay exceeded its node limit.');
      error.code = 'CLASSIC_POLICY_VERIFY_NODE_LIMIT';
      error.nodes = this.nodes;
      throw error;
    }
  }

  search(rawPosition, alpha, beta) {
    this.visit();
    const canonical = canonicalize(this.geometry, rawPosition);
    const position = canonical.position;
    const possible = possibleMoves(this.geometry, position.mask);
    if (possible === 0n) return DRAW;
    if (immediateWinningMoves(this.geometry, position) !== 0n) return WIN;

    const safe = nonLosingMoves(this.geometry, position);
    if (safe === 0n) return LOSS;
    if (position.moves >= this.geometry.cellCount - 2) return DRAW;

    const cached = this.probe(canonical.key);
    if (cached) {
      if (cached.lower >= beta) return cached.lower;
      if (cached.upper <= alpha) return cached.upper;
      alpha = Math.max(alpha, cached.lower);
      beta = Math.min(beta, cached.upper);
      if (alpha >= beta) return alpha;
    }

    for (const column of this.geometry.columnOrder) {
      const move = moveForColumn(this.geometry, position.mask, column);
      if ((safe & move) === 0n) continue;
      const childValue = this.search(play(position, move), -beta, -alpha);
      const value = childValue === DRAW ? DRAW : -childValue;
      if (value >= beta) {
        this.storeLower(canonical.key, value);
        return value;
      }
      if (value > alpha) alpha = value;
    }
    this.storeUpper(canonical.key, alpha);
    return alpha;
  }

  solve(position) {
    let minimum = LOSS;
    let maximum = WIN;
    while (minimum < maximum) {
      const middle = minimum + Math.floor((maximum - minimum) / 2);
      const value = this.search(position, middle, middle + 1);
      if (value <= middle) maximum = value;
      else minimum = value;
    }
    return minimum;
  }
}

function moveMaskColumn(moveMask, columns) {
  if (!Number.isInteger(moveMask) || moveMask <= 0
      || (moveMask & (moveMask - 1)) !== 0
      || (moveMask & ~((1 << columns) - 1)) !== 0) return -1;
  for (let column = 0; column < columns; column += 1) {
    if ((moveMask & (1 << column)) !== 0) return column;
  }
  return -1;
}

export function replayPerfectClassicPolicy(policy, options = {}) {
  const geometry = createGeometry(policy.rows, policy.columns, policy.connect);
  const exact = new IndependentExactSolver(geometry, {
    maximumNodes: options.maximumExactNodes ?? Infinity,
    tableBits: options.exactTableBits ?? 22,
  });
  const usedPolicy = new Set();
  const values = new Map();
  const visiting = new Set();
  let closureStates = 0;
  let handoffStates = 0;
  let terminalAiWins = 0;
  let terminalAiLosses = 0;
  let terminalDraws = 0;

  const evaluate = (rawPosition, aiTurn) => {
    const canonical = canonicalize(geometry, rawPosition);
    const stateKey = canonical.key | (aiTurn ? AI_TURN_BIT : 0n);
    const cached = values.get(stateKey);
    if (cached !== undefined) return cached;
    if (visiting.has(stateKey)) throw new Error('Classic policy closure unexpectedly contains a cycle.');
    visiting.add(stateKey);
    closureStates += 1;
    const position = canonical.position;
    const remaining = geometry.cellCount - position.moves;
    let value;

    if (aiTurn && remaining <= policy.handoffRemaining) {
      handoffStates += 1;
      value = exact.solve(position);
    } else if (aiTurn) {
      const record = policy.lookupKey(canonical.key);
      if (!record) {
        throw new Error(`Perfect classic policy is missing reachable key ${canonical.key}.`);
      }
      usedPolicy.add(canonical.key);
      const column = moveMaskColumn(record.moveMask, geometry.columns);
      const move = moveForColumn(geometry, position.mask, column);
      if (move === 0n) throw new Error('Perfect classic policy selects an illegal move.');
      if (hasAlignment(geometry, position.current | move)) {
        terminalAiWins += 1;
        value = WIN;
      } else {
        const child = play(position, move);
        if (possibleMoves(geometry, child.mask) === 0n) {
          terminalDraws += 1;
          value = DRAW;
        } else value = evaluate(child, false);
      }
      if (record.outcome !== value) {
        throw new Error(
          `Perfect classic policy outcome mismatch at key ${canonical.key}: `
          + `${record.outcome} instead of ${value}.`,
        );
      }
    } else {
      const possible = possibleMoves(geometry, position.mask);
      if (possible === 0n) {
        terminalDraws += 1;
        value = DRAW;
      } else {
        value = WIN;
        for (const column of geometry.columnOrder) {
          const move = moveForColumn(geometry, position.mask, column);
          if ((possible & move) === 0n) continue;
          let candidate;
          if (hasAlignment(geometry, position.current | move)) {
            terminalAiLosses += 1;
            candidate = LOSS;
          } else {
            const child = play(position, move);
            if (possibleMoves(geometry, child.mask) === 0n) {
              terminalDraws += 1;
              candidate = DRAW;
            } else candidate = evaluate(child, true);
          }
          if (candidate < value) value = candidate;
          if (value === LOSS) break;
        }
      }
    }

    visiting.delete(stateKey);
    values.set(stateKey, value);
    return value;
  };

  const rootValue = evaluate(
    { current: 0n, mask: 0n, moves: 0 },
    policy.role === PERFECT_CLASSIC_ROLE_FIRST,
  );
  if (rootValue !== policy.rootValue) {
    throw new Error(
      `Perfect classic policy root value mismatch: ${rootValue} instead of ${policy.rootValue}.`,
    );
  }
  if (usedPolicy.size !== policy.entryCount) {
    throw new Error(
      `Perfect classic policy has ${policy.entryCount - usedPolicy.size} unreachable record(s).`,
    );
  }
  if (closureStates !== policy.closureStates) {
    throw new Error(
      `Perfect classic policy closure mismatch: ${closureStates} instead of ${policy.closureStates}.`,
    );
  }
  return {
    format: 'connect4-perfect-classic-policy-replay-v1',
    rows: policy.rows,
    columns: policy.columns,
    connect: policy.connect,
    role: policy.role,
    handoffRemaining: policy.handoffRemaining,
    rootValue,
    entryCount: policy.entryCount,
    closureStates,
    handoffStates,
    terminalAiWins,
    terminalAiLosses,
    terminalDraws,
    exactNodes: exact.nodes,
    exactTableHits: exact.hits,
    exactTableStores: exact.stores,
    exactTableCollisions: exact.collisions,
    exactTableBits: exact.tableBits,
  };
}

async function hashFile(path) {
  const buffer = await readFile(path);
  return {
    bytes: buffer.length,
    sha256: createHash('sha256').update(buffer).digest('hex'),
  };
}

async function publishedRootValue(rows, columns, connect) {
  if (connect !== 4 || rows < 4 || columns < 4) return null;
  const reference = JSON.parse(await readFile(ROOT_VALUES, 'utf8'));
  return reference.boards.find((entry) => (
    entry.rows === rows && entry.columns === columns
  ))?.value ?? null;
}

function roleSelection(value) {
  if (value === undefined || value === 'both') {
    return [PERFECT_CLASSIC_ROLE_FIRST, PERFECT_CLASSIC_ROLE_SECOND];
  }
  const role = Number.parseInt(String(value), 10);
  if (role !== PERFECT_CLASSIC_ROLE_FIRST && role !== PERFECT_CLASSIC_ROLE_SECOND) {
    throw new RangeError('role must be 1, 2, or both.');
  }
  return [role];
}

async function generatePolicies(binary, options) {
  const rows = integerOption(options.rows, 6, 'rows', 1, 7);
  const columns = integerOption(options.columns, 7, 'columns', 1, 7);
  const connect = integerOption(options.connect, 4, 'connect', 1, Math.max(rows, columns));
  const cellCount = rows * columns;
  const handoffRemaining = integerOption(
    options.handoff_remaining,
    Math.min(24, cellCount),
    'handoff-remaining',
    0,
    cellCount,
  );
  const tableBits = integerOption(options.table_bits, 24, 'table-bits', 8, 27);
  const verifyTableBits = integerOption(
    options.verify_table_bits,
    22,
    'verify-table-bits',
    8,
    25,
  );
  const maximumNodes = integerOption(
    options.maximum_nodes,
    0,
    'maximum-nodes',
    0,
    Number.MAX_SAFE_INTEGER,
  );
  const maximumStates = integerOption(
    options.maximum_states,
    100_000_000,
    'maximum-states',
    1,
    Number.MAX_SAFE_INTEGER,
  );
  const maximumExactNodes = options.maximum_verify_nodes === undefined
    ? Infinity
    : integerOption(
      options.maximum_verify_nodes,
      0,
      'maximum-verify-nodes',
      1,
      Number.MAX_SAFE_INTEGER,
    );
  const output = resolve(options.output ?? join(
    ROOT,
    'generated',
    `perfect-classic-${rows}x${columns}-c${connect}`,
  ));
  await mkdir(output, { recursive: true });
  const expectedFirstValue = options.expected_root === undefined
    ? await publishedRootValue(rows, columns, connect)
    : integerOption(options.expected_root, 0, 'expected-root', -1, 1);
  const policies = [];

  for (const role of roleSelection(options.role)) {
    const filename = `${rows}x${columns}-c${connect}-role${role}.bin`;
    const path = join(output, filename);
    const result = await run(binary, [
      'generate',
      '--rows', String(rows),
      '--columns', String(columns),
      '--connect', String(connect),
      '--role', String(role),
      '--handoff-remaining', String(handoffRemaining),
      '--table-bits', String(tableBits),
      '--maximum-nodes', String(maximumNodes),
      '--maximum-states', String(maximumStates),
      '--output', path,
    ]);
    if (result.code !== 0) {
      throw new Error(`Classic policy generation failed for role ${role}.\n${result.stderr || result.stdout}`);
    }
    const summaries = parseJsonLines(result.stdout);
    const summary = summaries.at(-1);
    if (!summary || summary.format !== 'connect4-perfect-classic-policy-summary-v1') {
      throw new Error(`Classic policy generator returned no summary for role ${role}.`);
    }
    const bytes = await readFile(path);
    const policy = decodePerfectClassicPolicy(bytes, { rows, columns, connect, role });
    const expectedRoleValue = expectedFirstValue === null
      ? policy.rootValue
      : role === PERFECT_CLASSIC_ROLE_FIRST ? expectedFirstValue : -expectedFirstValue;
    if (policy.rootValue !== expectedRoleValue) {
      throw new Error(
        `Policy root value mismatch for role ${role}: `
        + `${policy.rootValue} instead of ${expectedRoleValue}.`,
      );
    }
    const replay = replayPerfectClassicPolicy(policy, {
      maximumExactNodes,
      exactTableBits: verifyTableBits,
    });
    const digest = await hashFile(path);
    policies.push({
      rows,
      columns,
      connect,
      role,
      handoffRemaining: policy.handoffRemaining,
      rootValue: policy.rootValue,
      entryCount: policy.entryCount,
      closureStates: policy.closureStates,
      file: `./${filename}`,
      ...digest,
      generator: summary,
      replay,
    });
  }

  const manifest = {
    format: 'connect4-perfect-classic-manifest-v1',
    generatedAt: new Date().toISOString(),
    sourceSha256: createHash('sha256').update(await readFile(SOURCE)).digest('hex'),
    policies,
  };
  await writeFile(join(output, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  return { output, manifest };
}

async function verifyPolicyManifest(path, options = {}) {
  const manifestPath = resolve(path);
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  if (manifest?.format !== 'connect4-perfect-classic-manifest-v1'
      || !Array.isArray(manifest.policies)) {
    throw new Error('Perfect classic policy manifest format is invalid.');
  }
  const directory = dirname(manifestPath);
  const maximumExactNodes = options.maximum_verify_nodes === undefined
    ? Infinity
    : integerOption(
      options.maximum_verify_nodes,
      0,
      'maximum-verify-nodes',
      1,
      Number.MAX_SAFE_INTEGER,
    );
  const verifyTableBits = integerOption(
    options.verify_table_bits,
    22,
    'verify-table-bits',
    8,
    25,
  );
  const replay = [];
  for (const entry of manifest.policies) {
    const pathToPolicy = resolve(directory, entry.file);
    const bytes = await readFile(pathToPolicy);
    const digest = await hashFile(pathToPolicy);
    if (digest.bytes !== entry.bytes || digest.sha256 !== entry.sha256) {
      throw new Error(`Perfect classic policy hash mismatch for ${entry.file}.`);
    }
    const policy = decodePerfectClassicPolicy(bytes, entry);
    if (policy.handoffRemaining !== entry.handoffRemaining
        || policy.rootValue !== entry.rootValue
        || policy.entryCount !== entry.entryCount
        || policy.closureStates !== entry.closureStates) {
      throw new Error(`Perfect classic policy metadata mismatch for ${entry.file}.`);
    }
    replay.push(replayPerfectClassicPolicy(policy, {
      maximumExactNodes,
      exactTableBits: verifyTableBits,
    }));
  }
  return { manifestPath, replay };
}

async function mergeManifests(inputPaths, output) {
  if (!Array.isArray(inputPaths) || inputPaths.length === 0) {
    throw new RangeError('At least one --input manifest is required.');
  }
  const outputPath = resolve(output);
  const outputDirectory = dirname(outputPath);
  await mkdir(outputDirectory, { recursive: true });
  const policies = [];
  const keys = new Set();
  for (const input of inputPaths) {
    const path = resolve(input);
    const manifest = JSON.parse(await readFile(path, 'utf8'));
    if (manifest?.format !== 'connect4-perfect-classic-manifest-v1'
        || !Array.isArray(manifest.policies)) {
      throw new Error(`Invalid perfect classic manifest: ${path}`);
    }
    for (const entry of manifest.policies) {
      const key = `${entry.rows}x${entry.columns}:c${entry.connect}:r${entry.role}`;
      if (keys.has(key)) throw new Error(`Duplicate perfect classic policy ${key}.`);
      keys.add(key);
      const source = resolve(dirname(path), entry.file);
      const filename = `${entry.rows}x${entry.columns}-c${entry.connect}-role${entry.role}.bin`;
      const target = join(outputDirectory, filename);
      await writeFile(target, await readFile(source));
      policies.push({ ...entry, file: `./${filename}` });
    }
  }
  policies.sort((first, second) => (
    first.rows - second.rows
    || first.columns - second.columns
    || first.connect - second.connect
    || first.role - second.role
  ));
  const manifest = {
    format: 'connect4-perfect-classic-manifest-v1',
    generatedAt: new Date().toISOString(),
    policies,
  };
  await writeFile(outputPath, `${JSON.stringify(manifest, null, 2)}\n`);
  return manifest;
}

async function verifySmall(binary, temporary) {
  const native = await run(binary, ['verify']);
  if (native.code !== 0) throw new Error(`Native policy verification failed.\n${native.stderr}`);
  const nativeRecords = parseJsonLines(native.stdout);
  if (nativeRecords.length !== 6) {
    throw new Error('Native policy verification returned the wrong case count.');
  }
  const generated = [];
  for (const [rows, columns, connect, expected] of [
    [2, 2, 2, WIN],
    [3, 3, 3, DRAW],
  ]) {
    const result = await generatePolicies(binary, {
      rows,
      columns,
      connect,
      handoff_remaining: 0,
      role: 'both',
      expected_root: expected,
      table_bits: 16,
      verify_table_bits: 14,
      maximum_nodes: 10_000_000,
      maximum_states: 1_000_000,
      output: join(temporary, `${rows}x${columns}-c${connect}`),
    });
    generated.push(...result.manifest.policies.map((entry) => entry.replay));
  }
  return { native: nativeRecords, replay: generated };
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const temporary = await mkdtemp(join(tmpdir(), 'connect4-classic-policy-'));
  try {
    if (options.command === 'merge-manifests') {
      if (!options.output || options.output === true) throw new RangeError('--output is required.');
      const manifest = await mergeManifests(options.inputs, options.output);
      process.stdout.write(`${JSON.stringify(manifest, null, 2)}\n`);
      return;
    }
    if (options.command === 'verify-reference') {
      if (!options.reference || options.reference === true) throw new RangeError('--reference is required.');
      const verified = await verifyPolicyManifest(options.reference, options);
      process.stdout.write(`${JSON.stringify(verified, null, 2)}\n`);
      return;
    }

    const compiled = await compile(temporary);
    if (compiled.warnings) process.stderr.write(`${compiled.warnings}\n`);
    if (options.command === 'verify') {
      const verified = await verifySmall(compiled.binary, temporary);
      process.stdout.write(`${JSON.stringify({ compiler: compiled.compiler, ...verified }, null, 2)}\n`);
      return;
    }
    if (options.command === 'generate') {
      const generated = await generatePolicies(compiled.binary, options);
      process.stdout.write(`${JSON.stringify({
        compiler: compiled.compiler,
        output: generated.output,
        manifest: generated.manifest,
      }, null, 2)}\n`);
      return;
    }
    throw new RangeError(`Unknown command: ${options.command}`);
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

await main();
