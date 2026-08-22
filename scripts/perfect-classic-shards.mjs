#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { decodePerfectClassicPolicy } from '../src/perfect-classic-policy.js';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const SOURCE = join(ROOT, 'native', 'perfect-classic-policy-shard.cpp');
const MAGIC = 'C4VPOL1\0';
const HEADER_SIZE = 24;
const RECORD_SIZE = 10;
const AI_TURN_BIT = 1n << 63n;

function parseArguments(argv) {
  const options = { command: argv[0] ?? 'help', inputs: [] };
  for (let index = 1; index < argv.length; index += 1) {
    const argument = argv[index];
    if (!argument.startsWith('--')) throw new RangeError(`Unexpected argument: ${argument}`);
    const name = argument.slice(2).replaceAll('-', '_');
    const value = argv[index + 1];
    if (name === 'input') {
      if (value === undefined || value.startsWith('--')) throw new RangeError('--input requires a path.');
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

function bigintOption(value, fallback, label) {
  try {
    const selected = value === undefined ? fallback : BigInt(String(value));
    if (selected < 0n || selected > 0xffffffffffffffffn) throw new Error();
    return selected;
  } catch {
    throw new RangeError(`${label} must be an unsigned 64-bit integer.`);
  }
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
  throw new Error('A C++20 compiler is required.');
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
  const binary = join(directory, 'perfect-classic-policy-shard');
  const result = await run(compiler, [
    '-std=c++20', '-O3', '-Wall', '-Wextra', '-Wpedantic', SOURCE, '-o', binary,
  ]);
  if (result.code !== 0) throw new Error(`Shard compiler failed.\n${result.stderr || result.stdout}`);
  return { binary, compiler, warnings: result.stderr.trim() };
}

function createGeometry(rows, columns, connect) {
  const stride = rows + 1;
  const columnBits = (1n << BigInt(rows)) - 1n;
  const columnWithSentinel = (1n << BigInt(stride)) - 1n;
  const bottomMasks = Array.from({ length: columns }, (_, column) => 1n << BigInt(column * stride));
  const columnMasks = bottomMasks.map((bottom) => bottom * columnBits);
  const bottomMask = bottomMasks.reduce((mask, bit) => mask | bit, 0n);
  const boardMask = bottomMask * columnBits;
  const centre = (columns - 1) / 2;
  const columnOrder = Array.from({ length: columns }, (_, column) => column)
    .sort((first, second) => Math.abs(first - centre) - Math.abs(second - centre) || first - second);
  return {
    rows, columns, connect, stride, cellCount: rows * columns,
    columnBits, columnWithSentinel, bottomMasks, columnMasks, bottomMask, boardMask,
    columnOrder, directions: [1, stride - 1, stride, stride + 1],
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
  return { current: position.current ^ position.mask, mask: position.mask | move, moves: position.moves + 1 };
}

function hasAlignment(geometry, bits) {
  for (const direction of geometry.directions) {
    let run = bits;
    for (let offset = 1; offset < geometry.connect && run !== 0n; offset += 1) {
      run &= bits >> BigInt(offset * direction);
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
    : { position: { current: mirroredCurrent, mask: mirroredMask, moves: position.moves }, key: mirrored };
}

function stateObject(position) {
  return { current: String(position.current), mask: String(position.mask), moves: position.moves };
}

function stateIdentity(position) {
  return `${position.current}:${position.mask}:${position.moves}`;
}

function moveMaskColumn(moveMask, columns) {
  if (!Number.isInteger(moveMask) || moveMask <= 0 || (moveMask & (moveMask - 1)) !== 0
      || (moveMask & ~((1 << columns) - 1)) !== 0) return -1;
  for (let column = 0; column < columns; column += 1) {
    if ((moveMask & (1 << column)) !== 0) return column;
  }
  return -1;
}

function parseRecords(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const entryCount = view.getUint32(16, true);
  const records = [];
  for (let index = 0; index < entryCount; index += 1) {
    const offset = HEADER_SIZE + index * RECORD_SIZE;
    records.push({
      key: view.getBigUint64(offset, true),
      moveMask: view.getUint8(offset + 8),
      outcome: view.getInt8(offset + 9),
    });
  }
  return records;
}

function writePolicy({ rows, columns, connect, role, handoffRemaining, rootValue, closureStates, records }) {
  const bytes = Buffer.alloc(HEADER_SIZE + records.length * RECORD_SIZE);
  bytes.write(MAGIC, 0, 'binary');
  bytes[8] = 1;
  bytes[9] = rows;
  bytes[10] = columns;
  bytes[11] = connect;
  bytes[12] = role;
  bytes[13] = handoffRemaining;
  bytes[14] = RECORD_SIZE;
  bytes.writeInt8(rootValue, 15);
  bytes.writeUInt32LE(records.length, 16);
  bytes.writeUInt32LE(closureStates, 20);
  records.forEach((record, index) => {
    const offset = HEADER_SIZE + index * RECORD_SIZE;
    bytes.writeBigUInt64LE(record.key, offset);
    bytes[offset + 8] = record.moveMask;
    bytes.writeInt8(record.outcome, offset + 9);
  });
  return bytes;
}

async function digest(path) {
  const bytes = await readFile(path);
  return { bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') };
}

function geometryOptions(options) {
  const rows = integerOption(options.rows, 7, 'rows', 1, 7);
  const columns = integerOption(options.columns, 7, 'columns', 1, 7);
  const connect = integerOption(options.connect, 4, 'connect', 1, Math.max(rows, columns));
  const role = integerOption(options.role, 1, 'role', 1, 2);
  return { rows, columns, connect, role, geometry: createGeometry(rows, columns, connect) };
}

async function initialFrontier(options) {
  const { rows, columns, connect, role, geometry } = geometryOptions(options);
  const rootValue = integerOption(options.root_value, 0, 'root-value', -1, 1);
  const states = new Map();
  const enqueue = (position) => {
    const canonical = canonicalize(geometry, position).position;
    states.set(stateIdentity(canonical), canonical);
  };

  if (role === 2) {
    const empty = { current: 0n, mask: 0n, moves: 0 };
    for (const column of geometry.columnOrder) {
      const move = moveForColumn(geometry, empty.mask, column);
      if (move !== 0n) enqueue(play(empty, move));
    }
  } else {
    const rootColumn = integerOption(options.root_column, Math.floor(columns / 2), 'root-column', 0, columns - 1);
    const empty = { current: 0n, mask: 0n, moves: 0 };
    const rootMove = moveForColumn(geometry, empty.mask, rootColumn);
    const opponent = play(empty, rootMove);
    for (const column of geometry.columnOrder) {
      const reply = moveForColumn(geometry, opponent.mask, column);
      if (reply === 0n || hasAlignment(geometry, opponent.current | reply)) continue;
      enqueue(play(opponent, reply));
    }
  }

  const include = [...states.values()]
    .sort((a, b) => (a.current + a.mask < b.current + b.mask ? -1 : 1))
    .map((position, index) => ({ id: String(index).padStart(3, '0'), ...stateObject(position) }));
  const result = { format: 'connect4-perfect-classic-shard-matrix-v1', rows, columns, connect, role, rootValue, include };
  if (options.output && options.output !== true) {
    const output = resolve(String(options.output));
    await mkdir(dirname(output), { recursive: true });
    await writeFile(output, `${JSON.stringify(result, null, 2)}\n`);
  }
  return result;
}

async function generateFragment(options) {
  const { rows, columns, connect, role, geometry } = geometryOptions(options);
  const start = {
    current: bigintOption(options.start_current, 0n, 'start-current'),
    mask: bigintOption(options.start_mask, 0n, 'start-mask'),
    moves: integerOption(options.start_moves, 0, 'start-moves', 0, geometry.cellCount),
  };
  const handoffRemaining = integerOption(options.handoff_remaining, 24, 'handoff-remaining', 0, geometry.cellCount);
  const frontierRemaining = options.frontier_remaining === undefined
    ? null
    : integerOption(options.frontier_remaining, 0, 'frontier-remaining', 0, geometry.cellCount);
  const tableBits = integerOption(options.table_bits, 26, 'table-bits', 8, 27);
  const maximumNodes = integerOption(options.maximum_nodes, 0, 'maximum-nodes', 0, Number.MAX_SAFE_INTEGER);
  const maximumStates = integerOption(options.maximum_states, 100_000_000, 'maximum-states', 1, Number.MAX_SAFE_INTEGER);
  if (!options.output || options.output === true) throw new RangeError('--output directory is required.');
  const output = resolve(String(options.output));
  await mkdir(output, { recursive: true });
  const temporary = await mkdtemp(join(tmpdir(), 'perfect-classic-shard-'));
  try {
    const compiled = await compile(temporary);
    if (compiled.warnings) process.stderr.write(`${compiled.warnings}\n`);
    const filename = 'fragment.bin';
    const path = join(output, filename);
    const args = [
      'generate', '--rows', String(rows), '--columns', String(columns), '--connect', String(connect),
      '--role', String(role), '--start-current', String(start.current), '--start-mask', String(start.mask),
      '--start-moves', String(start.moves), '--handoff-remaining', String(handoffRemaining),
      '--table-bits', String(tableBits), '--maximum-nodes', String(maximumNodes),
      '--maximum-states', String(maximumStates), '--output', path,
    ];
    if (frontierRemaining !== null) args.push('--frontier-remaining', String(frontierRemaining));
    const result = await run(compiled.binary, args);
    if (result.code !== 0) throw new Error(`Fragment generation failed.\n${result.stderr || result.stdout}`);
    const lines = result.stdout.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
    const summary = lines.find((entry) => entry.format === 'connect4-perfect-classic-policy-summary-v1');
    const frontier = lines.filter((entry) => entry.format === 'connect4-perfect-classic-fragment-frontier-v1')
      .map((entry) => ({ current: entry.current, mask: entry.mask, moves: entry.moves }));
    if (!summary) throw new Error('Fragment generator returned no summary.');
    const bytes = await readFile(path);
    const policy = decodePerfectClassicPolicy(bytes, { rows, columns, connect, role });
    if (policy.entryCount !== summary.entryCount || policy.closureStates !== summary.closureStates
        || policy.rootValue !== summary.rootValue || policy.handoffRemaining !== handoffRemaining
        || frontier.length !== summary.frontierStates) {
      throw new Error('Fragment binary metadata does not match its summary.');
    }
    const fileDigest = await digest(path);
    const manifest = {
      format: 'connect4-perfect-classic-fragment-manifest-v1',
      rows, columns, connect, role, handoffRemaining,
      start: stateObject(start),
      rootValue: policy.rootValue,
      entryCount: policy.entryCount,
      closureStates: policy.closureStates,
      file: `./${filename}`,
      ...fileDigest,
      summary,
      frontier,
    };
    await writeFile(join(output, 'fragment.json'), `${JSON.stringify(manifest, null, 2)}\n`);
    return { output, manifest };
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

async function readFragment(path) {
  const manifestPath = resolve(path);
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  if (manifest?.format !== 'connect4-perfect-classic-fragment-manifest-v1') {
    throw new Error(`Invalid fragment manifest: ${manifestPath}`);
  }
  const policyPath = resolve(dirname(manifestPath), manifest.file);
  const bytes = await readFile(policyPath);
  const actual = { bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') };
  if (actual.bytes !== manifest.bytes || actual.sha256 !== manifest.sha256) {
    throw new Error(`Fragment hash mismatch: ${manifestPath}`);
  }
  decodePerfectClassicPolicy(bytes, manifest);
  return { manifestPath, manifest, bytes, records: parseRecords(bytes) };
}

async function collectFrontier(options) {
  if (!options.inputs.length) throw new RangeError('At least one --input fragment is required.');
  const fragments = await Promise.all(options.inputs.map(readFragment));
  const first = fragments[0].manifest;
  const states = new Map();
  for (const { manifest } of fragments) {
    for (const field of ['rows', 'columns', 'connect', 'role', 'handoffRemaining']) {
      if (manifest[field] !== first[field]) throw new Error(`Fragment ${field} mismatch.`);
    }
    for (const raw of manifest.frontier) {
      const position = { current: BigInt(raw.current), mask: BigInt(raw.mask), moves: raw.moves };
      states.set(stateIdentity(position), position);
    }
  }
  const include = [...states.values()]
    .sort((a, b) => (a.current + a.mask < b.current + b.mask ? -1 : 1))
    .map((position, index) => ({ id: String(index).padStart(4, '0'), ...stateObject(position) }));
  const result = {
    format: 'connect4-perfect-classic-shard-matrix-v1',
    rows: first.rows, columns: first.columns, connect: first.connect, role: first.role,
    handoffRemaining: first.handoffRemaining, fragmentCount: fragments.length, include,
  };
  if (options.output && options.output !== true) {
    const output = resolve(String(options.output));
    await mkdir(dirname(output), { recursive: true });
    await writeFile(output, `${JSON.stringify(result, null, 2)}\n`);
  }
  return result;
}

function closureFromRecords({ geometry, role, handoffRemaining, records }) {
  const queue = [];
  const seen = new Set();
  const used = new Set();
  const stats = {
    closureStates: 0, aiStates: 0, opponentStates: 0, handoffStates: 0,
    terminalAiWins: 0, terminalAiLosses: 0, terminalDraws: 0, revisitedStates: 0,
  };
  const enqueue = (raw, aiTurn) => {
    const canonical = canonicalize(geometry, raw);
    const key = canonical.key | (aiTurn ? AI_TURN_BIT : 0n);
    if (seen.has(key)) { stats.revisitedStates += 1; return; }
    seen.add(key);
    queue.push({ position: canonical.position, key: canonical.key, aiTurn });
  };
  enqueue({ current: 0n, mask: 0n, moves: 0 }, role === 1);
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const state = queue[cursor];
    stats.closureStates += 1;
    const remaining = geometry.cellCount - state.position.moves;
    if (state.aiTurn && remaining <= handoffRemaining) {
      stats.handoffStates += 1;
      continue;
    }
    const possible = possibleMoves(geometry, state.position.mask);
    if (possible === 0n) { stats.terminalDraws += 1; continue; }
    if (state.aiTurn) {
      stats.aiStates += 1;
      const record = records.get(state.key);
      if (!record) throw new Error(`Assembled policy is missing reachable key ${state.key}.`);
      used.add(state.key);
      const column = moveMaskColumn(record.moveMask, geometry.columns);
      const move = moveForColumn(geometry, state.position.mask, column);
      if (move === 0n) throw new Error(`Assembled policy selects illegal column ${column}.`);
      if (hasAlignment(geometry, state.position.current | move)) {
        stats.terminalAiWins += 1;
        continue;
      }
      const child = play(state.position, move);
      if (possibleMoves(geometry, child.mask) === 0n) stats.terminalDraws += 1;
      else enqueue(child, false);
    } else {
      stats.opponentStates += 1;
      for (const column of geometry.columnOrder) {
        const move = moveForColumn(geometry, state.position.mask, column);
        if ((possible & move) === 0n) continue;
        if (hasAlignment(geometry, state.position.current | move)) {
          stats.terminalAiLosses += 1;
          continue;
        }
        const child = play(state.position, move);
        if (possibleMoves(geometry, child.mask) === 0n) stats.terminalDraws += 1;
        else enqueue(child, true);
      }
    }
  }
  return { used, stats };
}

async function assemble(options) {
  if (!options.inputs.length) throw new RangeError('At least one --input fragment is required.');
  if (!options.output || options.output === true) throw new RangeError('--output directory is required.');
  const fragments = await Promise.all(options.inputs.map(readFragment));
  const first = fragments[0].manifest;
  const { rows, columns, connect, role, handoffRemaining } = first;
  const rootValue = integerOption(options.root_value, role === 1 ? first.rootValue : -first.rootValue, 'root-value', -1, 1);
  const geometry = createGeometry(rows, columns, connect);
  const records = new Map();
  let duplicateChoices = 0;
  for (const fragment of fragments) {
    const manifest = fragment.manifest;
    for (const field of ['rows', 'columns', 'connect', 'role', 'handoffRemaining']) {
      if (manifest[field] !== first[field]) throw new Error(`Fragment ${field} mismatch.`);
    }
    for (const record of fragment.records) {
      const existing = records.get(record.key);
      if (!existing) records.set(record.key, record);
      else if (existing.outcome !== record.outcome) {
        throw new Error(`Fragment outcome conflict at key ${record.key}.`);
      } else if (existing.moveMask !== record.moveMask) {
        duplicateChoices += 1;
        if (record.moveMask < existing.moveMask) records.set(record.key, record);
      }
    }
  }
  if (role === 1) {
    const rootColumn = integerOption(options.root_column, Math.floor(columns / 2), 'root-column', 0, columns - 1);
    records.set(0n, { key: 0n, moveMask: 1 << rootColumn, outcome: rootValue });
  }
  const closure = closureFromRecords({ geometry, role, handoffRemaining, records });
  const usedRecords = [...closure.used].map((key) => records.get(key))
    .sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
  const output = resolve(String(options.output));
  await mkdir(output, { recursive: true });
  const filename = `${rows}x${columns}-c${connect}-role${role}.bin`;
  const policyBytes = writePolicy({
    rows, columns, connect, role, handoffRemaining, rootValue,
    closureStates: closure.stats.closureStates, records: usedRecords,
  });
  const policyPath = join(output, filename);
  await writeFile(policyPath, policyBytes);
  const policy = decodePerfectClassicPolicy(policyBytes, { rows, columns, connect, role });
  if (policy.entryCount !== usedRecords.length || policy.closureStates !== closure.stats.closureStates) {
    throw new Error('Assembled policy metadata mismatch.');
  }
  const fileDigest = await digest(policyPath);
  const entry = {
    rows, columns, connect, role, handoffRemaining, rootValue,
    entryCount: policy.entryCount, closureStates: policy.closureStates,
    file: `./${filename}`, ...fileDigest,
    generator: {
      format: 'connect4-perfect-classic-sharded-generation-v1',
      fragmentCount: fragments.length,
      sourceRecordCount: records.size,
      usedRecordCount: usedRecords.length,
      prunedRecordCount: records.size - usedRecords.length,
      duplicateChoices,
      ...closure.stats,
    },
  };
  const manifest = { format: 'connect4-perfect-classic-manifest-v1', generatedAt: new Date().toISOString(), policies: [entry] };
  await writeFile(join(output, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  return { output, manifest };
}

async function finalize(options) {
  if (!options.reference || options.reference === true || !options.verification || options.verification === true) {
    throw new RangeError('--reference and --verification are required.');
  }
  const reference = resolve(String(options.reference));
  const verification = JSON.parse(await readFile(resolve(String(options.verification)), 'utf8'));
  const manifest = JSON.parse(await readFile(reference, 'utf8'));
  if (!Array.isArray(manifest.policies) || manifest.policies.length !== 1
      || !Array.isArray(verification.replay) || verification.replay.length !== 1) {
    throw new Error('Finalization expects one policy and one replay record.');
  }
  const entry = manifest.policies[0];
  const replay = verification.replay[0];
  for (const field of ['rows', 'columns', 'connect', 'role', 'handoffRemaining', 'rootValue', 'entryCount', 'closureStates']) {
    if (entry[field] !== replay[field]) throw new Error(`Replay ${field} mismatch during finalization.`);
  }
  entry.replay = replay;
  manifest.verifiedAt = new Date().toISOString();
  await writeFile(reference, `${JSON.stringify(manifest, null, 2)}\n`);
  return manifest;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  let result;
  if (options.command === 'initial-frontier') result = await initialFrontier(options);
  else if (options.command === 'generate') result = await generateFragment(options);
  else if (options.command === 'collect-frontier') result = await collectFrontier(options);
  else if (options.command === 'assemble') result = await assemble(options);
  else if (options.command === 'finalize') result = await finalize(options);
  else throw new RangeError(`Unknown command: ${options.command}`);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();

export { assemble, collectFrontier, generateFragment, initialFrontier };
