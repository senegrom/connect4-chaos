#!/usr/bin/env node
import { isEntryPoint } from './entry-point.mjs';
import { buildNative } from './native-build.mjs';

import {
  access,
  copyFile,
  mkdir,
  mkdtemp,
  open,
  readFile,
  readdir,
  rename,
  rm,
  writeFile,
} from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const SOURCE = join(ROOT, 'native', 'perfect-chaos-prefix.cpp');
const FRONTIER_MAGIC = Buffer.from('C4CFRN1\0', 'binary');
const POLICY_MAGIC = Buffer.from('C4CPOL1\0', 'binary');
const ROLE_CODES = Object.freeze({ red: 1, yellow: 2 });
const ACTION_DROP = 0;
const ACTION_FLIP = 1;
const ACTION_CW = 2;
const ACTION_CCW = 3;

// The options each command reads. Any other is refused: a misspelt
// --reference used to verify the committed catalog and exit 0 without the
// candidate ever being checked.
const SHARDING = ['shards', 'shard_from_pieces', 'shard_workers', 'maximum_passes', 'seed_rejections', 'journal'];
const COMMAND_OPTIONS = Object.freeze({
  verify: [],
  'verify-reference': ['reference'],
  generate: ['frontier_pieces', 'output', ...SHARDING],
  'reproduce-reference': ['reference', 'output', ...SHARDING],
});

export function parseArguments(argv) {
  const options = { command: argv[0] ?? 'verify' };
  const known = COMMAND_OPTIONS[options.command];
  if (!known) throw new RangeError(`Unknown command: ${options.command}`);
  for (let index = 1; index < argv.length; index += 1) {
    const argument = argv[index];
    if (!argument.startsWith('--')) throw new RangeError(`Unexpected argument: ${argument}`);
    const name = argument.slice(2).replaceAll('-', '_');
    if (!known.includes(name)) throw new RangeError(`${options.command} has no option ${argument}.`);
    const value = argv[index + 1];
    if (value === undefined || value.startsWith('--')) options[name] = true;
    else {
      options[name] = value;
      index += 1;
    }
  }
  return options;
}

function journalDirectory(options, output) {
  if (options.journal === 'none' || options.journal === false) return null;
  const outputPath = resolve(output);
  const selected = typeof options.journal === 'string'
    ? resolve(options.journal)
    : `${outputPath}.journal`;
  if (selected === outputPath || selected.startsWith(`${outputPath}${sep}`)) {
    throw new RangeError('The prefix journal must be outside the generated output directory.');
  }
  return selected;
}

function integerOption(value, fallback, label, minimum = 0, maximum = Number.MAX_SAFE_INTEGER) {
  const selected = value === undefined ? fallback : Number.parseInt(String(value), 10);
  if (!Number.isInteger(selected) || selected < minimum || selected > maximum) {
    throw new RangeError(`${label} must be an integer from ${minimum} to ${maximum}.`);
  }
  return selected;
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

// One cached build per source, compiler and flag set, shared with the native
// tests; its identity also keys the segment journal.
async function compile() {
  const build = await buildNative(SOURCE, { name: 'perfect-chaos-prefix' });
  if (build.warnings) process.stderr.write(`${build.warnings}\n`);
  return { compiler: build.compiler, binary: build.binary, build };
}

function parseJsonLines(output) {
  return output.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function compareState(first, second) {
  if (first.rows !== second.rows) return first.rows - second.rows;
  if (first.columns !== second.columns) return first.columns - second.columns;
  if (first.aiTurn !== second.aiTurn) return Number(first.aiTurn) - Number(second.aiTurn);
  if (first.mover !== second.mover) return first.mover < second.mover ? -1 : 1;
  if (first.opponent !== second.opponent) return first.opponent < second.opponent ? -1 : 1;
  return 0;
}

function stateKey(state) {
  return `${state.rows}:${state.columns}:${state.aiTurn ? 1 : 0}:${state.mover.toString(16)}:${state.opponent.toString(16)}`;
}

function parseHeader(buffer, magic, recordSize, label) {
  if (buffer.length < 16 || !buffer.subarray(0, 8).equals(magic)) {
    throw new Error(`${label} has an invalid magic header.`);
  }
  const version = buffer[8];
  const role = buffer[9];
  const boundary = buffer[10];
  const size = buffer[11];
  const count = buffer.readUInt32LE(12);
  if (version !== 1 || role < 1 || role > 2 || boundary > 42 || size !== recordSize) {
    throw new Error(`${label} has an unsupported header.`);
  }
  if (buffer.length !== 16 + count * recordSize) {
    throw new Error(`${label} length does not match its record count.`);
  }
  return { version, role, boundary, count };
}

async function readFrontier(path) {
  const buffer = await readFile(path);
  const header = parseHeader(buffer, FRONTIER_MAGIC, 19, `Frontier ${path}`);
  const states = [];
  for (let index = 0, offset = 16; index < header.count; index += 1, offset += 19) {
    states.push({
      mover: buffer.readBigUInt64LE(offset),
      opponent: buffer.readBigUInt64LE(offset + 8),
      rows: buffer[offset + 16],
      columns: buffer[offset + 17],
      aiTurn: buffer[offset + 18] !== 0,
    });
  }
  for (let index = 1; index < states.length; index += 1) {
    if (compareState(states[index - 1], states[index]) >= 0) {
      throw new Error(`Frontier ${path} is not strictly sorted.`);
    }
  }
  return { ...header, states };
}

async function readPolicy(path) {
  const buffer = await readFile(path);
  const header = parseHeader(buffer, POLICY_MAGIC, 20, `Policy ${path}`);
  const records = [];
  for (let index = 0, offset = 16; index < header.count; index += 1, offset += 20) {
    const action = { type: buffer[offset + 18], column: buffer[offset + 19] };
    if (action.type > ACTION_CCW || (action.type !== ACTION_DROP && action.column !== 0)) {
      throw new Error(`Policy ${path} contains an invalid action.`);
    }
    records.push({
      state: {
        mover: buffer.readBigUInt64LE(offset),
        opponent: buffer.readBigUInt64LE(offset + 8),
        rows: buffer[offset + 16],
        columns: buffer[offset + 17],
        aiTurn: true,
      },
      action,
    });
  }
  for (let index = 1; index < records.length; index += 1) {
    if (compareState(records[index - 1].state, records[index].state) >= 0) {
      throw new Error(`Policy ${path} is not strictly sorted.`);
    }
  }
  return { ...header, records };
}

function encodeFrontier(role, boundary, states) {
  const ordered = [...new Map(states.map((state) => [stateKey(state), state])).values()]
    .sort(compareState);
  const buffer = Buffer.alloc(16 + ordered.length * 19);
  FRONTIER_MAGIC.copy(buffer, 0);
  buffer[8] = 1;
  buffer[9] = role;
  buffer[10] = boundary;
  buffer[11] = 19;
  buffer.writeUInt32LE(ordered.length, 12);
  for (let index = 0, offset = 16; index < ordered.length; index += 1, offset += 19) {
    const state = ordered[index];
    buffer.writeBigUInt64LE(state.mover, offset);
    buffer.writeBigUInt64LE(state.opponent, offset + 8);
    buffer[offset + 16] = state.rows;
    buffer[offset + 17] = state.columns;
    buffer[offset + 18] = state.aiTurn ? 1 : 0;
  }
  return buffer;
}


function sameAction(first, second) {
  return first.type === second.type && first.column === second.column;
}

function encodePolicy(role, boundary, records) {
  const selected = new Map();
  for (const record of records) {
    if (!record?.state?.aiTurn) throw new Error('Policy records must belong to AI-turn states.');
    const key = stateKey(record.state);
    const existing = selected.get(key);
    if (existing && !sameAction(existing.action, record.action)) {
      throw new Error(`Conflicting Perfect Chaos policy actions for ${key}.`);
    }
    if (!existing) selected.set(key, record);
  }
  const ordered = [...selected.values()].sort((first, second) => (
    compareState(first.state, second.state)
  ));
  const buffer = Buffer.alloc(16 + ordered.length * 20);
  POLICY_MAGIC.copy(buffer, 0);
  buffer[8] = 1;
  buffer[9] = role;
  buffer[10] = boundary;
  buffer[11] = 20;
  buffer.writeUInt32LE(ordered.length, 12);
  for (let index = 0, offset = 16; index < ordered.length; index += 1, offset += 20) {
    const { state, action } = ordered[index];
    buffer.writeBigUInt64LE(state.mover, offset);
    buffer.writeBigUInt64LE(state.opponent, offset + 8);
    buffer[offset + 16] = state.rows;
    buffer[offset + 17] = state.columns;
    buffer[offset + 18] = action.type;
    buffer[offset + 19] = action.column;
  }
  return buffer;
}

async function mergePolicies(target, paths) {
  let role = null;
  let boundary = null;
  const records = new Map();
  for (const path of paths) {
    const policy = await readPolicy(path);
    role ??= policy.role;
    boundary ??= policy.boundary;
    if (policy.role !== role || policy.boundary !== boundary) {
      throw new Error('Cannot merge policies with different roles or boundaries.');
    }
    for (const record of policy.records) {
      const key = stateKey(record.state);
      const existing = records.get(key);
      if (existing && !sameAction(existing.action, record.action)) {
        throw new Error(`Conflicting Perfect Chaos policy actions for ${key}.`);
      }
      if (!existing) records.set(key, record);
    }
  }
  await writeFile(target, encodePolicy(role, boundary, [...records.values()]));
  return { count: (await readPolicy(target)).count, conflicts: 0 };
}

async function mergeFrontiers(target, paths) {
  let role = null;
  let boundary = null;
  const states = [];
  for (const path of paths) {
    const frontier = await readFrontier(path);
    role ??= frontier.role;
    boundary ??= frontier.boundary;
    if (frontier.role !== role || frontier.boundary !== boundary) {
      throw new Error('Cannot merge frontiers with different roles or boundaries.');
    }
    for (const state of frontier.states) states.push(state);
  }
  await writeFile(target, encodeFrontier(role, boundary, states));
  return (await readFrontier(target)).count;
}


async function verifyLargeFrontierMerge(temporary) {
  const directory = join(temporary, 'large-frontier-merge');
  await mkdir(directory, { recursive: true });
  const inputPaths = [
    join(directory, 'first.frontier.bin'),
    join(directory, 'second.frontier.bin'),
  ];
  const boundary = 8;
  const partitionSizes = [180_000, 20_000];
  const totalStates = partitionSizes.reduce((total, count) => total + count, 0);
  const validPositions = [];
  for (let column = 0; column < 7; column += 1) {
    for (let row = 0; row < 6; row += 1) validPositions.push(column * 7 + row);
  }
  const combination = Array.from({ length: boundary }, (_, index) => index);
  const advanceCombination = () => {
    let index = combination.length - 1;
    while (index >= 0
        && combination[index] === validPositions.length - combination.length + index) {
      index -= 1;
    }
    if (index < 0) return false;
    combination[index] += 1;
    for (let next = index + 1; next < combination.length; next += 1) {
      combination[next] = combination[next - 1] + 1;
    }
    return true;
  };

  let produced = 0;
  for (let partition = 0; partition < partitionSizes.length; partition += 1) {
    const states = [];
    for (let index = 0; index < partitionSizes[partition]; index += 1) {
      let mover = 0n;
      for (const position of combination) {
        mover |= 1n << BigInt(validPositions[position]);
      }
      states.push({ mover, opponent: 0n, rows: 6, columns: 7, aiTurn: true });
      produced += 1;
      if (produced < totalStates && !advanceCombination()) {
        throw new Error('Synthetic large-frontier verification exhausted its state space.');
      }
    }
    await writeFile(
      inputPaths[partition],
      encodeFrontier(ROLE_CODES.red, boundary, states),
    );
  }

  const outputPath = join(directory, 'merged.frontier.bin');
  const mergedCount = await mergeFrontiers(outputPath, inputPaths);
  const merged = await readFrontier(outputPath);
  if (mergedCount !== totalStates || merged.count !== totalStates) {
    throw new Error(
      `Large frontier merge retained ${merged.count} of ${totalStates} synthetic states.`,
    );
  }
  return {
    inputFiles: inputPaths.length,
    inputStates: totalStates,
    mergedStates: merged.count,
  };
}

async function splitFrontier(path, requestedShards, directory, prefix = '') {
  const frontier = await readFrontier(path);
  const shardCount = Math.min(requestedShards, frontier.states.length);
  if (shardCount < 1) throw new Error(`Cannot shard an empty frontier: ${path}`);
  const buckets = Array.from({ length: shardCount }, () => []);
  frontier.states.forEach((state, index) => buckets[index % shardCount].push(state));
  const paths = [];
  for (let shardIndex = 0; shardIndex < buckets.length; shardIndex += 1) {
    const shardPath = join(
      directory,
      `${prefix}${String(shardIndex).padStart(3, '0')}.input.bin`,
    );
    await writeFile(
      shardPath,
      encodeFrontier(frontier.role, frontier.boundary, buckets[shardIndex]),
    );
    paths.push(shardPath);
  }
  return paths;
}

function stride(state) {
  return state.rows + 1;
}

function bit(state, column, rowFromBottom) {
  return 1n << BigInt(column * stride(state) + rowFromBottom);
}

function pieceCount(state) {
  let value = state.mover | state.opponent;
  let count = 0;
  while (value !== 0n) {
    value &= value - 1n;
    count += 1;
  }
  return count;
}

function hasWin(pieces, shiftStride, connect = 4) {
  for (const shift of [1, shiftStride, shiftStride - 1, shiftStride + 1]) {
    let run = pieces;
    for (let offset = 1; offset < connect; offset += 1) {
      run &= pieces >> BigInt(offset * shift);
    }
    if (run !== 0n) return true;
  }
  return false;
}

function isFull(state) {
  const occupied = state.mover | state.opponent;
  for (let column = 0; column < state.columns; column += 1) {
    if ((occupied & bit(state, column, state.rows - 1)) === 0n) return false;
  }
  return true;
}

function mirror(state) {
  const reflected = { ...state, mover: 0n, opponent: 0n };
  for (let column = 0; column < state.columns; column += 1) {
    for (let row = 0; row < state.rows; row += 1) {
      const source = bit(state, column, row);
      const target = bit(reflected, state.columns - 1 - column, row);
      if ((state.mover & source) !== 0n) reflected.mover |= target;
      if ((state.opponent & source) !== 0n) reflected.opponent |= target;
    }
  }
  return reflected;
}

function canonicalize(state) {
  const reflected = mirror(state);
  return reflected.mover < state.mover
    || (reflected.mover === state.mover && reflected.opponent < state.opponent)
    ? reflected
    : state;
}

function transform(state, type) {
  const output = {
    mover: 0n,
    opponent: 0n,
    rows: type === ACTION_FLIP ? state.rows : state.columns,
    columns: type === ACTION_FLIP ? state.columns : state.rows,
    aiTurn: state.aiTurn,
  };
  const targetColumns = Array.from({ length: output.columns }, () => []);
  for (let column = 0; column < state.columns; column += 1) {
    for (let row = 0; row < state.rows; row += 1) {
      const source = bit(state, column, row);
      const owner = (state.mover & source) !== 0n ? 1 : (state.opponent & source) !== 0n ? 2 : 0;
      if (!owner) continue;
      let targetColumn;
      let targetRow;
      if (type === ACTION_FLIP) {
        targetColumn = column;
        targetRow = state.rows - 1 - row;
      } else if (type === ACTION_CW) {
        targetColumn = row;
        targetRow = state.columns - 1 - column;
      } else {
        targetColumn = state.rows - 1 - row;
        targetRow = column;
      }
      targetColumns[targetColumn].push({ targetRow, owner });
    }
  }
  for (let column = 0; column < output.columns; column += 1) {
    targetColumns[column].sort((first, second) => first.targetRow - second.targetRow);
    targetColumns[column].forEach(({ owner }, row) => {
      if (owner === 1) output.mover |= bit(output, column, row);
      else output.opponent |= bit(output, column, row);
    });
  }
  return output;
}

function apply(state, action) {
  let next = { ...state };
  if (action.type === ACTION_DROP) {
    if (action.column >= state.columns) throw new Error('Policy drop column is outside the board.');
    const occupied = state.mover | state.opponent;
    let row = 0;
    while (row < state.rows && (occupied & bit(state, action.column, row)) !== 0n) row += 1;
    if (row >= state.rows) throw new Error('Policy attempts to drop in a full column.');
    next.mover |= bit(state, action.column, row);
    if (hasWin(next.mover, stride(next))) return { moverResult: 1 };
  } else {
    next = transform(state, action.type);
    const moverWon = hasWin(next.mover, stride(next));
    const opponentWon = hasWin(next.opponent, stride(next));
    if (moverWon && opponentWon) return { moverResult: -1 };
    if (moverWon) return { moverResult: 1 };
    if (opponentWon) return { moverResult: -1 };
  }
  if (isFull(next)) return { moverResult: 2 };
  return {
    moverResult: 0,
    state: canonicalize({
      mover: next.opponent,
      opponent: next.mover,
      rows: next.rows,
      columns: next.columns,
      aiTurn: !state.aiTurn,
    }),
  };
}

function legalActions(state) {
  if (isFull(state)) return [];
  const occupied = state.mover | state.opponent;
  const columns = Array.from({ length: state.columns }, (_, column) => column)
    .sort((first, second) => {
      const firstDistance = Math.abs(2 * first - (state.columns - 1));
      const secondDistance = Math.abs(2 * second - (state.columns - 1));
      return firstDistance - secondDistance || first - second;
    });
  const actions = columns
    .filter((column) => (occupied & bit(state, column, state.rows - 1)) === 0n)
    .map((column) => ({ type: ACTION_DROP, column }));
  actions.push(
    { type: ACTION_FLIP, column: 0 },
    { type: ACTION_CW, column: 0 },
    { type: ACTION_CCW, column: 0 },
  );
  return actions;
}

function terminalForAi(state, moverResult) {
  if (moverResult === 2) return 'draw';
  const moverWon = moverResult === 1;
  return moverWon === state.aiTurn ? 'ai-win' : 'ai-loss';
}

function deduplicatedTransitions(state, actions) {
  const seenTerminals = new Set();
  const seenStates = new Set();
  const transitions = [];
  for (const action of actions) {
    const transition = apply(state, action);
    if (transition.moverResult !== 0) {
      const terminal = terminalForAi(state, transition.moverResult);
      if (seenTerminals.has(terminal)) continue;
      seenTerminals.add(terminal);
      transitions.push({ terminal, action });
      continue;
    }
    const key = stateKey(transition.state);
    if (seenStates.has(key)) continue;
    seenStates.add(key);
    transitions.push({ terminal: null, action, state: transition.state });
  }
  return transitions;
}

// boundary is the segment's target as the caller - the manifest, for a
// committed certificate - knows it; the replay stops at the header's boundary,
// so a pair of files whose headers agree on another layer would otherwise
// replay a different segment than the one claimed.
async function replaySegment({
  role, inputStates, policyPath, frontierPath, boundary: segmentBoundary,
}) {
  const policy = await readPolicy(policyPath);
  const expected = await readFrontier(frontierPath);
  if (policy.role !== role || expected.role !== role || policy.boundary !== expected.boundary) {
    throw new Error('Policy/frontier role or boundary mismatch.');
  }
  if (!Number.isInteger(segmentBoundary) || expected.boundary !== segmentBoundary) {
    throw new Error(
      `${basename(frontierPath)} ends at ${expected.boundary} pieces, `
      + `but the segment it belongs to ends at ${segmentBoundary}.`,
    );
  }
  const policyMap = new Map(policy.records.map((record) => [stateKey(record.state), record.action]));
  if (policyMap.size !== policy.records.length) throw new Error('Policy contains duplicate states.');
  const usedPolicy = new Set();
  const visited = new Set();
  const queue = [];
  for (const raw of inputStates) {
    const state = canonicalize(raw);
    const key = stateKey(state);
    if (!visited.has(key)) {
      visited.add(key);
      queue.push(state);
    }
  }
  const boundary = [];
  let terminalAiWins = 0;
  let terminalDraws = 0;
  let revisitedEdges = 0;
  let aiStates = 0;
  let opponentStates = 0;

  const follow = (state, transition) => {
    if (transition.terminal === 'ai-loss') throw new Error('Replay reaches an AI-loss terminal.');
    if (transition.terminal === 'ai-win') {
      terminalAiWins += 1;
      return;
    }
    if (transition.terminal === 'draw') {
      terminalDraws += 1;
      return;
    }
    const key = stateKey(transition.state);
    if (visited.has(key)) revisitedEdges += 1;
    else {
      visited.add(key);
      queue.push(transition.state);
    }
  };

  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const state = queue[cursor];
    if (pieceCount(state) === expected.boundary) {
      boundary.push(state);
      continue;
    }
    if (pieceCount(state) > expected.boundary) throw new Error('Replay crossed the target boundary.');
    if (state.aiTurn) {
      aiStates += 1;
      const key = stateKey(state);
      const action = policyMap.get(key);
      if (!action) throw new Error(`Replay is missing policy state ${key}.`);
      usedPolicy.add(key);
      const [transition] = deduplicatedTransitions(state, [action]);
      follow(state, transition);
    } else {
      opponentStates += 1;
      for (const transition of deduplicatedTransitions(state, legalActions(state))) {
        follow(state, transition);
      }
    }
  }

  const actual = [...new Map(boundary.map((state) => [stateKey(state), state])).values()]
    .sort(compareState);
  if (actual.length !== expected.states.length
      || actual.some((state, index) => compareState(state, expected.states[index]) !== 0)) {
    throw new Error(`Replay frontier mismatch for ${basename(frontierPath)}.`);
  }
  if (usedPolicy.size !== policyMap.size) {
    throw new Error(`Policy ${basename(policyPath)} has ${policyMap.size - usedPolicy.size} unreachable records.`);
  }
  return {
    fromStates: inputStates.length,
    frontierStates: actual.length,
    closureStates: visited.size,
    aiStates,
    opponentStates,
    policyEntries: usedPolicy.size,
    terminalAiWins,
    terminalDraws,
    revisitedEdges,
  };
}

async function replayRole(directory, roleName, boundaries) {
  const role = ROLE_CODES[roleName];
  let inputStates = [{ mover: 0n, opponent: 0n, rows: 6, columns: 7, aiTurn: roleName === 'red' }];
  let from = 0;
  const segments = [];
  for (const boundary of boundaries) {
    const policyPath = join(directory, roleName, `${from}-${boundary}.policy.bin`);
    const frontierPath = join(directory, roleName, `${from}-${boundary}.frontier.bin`);
    const summary = await replaySegment({ role, inputStates, policyPath, frontierPath, boundary });
    segments.push({ fromPieces: from, frontierPieces: boundary, ...summary });
    inputStates = (await readFrontier(frontierPath)).states;
    from = boundary;
  }
  return { role: roleName, segments };
}

function maximumStates(boundary) {
  if (boundary <= 8) return 5_000_000;
  if (boundary <= 10) return 12_000_000;
  if (boundary <= 12) return 30_000_000;
  if (boundary <= 14) return 60_000_000;
  return 100_000_000;
}

async function nativeSegment(binary, args) {
  const result = await run(binary, args);
  const records = result.stdout.trim() ? parseJsonLines(result.stdout) : [];
  return { ...result, records };
}

function isSplittableResourceFailure(result) {
  const details = `${result.stderr ?? ''}\n${result.stdout ?? ''}`;
  return /Prefix graph exceeded its state limit\./.test(details)
    || /std::bad_alloc/.test(details)
    || result.code === 137
    || result.signal === 'SIGKILL';
}


async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function shardedNativeExtension({
  binary,
  inputFrontier,
  targetBoundary,
  maximumStateCount,
  policyPath,
  frontierPath,
  targetReject,
  rejectedPath,
  shardCount,
  minimumStatesPerShard = 2_000_000,
  shardWorkers = 1,
  journal = null,
}) {
  const shardDirectory = join(dirname(policyPath), `.shards-${basename(policyPath, '.policy.bin')}`);
  await rm(shardDirectory, { recursive: true, force: true });
  await mkdir(shardDirectory, { recursive: true });

  const policyPaths = [];
  const frontierPaths = [];
  const rejectedPaths = [];
  const summaries = [];

  try {
    const source = await readFrontier(inputFrontier);
    const inputPaths = await splitFrontier(
      inputFrontier,
      shardCount,
      shardDirectory,
      'root-',
    );
    const maximumStatesPerShard = Math.max(
      minimumStatesPerShard,
      Math.ceil(maximumStateCount / inputPaths.length),
    );
    const rejectSha256 = targetReject ? await sha256OfFile(targetReject) : null;
    const workerCount = Math.max(1, Math.min(shardWorkers, inputPaths.length));
    const pending = inputPaths.map((inputPath, index) => ({
      inputPath,
      label: String(index).padStart(3, '0'),
      depth: 0,
    }));
    const maximumAttempts = Math.max(1_024, source.count * 2 + inputPaths.length);
    let attempts = 0;
    let adaptiveSplits = 0;
    let maximumSplitDepth = 0;

    const processTask = async (task) => {
      const prefix = join(shardDirectory, `leaf-${task.label}`);
      const shardPolicy = `${prefix}.policy.bin`;
      const shardFrontier = `${prefix}.frontier.bin`;
      const shardRejected = `${prefix}.rejected.bin`;
      await Promise.all([
        rm(shardPolicy, { force: true }),
        rm(shardFrontier, { force: true }),
        rm(shardRejected, { force: true }),
      ]);

      const result = await journaledSegment(
        journal,
        {
          kind: 'extend-shard',
          frontierPieces: targetBoundary,
          maximumStates: maximumStatesPerShard,
          inputSha256: await sha256OfFile(task.inputPath),
          rejectSha256,
        },
        () => nativeSegment(binary, [
          'extend',
          '--input-frontier', task.inputPath,
          '--frontier-pieces', String(targetBoundary),
          '--maximum-states', String(maximumStatesPerShard),
          '--policy', shardPolicy,
          '--frontier', shardFrontier,
          ...(targetReject ? ['--reject-frontier', targetReject] : []),
          '--rejected', shardRejected,
        ]),
        {
          policyPath: shardPolicy,
          frontierPath: shardFrontier,
          rejectedPath: shardRejected,
        },
      );

      if (result.code === 0) {
        const summary = result.records.at(-1);
        if (!summary) throw new Error(`Shard ${task.label} returned no certificate summary.`);
        return {
          kind: 'safe',
          summary,
          policyPath: shardPolicy,
          frontierPath: shardFrontier,
        };
      }

      // A shard killed for memory can leave a half-written rejection file, so
      // resource failures are recognised first and a rejection only counts
      // when the solver finished writing a table that parses.
      if (isSplittableResourceFailure(result)) {
        const oversized = await readFrontier(task.inputPath);
        if (oversized.count <= 1) {
          const key = oversized.states[0] ? stateKey(oversized.states[0]) : 'empty';
          throw new Error(
            `A single input root exceeded the ${maximumStatesPerShard.toLocaleString()}-state `
            + `shard limit at ${targetBoundary} pieces: ${key}.`,
          );
        }
        const childPaths = await splitFrontier(
          task.inputPath,
          2,
          shardDirectory,
          `${task.label}-`,
        );
        await rm(task.inputPath, { force: true });
        return {
          kind: 'split',
          depth: task.depth + 1,
          children: childPaths.map((inputPath, index) => ({
            inputPath,
            label: `${task.label}.${index}`,
            depth: task.depth + 1,
          })),
        };
      }

      if (await rejectionTable(result, shardRejected)) {
        return { kind: 'rejected', rejectedPath: shardRejected };
      }

      throw new Error(
        `Shard ${task.label} failed without a rejection certificate.\n`
        + (result.stderr || result.stdout),
      );
    };

    while (pending.length > 0) {
      const batch = pending.splice(0, workerCount);
      attempts += batch.length;
      if (attempts > maximumAttempts) {
        throw new Error(
          `Adaptive sharding exceeded ${maximumAttempts.toLocaleString()} attempts.`,
        );
      }
      const settled = await Promise.allSettled(batch.map(processTask));
      const failed = settled.find((outcome) => outcome.status === 'rejected');
      if (failed) throw failed.reason;

      const children = [];
      for (const outcome of settled) {
        const result = outcome.value;
        if (result.kind === 'safe') {
          summaries.push(result.summary);
          policyPaths.push(result.policyPath);
          frontierPaths.push(result.frontierPath);
        } else if (result.kind === 'rejected') {
          rejectedPaths.push(result.rejectedPath);
        } else {
          adaptiveSplits += 1;
          maximumSplitDepth = Math.max(maximumSplitDepth, result.depth);
          children.push(...result.children);
        }
      }
      pending.splice(0, 0, ...children);
    }

    if (rejectedPaths.length > 0) {
      await mergeFrontiers(rejectedPath, rejectedPaths);
      return {
        code: 1,
        signal: null,
        stdout: '',
        stderr: `${rejectedPaths.length} shard(s) contained losing input roots.`,
        records: [],
      };
    }

    const policy = await mergePolicies(policyPath, policyPaths);
    const frontierStates = await mergeFrontiers(frontierPath, frontierPaths);
    const sum = (field) => summaries.reduce(
      (total, summary) => total + Number(summary[field] ?? 0),
      0,
    );
    return {
      code: 0,
      signal: null,
      stdout: '',
      stderr: '',
      records: [{
        format: 'connect4-chaos-prefix-sharded-certificate-v1',
        role: summaries[0]?.role,
        fromPieces: summaries[0]?.fromPieces,
        frontierPieces: targetBoundary,
        requestedShards: inputPaths.length,
        shards: policyPaths.length,
        shardWorkers: workerCount,
        adaptiveSplits,
        maximumSplitDepth,
        maximumStatesPerShard,
        inputRoots: sum('inputRoots'),
        shardGraphStates: sum('graphStates'),
        shardGraphEdges: sum('graphEdges'),
        shardLosingStates: sum('losingStates'),
        shardSafeStates: sum('safeStates'),
        shardClosureStates: sum('closureStates'),
        frontierStates,
        policyEntries: policy.count,
        policyConflicts: policy.conflicts,
        shardTerminalAiWins: sum('terminalAiWins'),
        shardTerminalDraws: sum('terminalDraws'),
        shardRevisitedEdges: sum('revisitedEdges'),
      }],
    };
  } finally {
    await rm(shardDirectory, { recursive: true, force: true });
  }
}

async function hashFile(path) {
  const handle = await open(path, 'r');
  const digest = createHash('sha256');
  const buffer = Buffer.allocUnsafe(1024 * 1024);
  let bytes = 0;
  try {
    while (true) {
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, null);
      if (bytesRead === 0) break;
      digest.update(buffer.subarray(0, bytesRead));
      bytes += bytesRead;
    }
  } finally {
    await handle.close();
  }
  return {
    path: basename(path),
    bytes,
    sha256: digest.digest('hex'),
  };
}


function note(message) {
  process.stderr.write(`[perfect-chaos-prefix ${new Date().toISOString()}] ${message}\n`);
}

async function sha256OfFile(path) {
  return (await hashFile(path)).sha256;
}

// Entries are keyed on what determines the binary - source digest, compiler,
// its version and the full flag list - not on the binary's own bytes: those
// change with every MinGW link, which made the journal useless on Windows.
async function createJournal(directory, build) {
  if (!directory) return null;
  await mkdir(directory, { recursive: true });
  const journal = {
    format: 'connect4-chaos-prefix-journal-v3',
    directory,
    sourceSha256: build.sourceSha256,
    compiler: build.compiler,
    compilerVersion: build.compilerVersion,
    flags: [...build.flags],
    hits: 0,
    misses: 0,
    stores: 0,
    invalidations: 0,
    resetStatistics() {
      this.hits = 0;
      this.misses = 0;
      this.stores = 0;
      this.invalidations = 0;
    },
    summary() {
      return {
        format: this.format,
        directory: this.directory,
        sourceSha256: this.sourceSha256,
        compiler: this.compiler,
        flags: this.flags,
        hits: this.hits,
        misses: this.misses,
        stores: this.stores,
        invalidations: this.invalidations,
      };
    },
  };
  return journal;
}

function journalKey(journal, descriptor) {
  const canonical = stable({
    format: journal.format,
    sourceSha256: journal.sourceSha256,
    compiler: journal.compiler,
    compilerVersion: journal.compilerVersion,
    flags: journal.flags,
    descriptor,
  });
  return createHash('sha256').update(JSON.stringify(canonical)).digest('hex');
}

async function invalidateJournalEntry(journal, entryDirectory) {
  await rm(entryDirectory, { recursive: true, force: true });
  journal.invalidations += 1;
}

async function journalLookup(journal, key, destinations) {
  const entryDirectory = join(journal.directory, key);
  let meta;
  try {
    meta = JSON.parse(await readFile(join(entryDirectory, 'meta.json'), 'utf8'));
  } catch (error) {
    if (error?.code !== 'ENOENT') await invalidateJournalEntry(journal, entryDirectory);
    return null;
  }
  if (meta?.format !== journal.format || meta?.key !== key
      || !Number.isInteger(meta.code) || !Array.isArray(meta.files)
      || !Array.isArray(meta.records)) {
    await invalidateJournalEntry(journal, entryDirectory);
    return null;
  }

  const restored = [];
  try {
    for (const file of meta.files) {
      if (!file || typeof file.name !== 'string'
          || !Number.isInteger(file.bytes) || file.bytes < 0
          || !/^[0-9a-f]{64}$/.test(file.sha256)) {
        throw new Error('Journal file metadata is malformed.');
      }
      const destination = destinations[file.name];
      if (!destination) throw new Error(`Unexpected journal output: ${file.name}.`);
      const bytes = await readFile(join(entryDirectory, file.name));
      const digest = createHash('sha256').update(bytes).digest('hex');
      if (bytes.length !== file.bytes || digest !== file.sha256) {
        throw new Error(`Journal output digest differs: ${file.name}.`);
      }
      restored.push({ destination, bytes });
    }
  } catch {
    await invalidateJournalEntry(journal, entryDirectory);
    return null;
  }

  for (const file of restored) await writeFile(file.destination, file.bytes);
  journal.hits += 1;
  return {
    code: meta.code,
    signal: null,
    stdout: '',
    stderr: typeof meta.stderr === 'string' ? meta.stderr : '',
    records: meta.records,
  };
}

async function journalStore(journal, key, result, storedFiles) {
  const entryDirectory = join(journal.directory, key);
  const temporary = `${entryDirectory}.tmp-${process.pid}-${Math.random().toString(16).slice(2)}`;
  await rm(temporary, { recursive: true, force: true });
  await mkdir(temporary, { recursive: true });
  const files = [];
  try {
    for (const [name, source] of Object.entries(storedFiles)) {
      const bytes = await readFile(source);
      await writeFile(join(temporary, name), bytes);
      files.push({
        name,
        bytes: bytes.length,
        sha256: createHash('sha256').update(bytes).digest('hex'),
      });
    }
    files.sort((first, second) => first.name.localeCompare(second.name));
    await writeFile(join(temporary, 'meta.json'), `${JSON.stringify({
      format: journal.format,
      key,
      code: result.code,
      stderr: result.stderr ?? '',
      records: result.records ?? [],
      files,
    }, null, 2)}\n`);
    try {
      await rename(temporary, entryDirectory);
      journal.stores += 1;
    } catch (error) {
      if (error?.code !== 'EEXIST' && error?.code !== 'ENOTEMPTY') throw error;
      await rm(temporary, { recursive: true, force: true });
    }
  } catch (error) {
    await rm(temporary, { recursive: true, force: true });
    throw error;
  }
}

// A segment rejects its input only by exiting 1 after writing a complete table
// of the losing roots. A crash, a kill part-way through the write or a
// resource limit is a failure, never a rejection, whatever file it left.
async function rejectionTable(result, path) {
  if (result.code !== 1 || typeof path !== 'string') return null;
  try {
    return await readFrontier(path);
  } catch {
    return null;
  }
}

async function journaledSegment(journal, descriptor, invoke, files) {
  if (!journal) return invoke();
  const destinations = {
    'policy.bin': files.policyPath,
    'frontier.bin': files.frontierPath,
    'rejected.bin': files.rejectedPath,
  };
  await Promise.all(Object.values(destinations)
    .filter((path) => typeof path === 'string')
    .map((path) => rm(path, { force: true })));
  const key = journalKey(journal, descriptor);
  const cached = await journalLookup(journal, key, destinations);
  if (cached) return cached;
  journal.misses += 1;
  const result = await invoke();
  // Outputs are parsed before they are stored: a journal entry is replayed
  // without rerunning anything, so it must never hold a torn table.
  if (result.code === 0) {
    await readPolicy(files.policyPath);
    await readFrontier(files.frontierPath);
    await journalStore(journal, key, result, {
      'policy.bin': files.policyPath,
      'frontier.bin': files.frontierPath,
    });
  } else if (await rejectionTable(result, files.rejectedPath)) {
    await journalStore(journal, key, result, { 'rejected.bin': files.rejectedPath });
  }
  return result;
}

async function corruptOneJournalOutput(journal) {
  const entries = (await readdir(journal.directory, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory() && /^[0-9a-f]{64}$/.test(entry.name))
    .sort((first, second) => first.name.localeCompare(second.name));
  for (const entry of entries) {
    const entryDirectory = join(journal.directory, entry.name);
    const meta = JSON.parse(await readFile(join(entryDirectory, 'meta.json'), 'utf8'));
    const first = meta.files?.[0];
    if (!first?.name) continue;
    await writeFile(join(entryDirectory, first.name), Buffer.from('corrupted journal output'));
    return { entry: entry.name, file: first.name };
  }
  throw new Error('Could not find a journal output to corrupt.');
}

export async function initializeRejections(output, roleName, boundaries, seedDirectory) {
  const roleDirectory = join(output, roleName);
  await mkdir(roleDirectory, { recursive: true });
  // A seed stops at some boundary, and past it the rejections start empty:
  // the committed certificate seeds reject-8 to reject-14 of a run to 16.
  // A missing seed directory, or a gap below its deepest rejection file, is
  // a mistake; a mistyped --seed-rejections used to run with no seeds at all.
  let deepestSeed = -Infinity;
  if (seedDirectory) {
    let names;
    try {
      names = await readdir(join(seedDirectory, roleName));
    } catch (error) {
      if (error?.code === 'ENOENT') throw new Error(`No seed rejections at ${join(seedDirectory, roleName)}.`);
      throw error;
    }
    for (const name of names) {
      const match = /^reject-(\d+)\.bin$/.exec(name);
      if (match) deepestSeed = Math.max(deepestSeed, Number(match[1]));
    }
  }
  const rejects = new Map();
  for (const boundary of boundaries) {
    const path = join(roleDirectory, `reject-${boundary}.bin`);
    const seed = boundary <= deepestSeed ? join(seedDirectory, roleName, `reject-${boundary}.bin`) : null;
    if (seed) {
      try {
        await copyFile(seed, path);
      } catch (error) {
        if (error?.code === 'ENOENT') throw new Error(`The seed rejections skip boundary ${boundary}: ${seed}`);
        throw error;
      }
      const decoded = await readFrontier(path);
      if (decoded.role !== ROLE_CODES[roleName] || decoded.boundary !== boundary) {
        throw new Error(`Seed rejection file has the wrong role or boundary: ${seed}`);
      }
    } else {
      await writeFile(path, encodeFrontier(ROLE_CODES[roleName], boundary, []));
    }
    rejects.set(boundary, path);
  }
  return { roleDirectory, rejects };
}

async function rejectionCounts(rejects) {
  const counts = {};
  for (const [boundary, path] of rejects) {
    counts[`at${boundary}`] = (await readFrontier(path)).count;
  }
  return counts;
}

async function generateRole(
  binary,
  output,
  roleName,
  boundaries,
  maximumPasses,
  seedDirectory = null,
  shardCount = 1,
  shardFromBoundary = 14,
  minimumStatesPerShard = 2_000_000,
  shardWorkers = 1,
  allowIncomplete = false,
  journal = null,
) {
  const { roleDirectory, rejects } = await initializeRejections(
    output,
    roleName,
    boundaries.slice(0, -1),
    seedDirectory,
  );

  for (let pass = 1; pass <= maximumPasses; pass += 1) {
    let from = 0;
    let inputFrontier = null;
    const nativeSummaries = [];
    let restart = false;

    for (const boundary of boundaries) {
      const policyPath = join(roleDirectory, `${from}-${boundary}.policy.bin`);
      const frontierPath = join(roleDirectory, `${from}-${boundary}.frontier.bin`);
      const targetReject = rejects.get(boundary);
      const newRejectPath = join(roleDirectory, `new-reject-${from}.bin`);
      const args = from === 0
        ? [
          'generate', '--role', roleName,
          '--frontier-pieces', String(boundary),
          '--maximum-states', String(maximumStates(boundary)),
          '--policy', policyPath,
          '--frontier', frontierPath,
          ...(targetReject ? ['--reject-frontier', targetReject] : []),
        ]
        : [
          'extend', '--input-frontier', inputFrontier,
          '--frontier-pieces', String(boundary),
          '--maximum-states', String(maximumStates(boundary)),
          '--policy', policyPath,
          '--frontier', frontierPath,
          ...(targetReject ? ['--reject-frontier', targetReject] : []),
          '--rejected', newRejectPath,
        ];
      const useShards = from > 0 && shardCount > 1 && boundary >= shardFromBoundary;
      note(`${roleName} pass ${pass}: segment ${from}→${boundary}`
        + `${useShards ? ` (${shardCount} requested shards)` : ''}…`);
      const result = useShards
        ? await shardedNativeExtension({
          binary,
          inputFrontier,
          targetBoundary: boundary,
          maximumStateCount: maximumStates(boundary),
          policyPath,
          frontierPath,
          targetReject,
          rejectedPath: newRejectPath,
          shardCount,
          minimumStatesPerShard,
          shardWorkers,
          journal,
        })
        : await journaledSegment(
          journal,
          {
            kind: from === 0 ? 'generate' : 'extend',
            role: roleName,
            fromPieces: from,
            frontierPieces: boundary,
            maximumStates: maximumStates(boundary),
            inputSha256: from === 0 ? null : await sha256OfFile(inputFrontier),
            rejectSha256: targetReject ? await sha256OfFile(targetReject) : null,
          },
          () => nativeSegment(binary, args),
          {
            policyPath,
            frontierPath,
            rejectedPath: from === 0 ? null : newRejectPath,
          },
        );
      if (result.code === 0) {
        const summary = result.records.at(-1);
        note(`${roleName} pass ${pass}: segment ${from}→${boundary} is safe `
          + `(${summary?.frontierStates ?? summary?.frontier_states ?? '?'} frontier states).`);
        nativeSummaries.push(summary);
        inputFrontier = frontierPath;
        from = boundary;
        continue;
      }
      if (from === 0) {
        throw new Error(`Root prefix became losing.\n${result.stderr || result.stdout}`);
      }
      if (!await rejectionTable(result, newRejectPath)) {
        throw new Error(
          `Segment ${from}→${boundary} failed without a rejection certificate.\n`
          + (result.stderr || result.stdout),
        );
      }
      const previousReject = rejects.get(from);
      const before = (await readFrontier(previousReject)).count;
      const merged = join(roleDirectory, `merged-reject-${from}.bin`);
      const count = await mergeFrontiers(merged, [previousReject, newRejectPath]);
      if (count <= before) throw new Error(`Prefix refinement at ${from} pieces made no progress.`);
      note(`${roleName} pass ${pass}: segment ${from}→${boundary} rejected `
        + `${count - before} new losing roots at ${from} pieces (${count} total).`);
      await rm(previousReject, { force: true });
      await writeFile(previousReject, await readFile(merged));
      await rm(merged, { force: true });
      restart = true;
      break;
    }

    if (restart) continue;
    note(`${roleName}: replaying the complete certificate…`);
    const replay = await replayRole(output, roleName, boundaries);
    return { nativeSummaries, replay, rejected: await rejectionCounts(rejects) };
  }
  if (!allowIncomplete) {
    throw new Error(`${roleName} prefix synthesis exceeded ${maximumPasses} refinement passes.`);
  }
  return {
    incomplete: true,
    passes: maximumPasses,
    rejected: await rejectionCounts(rejects),
  };
}

function roleBoundaries(target) {
  if (target < 8 || target % 2 !== 0) {
    throw new RangeError('The checkpoint frontier must be an even piece count of at least 8.');
  }
  const boundaries = [8];
  for (let boundary = 10; boundary <= target; boundary += 2) boundaries.push(boundary);
  return boundaries;
}


const COMMITTED_REFERENCE = join(ROOT, 'data', 'perfect-chaos-prefix');

/** Whether either directory is, or holds, the other. */
export function overlapping(first, second) {
  const inside = (child, parent) => {
    const path = relative(parent, child);
    return path === '' || (path !== '..' && !path.startsWith(`..${sep}`) && !isAbsolute(path));
  };
  return inside(first, second) || inside(second, first);
}

export async function generateReference(
  binary,
  output,
  target,
  maximumPasses = 200,
  seedDirectory = null,
  shardCount = 1,
  shardFromBoundary = 14,
  minimumStatesPerShard = 2_000_000,
  shardWorkers = 1,
  journal = null,
  keep = [],
) {
  if (target < 8 || target % 2 !== 0) {
    throw new RangeError('The reference frontier must be an even piece count of at least 8.');
  }
  // The output is deleted before anything is written, so it must neither
  // hold nor sit inside the seeds, the committed certificate or the one being
  // reproduced: `reproduce-reference --reference generated/x/manifest.json`
  // deleted the certificate it was about to check.
  for (const directory of [seedDirectory, COMMITTED_REFERENCE, ...keep]) {
    if (directory && overlapping(resolve(output), resolve(directory))) {
      throw new RangeError(`The output ${output} overlaps ${directory}, which generating would delete.`);
    }
  }
  const boundaries = [8];
  for (let boundary = 10; boundary <= target; boundary += 2) boundaries.push(boundary);
  await rm(output, { recursive: true, force: true });
  await mkdir(output, { recursive: true });
  const roles = {};
  for (const roleName of ['red', 'yellow']) {
    roles[roleName] = await generateRole(
      binary,
      output,
      roleName,
      boundaries,
      maximumPasses,
      seedDirectory,
      shardCount,
      shardFromBoundary,
      minimumStatesPerShard,
      shardWorkers,
      false,
      journal,
    );
  }

  const artifacts = {};
  for (const roleName of ['red', 'yellow']) {
    const files = [];
    let from = 0;
    for (const boundary of boundaries) {
      files.push(
        join(output, roleName, `${from}-${boundary}.policy.bin`),
        join(output, roleName, `${from}-${boundary}.frontier.bin`),
      );
      from = boundary;
    }
    for (const boundary of boundaries.slice(0, -1)) {
      files.push(join(output, roleName, `reject-${boundary}.bin`));
    }
    artifacts[roleName] = [];
    for (const path of files) artifacts[roleName].push(await hashFile(path));
  }

  const manifest = {
    format: 'connect4-chaos-layered-prefix-manifest-v1',
    theorem: 'finite-safety-game-with-quotient-cycles-lifting-to-threefold-draws',
    board: { rows: 6, columns: 7, connect: 4, chaosMode: true },
    boundaries,
    ...(shardCount > 1 ? { sharding: { count: shardCount, fromBoundary: shardFromBoundary, workers: shardWorkers } } : {}),
    sourceSha256: createHash('sha256').update(await readFile(SOURCE)).digest('hex'),
    roles,
    artifacts,
  };
  await writeFile(join(output, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  return manifest;
}

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stable(value[key])]));
  }
  return value;
}

// What a regeneration can reproduce: the certificate bytes and the solver and
// replay summaries. The committed manifest also carries provenance no
// generator writes - the preserved generator's source and commit, the original
// manifest, the promotion audit - and that generator's sourceSha256 rather
// than the current source's, so comparing whole manifests could never pass.
export function reproducesReference(generated, reference) {
  const comparable = (manifest) => JSON.stringify(stable({
    roles: manifest?.roles,
    artifacts: manifest?.artifacts,
  }));
  return comparable(generated) === comparable(reference);
}

async function verifyCommittedReference(referencePath, binary, scratch) {
  const reference = JSON.parse(await readFile(referencePath, 'utf8'));
  if (reference.format !== 'connect4-chaos-layered-prefix-manifest-v1') {
    throw new Error('Unsupported Perfect Chaos prefix manifest format.');
  }
  if (!Array.isArray(reference.boundaries) || reference.boundaries.length === 0) {
    throw new Error('Perfect Chaos prefix manifest has no boundaries.');
  }
  const expectedBoundaries = roleBoundaries(reference.boundaries.at(-1));
  if (JSON.stringify(reference.boundaries) !== JSON.stringify(expectedBoundaries)) {
    throw new Error('Perfect Chaos prefix manifest boundaries are not contiguous even layers.');
  }
  const directory = dirname(referencePath);
  const sourceHash = createHash('sha256').update(await readFile(SOURCE)).digest('hex');
  const verificationSourceHash = reference.verificationSourceSha256 ?? reference.sourceSha256;
  if (verificationSourceHash !== sourceHash) {
    throw new Error('The prefix verifier source does not match the committed manifest.');
  }
  if (reference.generatorSource !== undefined) {
    if (typeof reference.generatorSource !== 'string' || reference.generatorSource.length === 0) {
      throw new Error('The prefix generator source path is invalid.');
    }
    const generatorPath = resolve(directory, reference.generatorSource);
    const generatorRelative = relative(directory, generatorPath);
    if (generatorRelative === '..' || generatorRelative.startsWith('../')
        || generatorRelative.startsWith('..\\') || isAbsolute(generatorRelative)) {
      throw new Error('The prefix generator source escapes the certificate directory.');
    }
    const generatorSourceHash = createHash('sha256')
      .update(await readFile(generatorPath))
      .digest('hex');
    if (generatorSourceHash !== reference.sourceSha256) {
      throw new Error('The preserved prefix generator source does not match the manifest.');
    }
  }

  const native = await nativeSegment(binary, ['verify', '--directory', scratch]);
  if (native.code !== 0) throw new Error(`Native prefix verification failed.\n${native.stderr}`);

  const replay = {};
  for (const roleName of ['red', 'yellow']) {
    const expectedArtifacts = reference.artifacts?.[roleName];
    if (!Array.isArray(expectedArtifacts)) {
      throw new Error(`Manifest is missing ${roleName} artifacts.`);
    }
    for (const expected of expectedArtifacts) {
      const actual = await hashFile(join(directory, roleName, expected.path));
      if (actual.bytes !== expected.bytes || actual.sha256 !== expected.sha256) {
        throw new Error(`Artifact mismatch for ${roleName}/${expected.path}.`);
      }
    }
    replay[roleName] = await replayRole(directory, roleName, reference.boundaries);
    if (JSON.stringify(stable(replay[roleName]))
        !== JSON.stringify(stable(reference.roles?.[roleName]?.replay))) {
      throw new Error(`Independent replay mismatch for the ${roleName} certificate.`);
    }
  }
  return { reference, native: native.records, replay };
}


async function verifyPolicyConflicts(temporary) {
  const directory = join(temporary, 'policy-conflict');
  await mkdir(directory, { recursive: true });
  const state = {
    mover: 0n,
    opponent: 0n,
    rows: 6,
    columns: 7,
    aiTurn: true,
  };
  const flip = { state, action: { type: ACTION_FLIP, column: 0 } };
  const rotate = { state, action: { type: ACTION_CW, column: 0 } };

  let encodingRejected = false;
  try {
    encodePolicy(ROLE_CODES.red, 2, [flip, rotate]);
  } catch (error) {
    if (!/Conflicting Perfect Chaos policy actions/.test(String(error))) throw error;
    encodingRejected = true;
  }
  if (!encodingRejected) {
    throw new Error('The Perfect Chaos policy encoder silently selected a conflicting action.');
  }

  const first = join(directory, 'first.policy.bin');
  const second = join(directory, 'second.policy.bin');
  const merged = join(directory, 'merged.policy.bin');
  await writeFile(first, encodePolicy(ROLE_CODES.red, 2, [flip]));
  await writeFile(second, encodePolicy(ROLE_CODES.red, 2, [rotate]));
  let mergeRejected = false;
  try {
    await mergePolicies(merged, [first, second]);
  } catch (error) {
    if (!/Conflicting Perfect Chaos policy actions/.test(String(error))) throw error;
    mergeRejected = true;
  }
  if (!mergeRejected) {
    throw new Error('The Perfect Chaos policy merger silently selected a conflicting action.');
  }
  if (await exists(merged)) {
    throw new Error('The Perfect Chaos policy merger wrote output after a conflict.');
  }
  return { encodingRejected, mergeRejected };
}

async function verifyShardedSmall(binary, temporary) {
  const results = {};
  for (const [roleName, expectedFrontier] of [['red', 327], ['yellow', 974]]) {
    const roleDirectory = join(temporary, `sharded-${roleName}`);
    await mkdir(roleDirectory, { recursive: true });
    const firstPolicy = join(roleDirectory, '0-4.policy.bin');
    const firstFrontier = join(roleDirectory, '0-4.frontier.bin');
    const generated = await nativeSegment(binary, [
      'generate',
      '--role', roleName,
      '--frontier-pieces', '4',
      '--maximum-states', '1000000',
      '--policy', firstPolicy,
      '--frontier', firstFrontier,
    ]);
    if (generated.code !== 0) {
      throw new Error(`Small ${roleName} root generation failed.\n${generated.stderr}`);
    }

    const policyPath = join(roleDirectory, '4-6.policy.bin');
    const frontierPath = join(roleDirectory, '4-6.frontier.bin');
    const rejectedPath = join(roleDirectory, 'new-reject-4.bin');
    const sharded = await shardedNativeExtension({
      binary,
      inputFrontier: firstFrontier,
      targetBoundary: 6,
      maximumStateCount: 2_000_000,
      policyPath,
      frontierPath,
      targetReject: null,
      rejectedPath,
      shardCount: 2,
      shardWorkers: 2,
    });
    if (sharded.code !== 0) {
      throw new Error(`Small ${roleName} sharded extension unexpectedly rejected a root.`);
    }

    const inputStates = (await readFrontier(firstFrontier)).states;
    const replay = await replaySegment({
      role: ROLE_CODES[roleName],
      inputStates,
      policyPath,
      frontierPath,
      boundary: 6,
    });
    if (sharded.records.at(-1)?.shardWorkers !== 2) {
      throw new Error(`Small ${roleName} sharded extension did not use two workers.`);
    }
    if (replay.frontierStates !== expectedFrontier) {
      throw new Error(
        `Small ${roleName} sharded frontier mismatch: `
        + `${replay.frontierStates} instead of ${expectedFrontier}.`,
      );
    }
    results[roleName] = { summary: sharded.records.at(-1), replay };
  }

  const adaptiveDirectory = join(temporary, 'adaptive-red');
  await mkdir(adaptiveDirectory, { recursive: true });
  const adaptiveRootPolicy = join(adaptiveDirectory, '0-4.policy.bin');
  const adaptiveRootFrontier = join(adaptiveDirectory, '0-4.frontier.bin');
  const adaptiveRoot = await nativeSegment(binary, [
    'generate',
    '--role', 'red',
    '--frontier-pieces', '4',
    '--maximum-states', '1000000',
    '--policy', adaptiveRootPolicy,
    '--frontier', adaptiveRootFrontier,
  ]);
  if (adaptiveRoot.code !== 0) {
    throw new Error(`Adaptive sharding root generation failed.\n${adaptiveRoot.stderr}`);
  }
  const adaptivePolicy = join(adaptiveDirectory, '4-6.policy.bin');
  const adaptiveFrontier = join(adaptiveDirectory, '4-6.frontier.bin');
  const adaptiveRejected = join(adaptiveDirectory, 'new-reject-4.bin');
  const adaptive = await shardedNativeExtension({
    binary,
    inputFrontier: adaptiveRootFrontier,
    targetBoundary: 6,
    maximumStateCount: 10_000,
    minimumStatesPerShard: 10_000,
    policyPath: adaptivePolicy,
    frontierPath: adaptiveFrontier,
    targetReject: null,
    rejectedPath: adaptiveRejected,
    shardCount: 2,
    shardWorkers: 2,
  });
  if (adaptive.code !== 0) {
    throw new Error('Adaptive sharding unexpectedly rejected a safe root.');
  }
  const adaptiveSummary = adaptive.records.at(-1);
  if (!adaptiveSummary || adaptiveSummary.adaptiveSplits < 1
      || adaptiveSummary.shards <= adaptiveSummary.requestedShards
      || adaptiveSummary.shardWorkers !== 2) {
    throw new Error('Adaptive sharding did not subdivide the oversized test shard.');
  }
  const adaptiveReplay = await replaySegment({
    role: ROLE_CODES.red,
    inputStates: (await readFrontier(adaptiveRootFrontier)).states,
    policyPath: adaptivePolicy,
    frontierPath: adaptiveFrontier,
    boundary: 6,
  });
  if (adaptiveReplay.frontierStates !== 327) {
    throw new Error('Adaptive sharding changed the certified frontier.');
  }
  results.adaptive = { summary: adaptiveSummary, replay: adaptiveReplay };
  return results;
}

async function verifySmall(binary, temporary, build) {
  const native = await nativeSegment(binary, ['verify', '--directory', temporary]);
  if (native.code !== 0) throw new Error(`Native prefix verification failed.\n${native.stderr}`);
  const expected = [
    ['red', 0, 4, 101, 59],
    ['red', 4, 6, 598, 327],
    ['yellow', 0, 4, 302, 172],
    ['yellow', 4, 6, 1754, 974],
  ];
  if (native.records.length !== expected.length) throw new Error('Native prefix verifier returned the wrong case count.');
  expected.forEach(([role, from, to, closure, frontier], index) => {
    const record = native.records[index];
    if (record.role !== role || record.fromPieces !== from || record.frontierPieces !== to
        || record.closureStates !== closure || record.frontierStates !== frontier) {
      throw new Error(`Native prefix verification mismatch: ${JSON.stringify(record)}`);
    }
  });
  const largeFrontierMerge = await verifyLargeFrontierMerge(temporary);
  const sharding = await verifyShardedSmall(binary, temporary);
  const policyConflicts = await verifyPolicyConflicts(temporary);
  const generated = join(temporary, 'small-reference');
  const journal = await createJournal(join(temporary, 'small-journal'), build);
  const manifest = await generateReference(
    binary, generated, 8, 20, null, 1, 14, 2_000_000, 1, journal,
  );
  if (manifest.roles.red.replay.segments.at(-1).frontierStates !== 1477
      || manifest.roles.yellow.replay.segments.at(-1).frontierStates !== 4515) {
    throw new Error('Independent replay did not reproduce the eight-piece reference frontiers.');
  }
  const freshJournal = journal.summary();
  if (freshJournal.hits !== 0 || freshJournal.misses < 2 || freshJournal.stores < 2) {
    throw new Error('A fresh prefix journal did not record deterministic segment misses.');
  }

  journal.resetStatistics();
  const regenerated = join(temporary, 'small-reference-journaled');
  const rerun = await generateReference(
    binary, regenerated, 8, 20, null, 1, 14, 2_000_000, 1, journal,
  );
  const reusedJournal = journal.summary();
  if (reusedJournal.misses !== 0 || reusedJournal.hits < 2) {
    throw new Error('The prefix journal did not reuse completed deterministic segments.');
  }
  if (JSON.stringify(stable(rerun)) !== JSON.stringify(stable(manifest))) {
    throw new Error('Journal-backed regeneration diverged from the fresh reference.');
  }

  const probe = { kind: 'probe', inputSha256: 'a'.repeat(64) };
  const keyA = journalKey(journal, probe);
  const keyB = journalKey(journal, { ...probe, inputSha256: 'b'.repeat(64) });
  if (keyA === keyB
      || keyA === journalKey({ ...journal, sourceSha256: '0'.repeat(64) }, probe)
      || keyA === journalKey({ ...journal, flags: [...journal.flags, '-O0'] }, probe)) {
    throw new Error('The prefix journal key is not bound to exact inputs, source and flags.');
  }

  const corrupted = await corruptOneJournalOutput(journal);
  journal.resetStatistics();
  const recoveredOutput = join(temporary, 'small-reference-recovered');
  const recovered = await generateReference(
    binary, recoveredOutput, 8, 20, null, 1, 14, 2_000_000, 1, journal,
  );
  const recoveredJournal = journal.summary();
  if (recoveredJournal.invalidations < 1 || recoveredJournal.misses < 1
      || recoveredJournal.hits < 1) {
    throw new Error('A corrupted journal entry was not safely invalidated and regenerated.');
  }
  if (JSON.stringify(stable(recovered)) !== JSON.stringify(stable(manifest))) {
    throw new Error('Recovery from a corrupted journal entry changed the certificate.');
  }
  return {
    native: native.records,
    largeFrontierMerge,
    sharding,
    policyConflicts,
    replay: manifest.roles,
    journal: {
      fresh: freshJournal,
      reused: reusedJournal,
      corrupted,
      recovered: recoveredJournal,
    },
  };
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const temporary = await mkdtemp(join(tmpdir(), 'connect4-chaos-prefix-'));
  try {
    const { compiler, binary, build } = await compile();
    if (options.command === 'verify') {
      const result = await verifySmall(binary, temporary, build);
      process.stdout.write(`${JSON.stringify({ compiler, ...result }, null, 2)}\n`);
      return;
    }
    if (options.command === 'generate') {
      const target = integerOption(options.frontier_pieces, 12, 'frontier-pieces', 8, 42);
      const output = resolve(options.output ?? join(ROOT, 'generated', `perfect-chaos-prefix-${target}`));
      const passes = integerOption(options.maximum_passes, 200, 'maximum-passes', 1, 10_000);
      const seedDirectory = options.seed_rejections ? resolve(options.seed_rejections) : null;
      const shards = integerOption(options.shards, 1, 'shards', 1, 256);
      const shardWorkers = integerOption(options.shard_workers, 1, 'shard-workers', 1, 32);
      const shardFromBoundary = integerOption(
        options.shard_from_pieces,
        14,
        'shard-from-pieces',
        2,
        42,
      );
      const journal = await createJournal(journalDirectory(options, output), build);
      const manifest = await generateReference(
        binary,
        output,
        target,
        passes,
        seedDirectory,
        shards,
        shardFromBoundary,
        2_000_000,
        shardWorkers,
        journal,
      );
      process.stdout.write(`${JSON.stringify({
        compiler,
        output,
        manifest,
        ...(journal ? { journal: journal.summary() } : {}),
      }, null, 2)}
`);
      return;
    }
    if (options.command === 'verify-reference') {
      const referencePath = resolve(options.reference ?? join(ROOT, 'data', 'perfect-chaos-prefix', 'manifest.json'));
      const verified = await verifyCommittedReference(referencePath, binary, temporary);
      process.stdout.write(`${JSON.stringify({
        compiler,
        verified: referencePath,
        native: verified.native,
        replay: verified.replay,
      }, null, 2)}\n`);
      return;
    }
    if (options.command === 'reproduce-reference') {
      const referencePath = resolve(options.reference ?? join(ROOT, 'data', 'perfect-chaos-prefix', 'manifest.json'));
      const reference = JSON.parse(await readFile(referencePath, 'utf8'));
      const output = resolve(options.output ?? join(ROOT, 'generated', `perfect-chaos-prefix-${reference.boundaries.at(-1)}`));
      const passes = integerOption(options.maximum_passes, 500, 'maximum-passes', 1, 10_000);
      const seedDirectory = resolve(options.seed_rejections ?? dirname(referencePath));
      const shards = integerOption(
        options.shards,
        reference.sharding?.count ?? 1,
        'shards',
        1,
        256,
      );
      const shardFromBoundary = integerOption(
        options.shard_from_pieces,
        reference.sharding?.fromBoundary ?? 14,
        'shard-from-pieces',
        2,
        42,
      );
      const shardWorkers = integerOption(
        options.shard_workers,
        reference.sharding?.workers ?? 1,
        'shard-workers',
        1,
        32,
      );
      const journal = await createJournal(journalDirectory(options, output), build);
      const generated = await generateReference(
        binary,
        output,
        reference.boundaries.at(-1),
        passes,
        seedDirectory,
        shards,
        shardFromBoundary,
        2_000_000,
        shardWorkers,
        journal,
        [dirname(referencePath)],
      );
      if (!reproducesReference(generated, reference)) {
        throw new Error(
          'Regenerated Perfect Chaos prefix certificates or summaries do not match the committed reference.',
        );
      }
      process.stdout.write(`${JSON.stringify({
        compiler,
        reproduced: referencePath,
        output,
        ...(journal ? { journal: journal.summary() } : {}),
      }, null, 2)}
`);
      return;
    }
    throw new RangeError(`Unknown command: ${options.command}`);
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

if (isEntryPoint(import.meta.url)) await main();

export {
  createJournal,
  encodeFrontier,
  encodePolicy,
  journalKey,
  journaledSegment,
  readFrontier,
  replaySegment,
};
