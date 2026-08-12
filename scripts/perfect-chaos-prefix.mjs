#!/usr/bin/env node

import { constants as fsConstants } from 'node:fs';
import {
  access,
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  stat,
  writeFile,
} from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
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

function integerOption(value, fallback, label, minimum = 0, maximum = Number.MAX_SAFE_INTEGER) {
  const selected = value === undefined ? fallback : Number.parseInt(String(value), 10);
  if (!Number.isInteger(selected) || selected < minimum || selected > maximum) {
    throw new RangeError(`${label} must be an integer from ${minimum} to ${maximum}.`);
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

async function compile(directory) {
  const compiler = await findCompiler();
  const binary = join(directory, 'perfect-chaos-prefix');
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
    throw new Error(`Prefix compiler failed.\n${result.stderr || result.stdout}`);
  }
  return { compiler, binary };
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


function compareAction(first, second) {
  if (first.type !== second.type) return first.type - second.type;
  return first.column - second.column;
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
    if (!existing || compareAction(record.action, existing.action) < 0) {
      selected.set(key, record);
    }
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
  let conflicts = 0;
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
      if (existing && !sameAction(existing.action, record.action)) conflicts += 1;
      if (!existing || compareAction(record.action, existing.action) < 0) {
        records.set(key, record);
      }
    }
  }
  await writeFile(target, encodePolicy(role, boundary, [...records.values()]));
  return { count: (await readPolicy(target)).count, conflicts };
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
    states.push(...frontier.states);
  }
  await writeFile(target, encodeFrontier(role, boundary, states));
  return (await readFrontier(target)).count;
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

async function replaySegment({ role, inputStates, policyPath, frontierPath }) {
  const policy = await readPolicy(policyPath);
  const expected = await readFrontier(frontierPath);
  if (policy.role !== role || expected.role !== role || policy.boundary !== expected.boundary) {
    throw new Error('Policy/frontier role or boundary mismatch.');
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
    const summary = await replaySegment({ role, inputStates, policyPath, frontierPath });
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
    const pending = inputPaths.map((inputPath, index) => ({
      inputPath,
      label: String(index).padStart(3, '0'),
      depth: 0,
    }));
    const maximumAttempts = Math.max(1_024, source.count * 2 + inputPaths.length);
    let attempts = 0;
    let adaptiveSplits = 0;
    let maximumSplitDepth = 0;

    while (pending.length > 0) {
      attempts += 1;
      if (attempts > maximumAttempts) {
        throw new Error(
          `Adaptive sharding exceeded ${maximumAttempts.toLocaleString()} attempts.`,
        );
      }
      const task = pending.shift();
      const prefix = join(shardDirectory, `leaf-${task.label}`);
      const shardPolicy = `${prefix}.policy.bin`;
      const shardFrontier = `${prefix}.frontier.bin`;
      const shardRejected = `${prefix}.rejected.bin`;
      await Promise.all([
        rm(shardPolicy, { force: true }),
        rm(shardFrontier, { force: true }),
        rm(shardRejected, { force: true }),
      ]);

      const result = await nativeSegment(binary, [
        'extend',
        '--input-frontier', task.inputPath,
        '--frontier-pieces', String(targetBoundary),
        '--maximum-states', String(maximumStatesPerShard),
        '--policy', shardPolicy,
        '--frontier', shardFrontier,
        ...(targetReject ? ['--reject-frontier', targetReject] : []),
        '--rejected', shardRejected,
      ]);

      if (result.code === 0) {
        const summary = result.records.at(-1);
        if (!summary) throw new Error(`Shard ${task.label} returned no certificate summary.`);
        summaries.push(summary);
        policyPaths.push(shardPolicy);
        frontierPaths.push(shardFrontier);
        continue;
      }

      if (await exists(shardRejected)) {
        rejectedPaths.push(shardRejected);
        continue;
      }

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
        const childDepth = task.depth + 1;
        adaptiveSplits += 1;
        maximumSplitDepth = Math.max(maximumSplitDepth, childDepth);
        pending.splice(0, 0, ...childPaths.map((inputPath, index) => ({
          inputPath,
          label: `${task.label}.${index}`,
          depth: childDepth,
        })));
        await rm(task.inputPath, { force: true });
        continue;
      }

      throw new Error(
        `Shard ${task.label} failed without a rejection certificate.
`
        + (result.stderr || result.stdout),
      );
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
  const buffer = await readFile(path);
  return {
    path: basename(path),
    bytes: buffer.length,
    sha256: createHash('sha256').update(buffer).digest('hex'),
  };
}

async function initializeRejections(output, roleName, boundaries, seedDirectory) {
  const roleDirectory = join(output, roleName);
  await mkdir(roleDirectory, { recursive: true });
  const rejects = new Map();
  for (const boundary of boundaries) {
    const path = join(roleDirectory, `reject-${boundary}.bin`);
    const seed = seedDirectory ? join(seedDirectory, roleName, `reject-${boundary}.bin`) : null;
    if (seed) {
      try {
        await copyFile(seed, path);
        const decoded = await readFrontier(path);
        if (decoded.role !== ROLE_CODES[roleName] || decoded.boundary !== boundary) {
          throw new Error(`Seed rejection file has the wrong role or boundary: ${seed}`);
        }
      } catch (error) {
        if (error?.code !== 'ENOENT') throw error;
        await writeFile(path, encodeFrontier(ROLE_CODES[roleName], boundary, []));
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
  allowIncomplete = false,
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
        })
        : await nativeSegment(binary, args);
      if (result.code === 0) {
        nativeSummaries.push(result.records.at(-1));
        inputFrontier = frontierPath;
        from = boundary;
        continue;
      }
      if (from === 0) {
        throw new Error(`Root prefix became losing.\n${result.stderr || result.stdout}`);
      }
      const previousReject = rejects.get(from);
      const before = (await readFrontier(previousReject)).count;
      const merged = join(roleDirectory, `merged-reject-${from}.bin`);
      const count = await mergeFrontiers(merged, [previousReject, newRejectPath]);
      if (count <= before) throw new Error(`Prefix refinement at ${from} pieces made no progress.`);
      await rm(previousReject, { force: true });
      await writeFile(previousReject, await readFile(merged));
      await rm(merged, { force: true });
      restart = true;
      break;
    }

    if (restart) continue;
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

async function prepareRole(
  binary,
  output,
  roleName,
  targetBoundaries,
  maximumPasses,
  seedDirectory = null,
  shardCount = 1,
  shardFromBoundary = 14,
  minimumStatesPerShard = 2_000_000,
  allowIncomplete = false,
) {
  if (targetBoundaries.length < 2) {
    throw new RangeError('A prepared prefix requires at least two boundaries.');
  }
  const preparedBoundaries = targetBoundaries.slice(0, -1);
  const { rejects } = await initializeRejections(
    output,
    roleName,
    preparedBoundaries,
    seedDirectory,
  );
  const roleDirectory = join(output, roleName);

  for (let pass = 1; pass <= maximumPasses; pass += 1) {
    let from = 0;
    let inputFrontier = null;
    const nativeSummaries = [];
    let restart = false;

    for (const boundary of preparedBoundaries) {
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
          '--reject-frontier', targetReject,
        ]
        : [
          'extend', '--input-frontier', inputFrontier,
          '--frontier-pieces', String(boundary),
          '--maximum-states', String(maximumStates(boundary)),
          '--policy', policyPath,
          '--frontier', frontierPath,
          '--reject-frontier', targetReject,
          '--rejected', newRejectPath,
        ];
      const useShards = from > 0 && shardCount > 1 && boundary >= shardFromBoundary;
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
        })
        : await nativeSegment(binary, args);
      if (result.code === 0) {
        nativeSummaries.push(result.records.at(-1));
        inputFrontier = frontierPath;
        from = boundary;
        continue;
      }
      if (from === 0) {
        throw new Error(`Prepared root prefix became losing.\n${result.stderr || result.stdout}`);
      }
      const previousReject = rejects.get(from);
      const before = (await readFrontier(previousReject)).count;
      const merged = join(roleDirectory, `merged-reject-${from}.bin`);
      const count = await mergeFrontiers(merged, [previousReject, newRejectPath]);
      if (count <= before) {
        throw new Error(`Prepared prefix refinement at ${from} pieces made no progress.`);
      }
      await writeFile(previousReject, await readFile(merged));
      await rm(merged, { force: true });
      restart = true;
      break;
    }

    if (restart) continue;
    const replay = await replayRole(output, roleName, preparedBoundaries);
    return {
      nativeSummaries,
      replay,
      rejected: await rejectionCounts(rejects),
      preparedFrontier: preparedBoundaries.at(-1),
      targetFrontier: targetBoundaries.at(-1),
    };
  }

  if (!allowIncomplete) {
    throw new Error(`${roleName} prefix preparation exceeded ${maximumPasses} refinement passes.`);
  }
  return {
    incomplete: true,
    passes: maximumPasses,
    rejected: await rejectionCounts(rejects),
    preparedFrontier: preparedBoundaries.at(-1),
    targetFrontier: targetBoundaries.at(-1),
  };
}

async function cleanIncompleteRoleDirectory(roleDirectory) {
  const entries = await readdir(roleDirectory, { withFileTypes: true });
  await Promise.all(entries.map(async (entry) => {
    if (!entry.isFile() || /^reject-\d+\.bin$/.test(entry.name)) return;
    await rm(join(roleDirectory, entry.name), { force: true });
  }));
}

async function writeRoleCheckpoint({
  output,
  roleName,
  target,
  boundaries,
  complete,
  passBudget,
  shardCount,
  shardFromBoundary,
  minimumStatesPerShard,
  result,
  mode,
}) {
  const roleDirectory = join(output, roleName);
  if (!complete) await cleanIncompleteRoleDirectory(roleDirectory);
  const preparedBoundaries = mode === 'preparation' ? boundaries.slice(0, -1) : boundaries;
  const artifacts = [];
  const counts = {};
  for (const boundary of boundaries.slice(0, -1)) {
    const path = join(roleDirectory, `reject-${boundary}.bin`);
    const decoded = await readFrontier(path);
    counts[`at${boundary}`] = decoded.count;
    artifacts.push(await hashFile(path));
  }
  if (complete) {
    let from = 0;
    for (const boundary of preparedBoundaries) {
      artifacts.push(
        await hashFile(join(roleDirectory, `${from}-${boundary}.policy.bin`)),
        await hashFile(join(roleDirectory, `${from}-${boundary}.frontier.bin`)),
      );
      from = boundary;
    }
  }

  const checkpoint = {
    format: 'connect4-chaos-prefix-role-checkpoint-v1',
    theorem: 'finite-safety-game-with-quotient-cycles-lifting-to-threefold-draws',
    sourceSha256: createHash('sha256').update(await readFile(SOURCE)).digest('hex'),
    role: roleName,
    mode,
    target,
    boundaries,
    complete,
    passBudget,
    sharding: {
      count: shardCount,
      fromBoundary: shardFromBoundary,
      minimumStatesPerShard,
    },
    rejectionCounts: counts,
    artifacts,
    ...(complete ? { result } : {}),
  };
  await writeFile(join(output, 'checkpoint.json'), `${JSON.stringify(checkpoint, null, 2)}
`);
  return checkpoint;
}

function roleBoundaries(target) {
  if (target < 8 || target % 2 !== 0) {
    throw new RangeError('The checkpoint frontier must be an even piece count of at least 8.');
  }
  const boundaries = [8];
  for (let boundary = 10; boundary <= target; boundary += 2) boundaries.push(boundary);
  return boundaries;
}

async function checkpointRole({
  binary,
  output,
  roleName,
  target,
  passBudget,
  seedDirectory,
  shardCount,
  shardFromBoundary,
  minimumStatesPerShard,
  mode = 'synthesis',
}) {
  if (!Object.hasOwn(ROLE_CODES, roleName)) {
    throw new RangeError(`Unknown Perfect Chaos role: ${roleName}`);
  }
  const boundaries = roleBoundaries(target);
  if (seedDirectory && resolve(seedDirectory) === resolve(output)) {
    throw new RangeError('The checkpoint output must differ from its seed directory.');
  }
  await rm(output, { recursive: true, force: true });
  await mkdir(output, { recursive: true });
  const result = mode === 'preparation'
    ? await prepareRole(
      binary,
      output,
      roleName,
      boundaries,
      passBudget,
      seedDirectory,
      shardCount,
      shardFromBoundary,
      minimumStatesPerShard,
      true,
    )
    : await generateRole(
      binary,
      output,
      roleName,
      boundaries,
      passBudget,
      seedDirectory,
      shardCount,
      shardFromBoundary,
      minimumStatesPerShard,
      true,
    );
  const complete = !result.incomplete;
  return writeRoleCheckpoint({
    output,
    roleName,
    target,
    boundaries,
    complete,
    passBudget,
    shardCount,
    shardFromBoundary,
    minimumStatesPerShard,
    result,
    mode,
  });
}

async function assembleReference(output, target, generation = null) {
  const boundaries = roleBoundaries(target);
  const roles = {};
  const artifacts = {};
  for (const roleName of ['red', 'yellow']) {
    const roleDirectory = join(output, roleName);
    const replay = await replayRole(output, roleName, boundaries);
    const rejected = {};
    const files = [];
    let from = 0;
    for (const boundary of boundaries) {
      files.push(
        join(roleDirectory, `${from}-${boundary}.policy.bin`),
        join(roleDirectory, `${from}-${boundary}.frontier.bin`),
      );
      from = boundary;
    }
    for (const boundary of boundaries.slice(0, -1)) {
      const rejectionPath = join(roleDirectory, `reject-${boundary}.bin`);
      rejected[`at${boundary}`] = (await readFrontier(rejectionPath)).count;
      files.push(rejectionPath);
    }
    roles[roleName] = { nativeSummaries: [], replay, rejected };
    artifacts[roleName] = [];
    for (const path of files) artifacts[roleName].push(await hashFile(path));
  }

  const manifest = {
    format: 'connect4-chaos-layered-prefix-manifest-v1',
    theorem: 'finite-safety-game-with-quotient-cycles-lifting-to-threefold-draws',
    board: { rows: 6, columns: 7, connect: 4, chaosMode: true },
    boundaries,
    ...(generation ? { generation } : {}),
    sourceSha256: createHash('sha256').update(await readFile(SOURCE)).digest('hex'),
    roles,
    artifacts,
  };
  await writeFile(join(output, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  return manifest;
}


async function generateReference(
  binary,
  output,
  target,
  maximumPasses = 200,
  seedDirectory = null,
  shardCount = 1,
  shardFromBoundary = 14,
  minimumStatesPerShard = 2_000_000,
) {
  if (target < 8 || target % 2 !== 0) {
    throw new RangeError('The reference frontier must be an even piece count of at least 8.');
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
    ...(shardCount > 1 ? { sharding: { count: shardCount, fromBoundary: shardFromBoundary } } : {}),
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

async function verifyCommittedReference(referencePath, binary) {
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
  if (reference.sourceSha256 !== sourceHash) {
    throw new Error('The prefix solver source does not match the committed manifest.');
  }

  const native = await nativeSegment(binary, ['verify']);
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
    });
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
    shardCount: 1,
  });
  if (adaptive.code !== 0) {
    throw new Error('Adaptive sharding unexpectedly rejected a safe root.');
  }
  const adaptiveSummary = adaptive.records.at(-1);
  if (!adaptiveSummary || adaptiveSummary.adaptiveSplits < 1
      || adaptiveSummary.shards <= adaptiveSummary.requestedShards) {
    throw new Error('Adaptive sharding did not subdivide the oversized test shard.');
  }
  const adaptiveReplay = await replaySegment({
    role: ROLE_CODES.red,
    inputStates: (await readFrontier(adaptiveRootFrontier)).states,
    policyPath: adaptivePolicy,
    frontierPath: adaptiveFrontier,
  });
  if (adaptiveReplay.frontierStates !== 327) {
    throw new Error('Adaptive sharding changed the certified frontier.');
  }
  results.adaptive = { summary: adaptiveSummary, replay: adaptiveReplay };
  return results;
}

async function verifySmall(binary, temporary) {
  const native = await nativeSegment(binary, ['verify']);
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
  const sharding = await verifyShardedSmall(binary, temporary);
  const generated = join(temporary, 'small-reference');
  const manifest = await generateReference(binary, generated, 8, 20);
  if (manifest.roles.red.replay.segments.at(-1).frontierStates !== 1477
      || manifest.roles.yellow.replay.segments.at(-1).frontierStates !== 4515) {
    throw new Error('Independent replay did not reproduce the eight-piece reference frontiers.');
  }
  return { native: native.records, sharding, replay: manifest.roles };
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const temporary = await mkdtemp(join(tmpdir(), 'connect4-chaos-prefix-'));
  try {
    const { compiler, binary } = await compile(temporary);
    if (options.command === 'verify') {
      const result = await verifySmall(binary, temporary);
      process.stdout.write(`${JSON.stringify({ compiler, ...result }, null, 2)}\n`);
      return;
    }
    if (options.command === 'replay-role') {
      const roleName = String(options.role ?? '');
      if (!Object.hasOwn(ROLE_CODES, roleName)) {
        throw new RangeError(`Unknown Perfect Chaos role: ${roleName}`);
      }
      const target = integerOption(options.frontier_pieces, 16, 'frontier-pieces', 8, 42);
      const directory = resolve(options.directory ?? join(ROOT, 'data', 'perfect-chaos-prefix'));
      const replay = await replayRole(directory, roleName, roleBoundaries(target));
      process.stdout.write(`${JSON.stringify({ compiler, directory, replay }, null, 2)}\n`);
      return;
    }
    if (options.command === 'assemble-reference') {
      const target = integerOption(options.frontier_pieces, 16, 'frontier-pieces', 8, 42);
      const directory = resolve(options.directory ?? join(ROOT, 'generated', `perfect-chaos-prefix-${target}`));
      const manifest = await assembleReference(directory, target, {
        method: String(options.method ?? 'distributed-frontier-classification-v1'),
      });
      const verified = await verifyCommittedReference(join(directory, 'manifest.json'), binary);
      process.stdout.write(`${JSON.stringify({ compiler, directory, manifest, replay: verified.replay }, null, 2)}\n`);
      return;
    }
    if (options.command === 'advance-role' || options.command === 'prepare-role') {
      const roleName = String(options.role ?? '');
      const target = integerOption(options.frontier_pieces, 16, 'frontier-pieces', 8, 42);
      const output = resolve(
        options.output ?? join(ROOT, 'generated', `perfect-chaos-${roleName}-${target}-checkpoint`),
      );
      const passBudget = integerOption(options.pass_budget, 10, 'pass-budget', 1, 10_000);
      const seedDirectory = options.seed_rejections ? resolve(options.seed_rejections) : null;
      const shards = integerOption(options.shards, 1, 'shards', 1, 256);
      const shardFromBoundary = integerOption(
        options.shard_from_pieces,
        14,
        'shard-from-pieces',
        2,
        42,
      );
      const minimumStatesPerShard = integerOption(
        options.minimum_states_per_shard,
        2_000_000,
        'minimum-states-per-shard',
        10_000,
        100_000_000,
      );
      const checkpoint = await checkpointRole({
        binary,
        output,
        roleName,
        target,
        passBudget,
        seedDirectory,
        shardCount: shards,
        shardFromBoundary,
        minimumStatesPerShard,
        mode: options.command === 'prepare-role' ? 'preparation' : 'synthesis',
      });
      process.stdout.write(`${JSON.stringify({ compiler, output, checkpoint }, null, 2)}\n`);
      return;
    }
    if (options.command === 'generate') {
      const target = integerOption(options.frontier_pieces, 12, 'frontier-pieces', 8, 42);
      const output = resolve(options.output ?? join(ROOT, 'generated', `perfect-chaos-prefix-${target}`));
      const passes = integerOption(options.maximum_passes, 200, 'maximum-passes', 1, 10_000);
      const seedDirectory = options.seed_rejections ? resolve(options.seed_rejections) : null;
      const shards = integerOption(options.shards, 1, 'shards', 1, 256);
      const shardFromBoundary = integerOption(
        options.shard_from_pieces,
        14,
        'shard-from-pieces',
        2,
        42,
      );
      const manifest = await generateReference(
        binary,
        output,
        target,
        passes,
        seedDirectory,
        shards,
        shardFromBoundary,
      );
      process.stdout.write(`${JSON.stringify({ compiler, output, manifest }, null, 2)}\n`);
      return;
    }
    if (options.command === 'verify-reference') {
      const referencePath = resolve(options.reference ?? join(ROOT, 'data', 'perfect-chaos-prefix', 'manifest.json'));
      const verified = await verifyCommittedReference(referencePath, binary);
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
      const generated = await generateReference(
        binary,
        output,
        reference.boundaries.at(-1),
        passes,
        seedDirectory,
      );
      if (JSON.stringify(stable(generated)) !== JSON.stringify(stable(reference))) {
        throw new Error('Regenerated Perfect Chaos prefix manifest does not match the committed reference.');
      }
      process.stdout.write(`${JSON.stringify({ compiler, reproduced: referencePath, output }, null, 2)}\n`);
      return;
    }
    throw new RangeError(`Unknown command: ${options.command}`);
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

await main();
