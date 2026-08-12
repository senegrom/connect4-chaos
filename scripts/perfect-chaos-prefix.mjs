#!/usr/bin/env node

import { constants as fsConstants } from 'node:fs';
import {
  access,
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
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

async function hashFile(path) {
  const buffer = await readFile(path);
  return {
    path: basename(path),
    bytes: buffer.length,
    sha256: createHash('sha256').update(buffer).digest('hex'),
  };
}

async function generateRole(binary, output, roleName, boundaries, maximumPasses, seedDirectory = null) {
  const roleDirectory = join(output, roleName);
  await mkdir(roleDirectory, { recursive: true });
  const rejects = new Map();
  for (const boundary of boundaries.slice(0, -1)) {
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

  for (let pass = 1; pass <= maximumPasses; pass += 1) {
    let from = 0;
    let inputFrontier = null;
    const nativeSummaries = [];
    let restart = false;

    for (const boundary of boundaries) {
      const policyPath = join(roleDirectory, `${from}-${boundary}.policy.bin`);
      const frontierPath = join(roleDirectory, `${from}-${boundary}.frontier.bin`);
      const targetReject = rejects.get(boundary);
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
          '--rejected', join(roleDirectory, `new-reject-${from}.bin`),
        ];
      const result = await nativeSegment(binary, args);
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
      const newReject = join(roleDirectory, `new-reject-${from}.bin`);
      const before = (await readFrontier(previousReject)).count;
      const merged = join(roleDirectory, `merged-reject-${from}.bin`);
      const count = await mergeFrontiers(merged, [previousReject, newReject]);
      if (count <= before) throw new Error(`Prefix refinement at ${from} pieces made no progress.`);
      await rm(previousReject, { force: true });
      await writeFile(previousReject, await readFile(merged));
      await rm(merged, { force: true });
      restart = true;
      break;
    }

    if (restart) continue;
    const replay = await replayRole(output, roleName, boundaries);
    const rejected = {};
    for (const [boundary, path] of rejects) {
      rejected[`at${boundary}`] = (await readFrontier(path)).count;
    }
    return { nativeSummaries, replay, rejected };
  }
  throw new Error(`${roleName} prefix synthesis exceeded ${maximumPasses} refinement passes.`);
}

async function generateReference(binary, output, target, maximumPasses = 200, seedDirectory = null) {
  if (target < 8 || target % 2 !== 0) {
    throw new RangeError('The reference frontier must be an even piece count of at least 8.');
  }
  const boundaries = [8];
  for (let boundary = 10; boundary <= target; boundary += 2) boundaries.push(boundary);
  await rm(output, { recursive: true, force: true });
  await mkdir(output, { recursive: true });
  const roles = {};
  for (const roleName of ['red', 'yellow']) {
    roles[roleName] = await generateRole(binary, output, roleName, boundaries, maximumPasses, seedDirectory);
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
  const generated = join(temporary, 'small-reference');
  const manifest = await generateReference(binary, generated, 8, 20);
  if (manifest.roles.red.replay.segments.at(-1).frontierStates !== 1477
      || manifest.roles.yellow.replay.segments.at(-1).frontierStates !== 4515) {
    throw new Error('Independent replay did not reproduce the eight-piece reference frontiers.');
  }
  return { native: native.records, replay: manifest.roles };
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
    if (options.command === 'generate') {
      const target = integerOption(options.frontier_pieces, 12, 'frontier-pieces', 8, 42);
      const output = resolve(options.output ?? join(ROOT, 'generated', `perfect-chaos-prefix-${target}`));
      const passes = integerOption(options.maximum_passes, 200, 'maximum-passes', 1, 10_000);
      const seedDirectory = options.seed_rejections ? resolve(options.seed_rejections) : null;
      const manifest = await generateReference(binary, output, target, passes, seedDirectory);
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
