#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { createWriteStream } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { once } from 'node:events';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { solveChaosProofPosition } from '../src/chaos-proof.js';
import { CHAOS_DRAW, CHAOS_LOSS, CHAOS_WIN } from '../src/chaos-solver.js';
import { EMPTY, RED, YELLOW, createBoard } from '../src/engine.js';

const FRONTIER_MAGIC = Buffer.from('C4CFRN1\0', 'binary');
const FRONTIER_RECORD_SIZE = 19;
const ROLE_NAMES = Object.freeze({ 1: 'red', 2: 'yellow' });

function parseArguments(argv) {
  const options = { command: 'scan' };
  let index = 0;
  if (argv[0] && !argv[0].startsWith('--')) {
    options.command = argv[0];
    index = 1;
  }

  for (; index < argv.length; index += 1) {
    const argument = argv[index];
    if (!argument.startsWith('--')) throw new RangeError(`Unexpected argument: ${argument}`);
    const name = argument.slice(2).replaceAll('-', '_');
    const value = argv[index + 1];
    if (name === 'input') {
      if (value === undefined || value.startsWith('--')) {
        throw new RangeError('--input requires a frontier path.');
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

function bitIndex(state, column, rowFromBottom) {
  return BigInt(column * (state.rows + 1) + rowFromBottom);
}

function bit(state, column, rowFromBottom) {
  return 1n << bitIndex(state, column, rowFromBottom);
}

function popcount(value) {
  let remaining = value;
  let count = 0;
  while (remaining !== 0n) {
    remaining &= remaining - 1n;
    count += 1;
  }
  return count;
}

function compareMaskStates(first, second) {
  if (first.rows !== second.rows) return first.rows - second.rows;
  if (first.columns !== second.columns) return first.columns - second.columns;
  if (first.aiTurn !== second.aiTurn) return Number(first.aiTurn) - Number(second.aiTurn);
  if (first.mover !== second.mover) return first.mover < second.mover ? -1 : 1;
  if (first.opponent !== second.opponent) return first.opponent < second.opponent ? -1 : 1;
  return 0;
}

function maskStateKey(state) {
  return `${state.rows}:${state.columns}:${state.aiTurn ? 1 : 0}:`
    + `${state.mover.toString(16)}:${state.opponent.toString(16)}`;
}

function validateMaskState(state, boundary, label) {
  if (!state || typeof state.mover !== 'bigint' || typeof state.opponent !== 'bigint') {
    throw new Error(`${label} must contain bigint mover and opponent masks.`);
  }
  if (typeof state.aiTurn !== 'boolean') throw new Error(`${label} must declare aiTurn.`);
  if (!Number.isInteger(state.rows) || !Number.isInteger(state.columns)
      || state.rows < 1 || state.columns < 1 || state.rows * state.columns > 42) {
    throw new Error(`${label} has unsupported board dimensions.`);
  }
  if ((state.mover & state.opponent) !== 0n) {
    throw new Error(`${label} has overlapping mover and opponent masks.`);
  }

  let validMask = 0n;
  for (let column = 0; column < state.columns; column += 1) {
    let foundEmpty = false;
    for (let row = 0; row < state.rows; row += 1) {
      const cell = bit(state, column, row);
      validMask |= cell;
      const occupied = ((state.mover | state.opponent) & cell) !== 0n;
      if (!occupied) foundEmpty = true;
      else if (foundEmpty) throw new Error(`${label} violates gravity.`);
    }
  }
  if (((state.mover | state.opponent) & ~validMask) !== 0n) {
    throw new Error(`${label} uses sentinel or out-of-board bits.`);
  }
  if (popcount(state.mover | state.opponent) !== boundary) {
    throw new Error(`${label} does not contain the frontier piece count.`);
  }
}

function orderedUniqueStates(states, boundary, label) {
  if (!Array.isArray(states)) throw new TypeError(`${label} states must be an array.`);
  const unique = new Map();
  states.forEach((state, index) => {
    validateMaskState(state, boundary, `${label} record ${index}`);
    unique.set(maskStateKey(state), { ...state });
  });
  return [...unique.values()].sort(compareMaskStates);
}

export function encodeChaosFrontier(role, boundary, states, label = 'Perfect Chaos frontier') {
  if (!ROLE_NAMES[role]) throw new RangeError(`${label} role must be Red or Yellow.`);
  if (!Number.isInteger(boundary) || boundary < 0 || boundary > 42) {
    throw new RangeError(`${label} boundary must be an integer from 0 through 42.`);
  }
  const ordered = orderedUniqueStates(states, boundary, label);
  const buffer = Buffer.alloc(16 + ordered.length * FRONTIER_RECORD_SIZE);
  FRONTIER_MAGIC.copy(buffer, 0);
  buffer[8] = 1;
  buffer[9] = role;
  buffer[10] = boundary;
  buffer[11] = FRONTIER_RECORD_SIZE;
  buffer.writeUInt32LE(ordered.length, 12);
  for (let index = 0, offset = 16; index < ordered.length;
    index += 1, offset += FRONTIER_RECORD_SIZE) {
    const state = ordered[index];
    buffer.writeBigUInt64LE(state.mover, offset);
    buffer.writeBigUInt64LE(state.opponent, offset + 8);
    buffer[offset + 16] = state.rows;
    buffer[offset + 17] = state.columns;
    buffer[offset + 18] = state.aiTurn ? 1 : 0;
  }
  return buffer;
}

export function decodeChaosFrontier(buffer, label = 'Perfect Chaos frontier') {
  if (!Buffer.isBuffer(buffer) || buffer.length < 16 || !buffer.subarray(0, 8).equals(FRONTIER_MAGIC)) {
    throw new Error(`${label} has an invalid magic header.`);
  }
  const version = buffer[8];
  const role = buffer[9];
  const boundary = buffer[10];
  const recordSize = buffer[11];
  const count = buffer.readUInt32LE(12);
  if (version !== 1 || !ROLE_NAMES[role] || boundary > 42 || recordSize !== FRONTIER_RECORD_SIZE) {
    throw new Error(`${label} has an unsupported header.`);
  }
  if (buffer.length !== 16 + count * FRONTIER_RECORD_SIZE) {
    throw new Error(`${label} length does not match its record count.`);
  }

  const states = [];
  let previous = null;
  for (let index = 0, offset = 16; index < count; index += 1, offset += FRONTIER_RECORD_SIZE) {
    const state = {
      mover: buffer.readBigUInt64LE(offset),
      opponent: buffer.readBigUInt64LE(offset + 8),
      rows: buffer[offset + 16],
      columns: buffer[offset + 17],
      aiTurn: buffer[offset + 18] !== 0,
    };
    validateMaskState(state, boundary, `${label} record ${index}`);
    if (previous && compareMaskStates(previous, state) >= 0) {
      throw new Error(`${label} states must be strictly sorted without duplicates.`);
    }
    states.push(state);
    previous = state;
  }
  return { version, role, roleName: ROLE_NAMES[role], boundary, states };
}

export function chaosFrontierStateToBoard(state) {
  const board = createBoard(state.rows, state.columns);
  for (let column = 0; column < state.columns; column += 1) {
    for (let rowFromBottom = 0; rowFromBottom < state.rows; rowFromBottom += 1) {
      const cell = bit(state, column, rowFromBottom);
      const row = state.rows - 1 - rowFromBottom;
      if ((state.mover & cell) !== 0n) board[row][column] = RED;
      else if ((state.opponent & cell) !== 0n) board[row][column] = YELLOW;
      else board[row][column] = EMPTY;
    }
  }
  return board;
}

function aiBounds(aiTurn, lower, upper) {
  return aiTurn
    ? { lower, upper }
    : { lower: -upper, upper: -lower };
}

function serializeActionBounds(aiTurn, entries) {
  return entries.map((entry) => {
    const bounds = aiBounds(aiTurn, entry.lower, entry.upper);
    return {
      action: entry.action,
      moverLower: entry.lower,
      moverUpper: entry.upper,
      aiLower: bounds.lower,
      aiUpper: bounds.upper,
      frontier: entry.frontier,
    };
  });
}

function stateIdentity(state) {
  return {
    rows: state.rows,
    columns: state.columns,
    aiTurn: state.aiTurn,
    mover: state.mover.toString(16),
    opponent: state.opponent.toString(16),
  };
}

function artifactMetadata(path, buffer, records) {
  return {
    path,
    records,
    bytes: buffer.length,
    sha256: createHash('sha256').update(buffer).digest('hex'),
  };
}

async function ensureParent(path) {
  await mkdir(dirname(path), { recursive: true });
}

async function writeLine(stream, value) {
  if (!stream.write(`${JSON.stringify(value)}\n`)) await once(stream, 'drain');
}

export async function mergeChaosRejectionFiles(inputPaths, output) {
  if (!Array.isArray(inputPaths) || inputPaths.length === 0) {
    throw new RangeError('At least one --input rejection frontier is required.');
  }
  if (!output || output === true) throw new RangeError('--output is required.');

  let role = null;
  let boundary = null;
  const states = [];
  const inputs = [];
  for (const rawPath of inputPaths) {
    const path = resolve(String(rawPath));
    const frontier = decodeChaosFrontier(await readFile(path), path);
    role ??= frontier.role;
    boundary ??= frontier.boundary;
    if (frontier.role !== role || frontier.boundary !== boundary) {
      throw new Error('Rejection frontiers must have matching roles and boundaries.');
    }
    inputs.push(path);
    states.push(...frontier.states);
  }

  const outputPath = resolve(String(output));
  const buffer = encodeChaosFrontier(role, boundary, states, outputPath);
  await ensureParent(outputPath);
  await writeFile(outputPath, buffer);
  const merged = decodeChaosFrontier(buffer, outputPath);
  return {
    format: 'connect4-chaos-bounded-rejection-merge-v1',
    role: merged.roleName,
    boundary,
    inputs,
    artifact: artifactMetadata(outputPath, buffer, merged.states.length),
  };
}

export async function runPerfectChaosBridge(rawOptions = {}) {
  if (!rawOptions.frontier || rawOptions.frontier === true) {
    throw new RangeError('--frontier is required.');
  }
  const frontierPath = resolve(String(rawOptions.frontier));
  const frontier = decodeChaosFrontier(await readFile(frontierPath), frontierPath);
  const dropDepth = integerOption(rawOptions.drop_depth, 2, 'drop-depth', 1, 42);
  const maximumStates = integerOption(
    rawOptions.maximum_states,
    150_000,
    'maximum-states',
    1,
    2_000_000,
  );
  const connect = integerOption(rawOptions.connect, 4, 'connect', 1, 10);
  const shardCount = integerOption(rawOptions.shard_count, 1, 'shard-count', 1, 65_536);
  const shardIndex = integerOption(rawOptions.shard_index, 0, 'shard-index', 0, shardCount - 1);
  const start = integerOption(rawOptions.start, 0, 'start', 0, frontier.states.length);
  const limit = integerOption(
    rawOptions.limit,
    frontier.states.length,
    'limit',
    0,
    frontier.states.length,
  );
  const outputPath = rawOptions.output && rawOptions.output !== true
    ? resolve(String(rawOptions.output))
    : null;
  const rejectionPath = rawOptions.rejections && rawOptions.rejections !== true
    ? resolve(String(rawOptions.rejections))
    : null;
  if (outputPath && rejectionPath && outputPath === rejectionPath) {
    throw new RangeError('--output and --rejections must use different paths.');
  }
  if (outputPath) await ensureParent(outputPath);
  const stream = outputPath ? createWriteStream(outputPath, { encoding: 'utf8' }) : process.stdout;
  const rejectedStates = [];

  const summary = {
    format: 'connect4-chaos-bounded-bridge-summary-v1',
    frontier: frontierPath,
    role: frontier.roleName,
    boundary: frontier.boundary,
    dropDepth,
    maximumStates,
    shardIndex,
    shardCount,
    selected: 0,
    solved: 0,
    unresolved: 0,
    limits: 0,
    aiWins: 0,
    aiDraws: 0,
    aiLosses: 0,
    rejections: 0,
  };

  try {
    for (let index = start; index < frontier.states.length && summary.selected < limit; index += 1) {
      if (index % shardCount !== shardIndex) continue;
      const state = frontier.states[index];
      summary.selected += 1;
      const identity = stateIdentity(state);
      try {
        const proof = solveChaosProofPosition({
          board: chaosFrontierStateToBoard(state),
          currentPlayer: RED,
          connect,
          chaosMode: true,
        }, { dropDepth, maximumStates });
        const bounds = aiBounds(state.aiTurn, proof.lowerValue, proof.upperValue);
        const aiValue = proof.solved ? bounds.lower : null;
        if (proof.solved) {
          summary.solved += 1;
          if (aiValue === CHAOS_WIN) summary.aiWins += 1;
          else if (aiValue === CHAOS_DRAW) summary.aiDraws += 1;
          else if (aiValue === CHAOS_LOSS) summary.aiLosses += 1;
        } else summary.unresolved += 1;
        if (bounds.upper === CHAOS_LOSS) {
          rejectedStates.push(state);
          summary.rejections += 1;
        }

        await writeLine(stream, {
          format: 'connect4-chaos-bounded-bridge-record-v1',
          index,
          role: frontier.roleName,
          boundary: frontier.boundary,
          state: identity,
          status: proof.solved ? 'solved' : 'bounded',
          moverLower: proof.lowerValue,
          moverUpper: proof.upperValue,
          aiLower: bounds.lower,
          aiUpper: bounds.upper,
          aiValue,
          rejected: bounds.upper === CHAOS_LOSS,
          action: proof.action,
          actionBounds: serializeActionBounds(state.aiTurn, proof.actionBounds),
          states: proof.nodes,
          elapsedMs: proof.elapsedMs,
          graph: proof.graph,
        });
      } catch (error) {
        if (error?.code !== 'CHAOS_PROOF_GRAPH_LIMIT') throw error;
        summary.limits += 1;
        await writeLine(stream, {
          format: 'connect4-chaos-bounded-bridge-record-v1',
          index,
          role: frontier.roleName,
          boundary: frontier.boundary,
          state: identity,
          status: 'state-limit',
          rejected: false,
          dropDepth,
          maximumStates,
          states: error.states,
        });
      }
    }
  } finally {
    if (outputPath) {
      stream.end();
      await once(stream, 'finish');
    }
  }

  if (rejectionPath) {
    const buffer = encodeChaosFrontier(
      frontier.role,
      frontier.boundary,
      rejectedStates,
      rejectionPath,
    );
    await ensureParent(rejectionPath);
    await writeFile(rejectionPath, buffer);
    summary.rejectionArtifact = artifactMetadata(
      rejectionPath,
      buffer,
      decodeChaosFrontier(buffer, rejectionPath).states.length,
    );
  }

  if (rawOptions.quiet !== true) process.stderr.write(`${JSON.stringify(summary)}\n`);
  return summary;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.command === 'scan') {
    await runPerfectChaosBridge(options);
    return;
  }
  if (options.command === 'merge-rejections') {
    const summary = await mergeChaosRejectionFiles(options.inputs, options.output);
    if (options.quiet !== true) process.stderr.write(`${JSON.stringify(summary)}\n`);
    return;
  }
  throw new RangeError(`Unknown command: ${options.command}`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
