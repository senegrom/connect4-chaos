#!/usr/bin/env node

import { createWriteStream } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { once } from 'node:events';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { solveChaosProofPosition } from '../src/chaos-proof.js';
import { CHAOS_DRAW, CHAOS_LOSS, CHAOS_WIN } from '../src/chaos-solver.js';
import { EMPTY, RED, YELLOW, createBoard } from '../src/engine.js';

const FRONTIER_MAGIC = Buffer.from('C4CFRN1\0', 'binary');
const FRONTIER_RECORD_SIZE = 19;
const ROLE_NAMES = Object.freeze({ 1: 'red', 2: 'yellow' });

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

function validateMaskState(state, boundary, label) {
  if (state.rows < 1 || state.columns < 1 || state.rows * state.columns > 42) {
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
  for (let index = 0, offset = 16; index < count; index += 1, offset += FRONTIER_RECORD_SIZE) {
    const state = {
      mover: buffer.readBigUInt64LE(offset),
      opponent: buffer.readBigUInt64LE(offset + 8),
      rows: buffer[offset + 16],
      columns: buffer[offset + 17],
      aiTurn: buffer[offset + 18] !== 0,
    };
    validateMaskState(state, boundary, `${label} record ${index}`);
    states.push(state);
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

async function writeLine(stream, value) {
  if (!stream.write(`${JSON.stringify(value)}\n`)) await once(stream, 'drain');
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
  const stream = outputPath ? createWriteStream(outputPath, { encoding: 'utf8' }) : process.stdout;

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

  if (rawOptions.quiet !== true) process.stderr.write(`${JSON.stringify(summary)}\n`);
  return summary;
}

async function main() {
  await runPerfectChaosBridge(parseArguments(process.argv.slice(2)));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
