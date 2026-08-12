#!/usr/bin/env node

import { readFile, writeFile } from 'node:fs/promises';
import process from 'node:process';

import {
  CHAOS_DRAW,
  CHAOS_WIN,
  canonicalChaosPosition,
  solveChaosPosition,
} from '../src/chaos-solver.js';
import {
  ACTION_DROP,
  ACTION_ROTATE_CW,
  RED,
  YELLOW,
  applyAction,
  createBoard,
  legalActions,
  resolveActionOutcome,
} from '../src/engine.js';

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

function integerOption(value, fallback, label, minimum = 0) {
  const selected = value === undefined ? fallback : Number.parseInt(String(value), 10);
  if (!Number.isInteger(selected) || selected < minimum) {
    throw new RangeError(`${label} must be an integer of at least ${minimum}.`);
  }
  return selected;
}

function exact(position, maximumStates = 2_000_000) {
  return solveChaosPosition(position, { maximumStates });
}

function fixturePosition() {
  return {
    board: [
      [1, 1, 1, 2, 1, 0, 0],
      [2, 2, 2, 1, 2, 0, 0],
      [2, 1, 2, 1, 2, 1, 0],
      [2, 1, 1, 1, 2, 2, 0],
      [1, 2, 2, 2, 1, 2, 2],
      [1, 1, 2, 2, 1, 1, 1],
    ],
    currentPlayer: RED,
    connect: 4,
    chaosMode: true,
  };
}

function verify() {
  const cases = [
    {
      name: '2x2-connect2',
      position: { board: createBoard(2, 2), currentPlayer: RED, connect: 2, chaosMode: true },
      value: CHAOS_WIN,
      states: 6,
    },
    {
      name: '3x3-connect3',
      position: { board: createBoard(3, 3), currentPlayer: RED, connect: 3, chaosMode: true },
      value: CHAOS_DRAW,
      states: 628,
    },
    {
      name: '6x7-endgame-fixture',
      position: fixturePosition(),
      value: CHAOS_WIN,
      action: { type: ACTION_ROTATE_CW },
      states: 2_585,
    },
  ];

  const results = [];
  for (const sample of cases) {
    const result = exact(sample.position);
    if (result.value !== sample.value) {
      throw new Error(`${sample.name}: expected value ${sample.value}, received ${result.value}.`);
    }
    if (result.nodes !== sample.states) {
      throw new Error(`${sample.name}: expected ${sample.states} states, received ${result.nodes}.`);
    }
    if (sample.action && JSON.stringify(result.action) !== JSON.stringify(sample.action)) {
      throw new Error(
        `${sample.name}: expected ${JSON.stringify(sample.action)}, received ${JSON.stringify(result.action)}.`,
      );
    }
    results.push({
      name: sample.name,
      value: result.value,
      action: result.action,
      states: result.nodes,
      rank: result.graph.rank,
      elapsedMs: Math.round(result.elapsedMs * 100) / 100,
    });
  }

  return {
    format: 'connect4-chaos-perfect-chaos-verification-v1',
    theorem: 'ranked-attractor-with-unresolved-cycles-as-draws',
    cases: results,
  };
}

async function solveFile(options) {
  if (!options.input) throw new RangeError('--input is required.');
  const position = JSON.parse(await readFile(options.input, 'utf8'));
  return exact(position, integerOption(options.maximum_states, 2_000_000, 'maximum-states', 1));
}

function continuingChild(board, action, connect) {
  const result = applyAction(board, action, RED);
  if (!result) return null;
  const outcome = resolveActionOutcome(
    result.board,
    connect,
    RED,
    action.type,
    action.type === ACTION_DROP ? { row: result.row, column: result.column } : null,
  );
  if (outcome.status !== 'playing') return null;
  return canonicalChaosPosition(result.board, YELLOW);
}

async function enumerateRoot(options) {
  const depth = integerOption(options.depth, 8, 'depth');
  const maximumStates = integerOption(
    options.maximum_states,
    2_000_000,
    'maximum-states',
    1,
  );
  const connect = 4;
  const root = canonicalChaosPosition(createBoard(6, 7), RED);
  let frontier = [root];
  const seen = new Set([root.key]);
  const layers = [{ depth: 0, frontier: 1, cumulative: 1 }];

  for (let ply = 1; ply <= depth; ply += 1) {
    const next = new Map();
    for (const state of frontier) {
      for (const action of legalActions(state.board, true)) {
        const child = continuingChild(state.board, action, connect);
        if (!child || seen.has(child.key) || next.has(child.key)) continue;
        if (seen.size + next.size >= maximumStates) {
          throw new RangeError(
            `Root enumeration exceeded the ${maximumStates.toLocaleString()}-state limit.`,
          );
        }
        next.set(child.key, child);
      }
    }
    frontier = [...next.values()];
    for (const key of next.keys()) seen.add(key);
    layers.push({ depth: ply, frontier: frontier.length, cumulative: seen.size });
    if (frontier.length === 0) break;
  }

  frontier.sort((first, second) => first.key.localeCompare(second.key));
  if (options.output) {
    const records = frontier.map((state) => JSON.stringify({
      id: state.key,
      position: {
        board: state.board,
        currentPlayer: RED,
        connect,
        chaosMode: true,
      },
    }));
    await writeFile(options.output, `${records.join('\n')}${records.length ? '\n' : ''}`);
  }

  return {
    format: 'connect4-chaos-perfect-chaos-root-enumeration-v1',
    depth,
    states: seen.size,
    frontier: frontier.length,
    output: options.output ?? null,
    layers,
  };
}

async function solveFrontier(options) {
  if (!options.input) throw new RangeError('--input is required.');
  if (!options.output) throw new RangeError('--output is required.');
  const shardIndex = integerOption(options.shard_index, 0, 'shard-index');
  const shardCount = integerOption(options.shard_count, 1, 'shard-count', 1);
  if (shardIndex >= shardCount) throw new RangeError('shard-index must be smaller than shard-count.');
  const maximumStates = integerOption(options.maximum_states, 2_000_000, 'maximum-states', 1);
  const lines = (await readFile(options.input, 'utf8')).split(/\r?\n/).filter(Boolean);
  const output = [];

  for (let index = 0; index < lines.length; index += 1) {
    if (index % shardCount !== shardIndex) continue;
    const record = JSON.parse(lines[index]);
    const position = record.position ?? record;
    const result = exact(position, maximumStates);
    output.push(JSON.stringify({
      id: record.id ?? index,
      value: result.value,
      action: result.action,
      rank: result.graph.rank,
      states: result.nodes,
    }));
  }

  await writeFile(options.output, `${output.join('\n')}${output.length ? '\n' : ''}`);
  return {
    input: options.input,
    output: options.output,
    shardIndex,
    shardCount,
    solved: output.length,
  };
}

const options = parseArguments(process.argv.slice(2));
let result;
if (options.command === 'verify') result = verify();
else if (options.command === 'solve') result = await solveFile(options);
else if (options.command === 'frontier') result = await solveFrontier(options);
else if (options.command === 'enumerate') result = await enumerateRoot(options);
else throw new RangeError(`Unknown command: ${options.command}`);

process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
