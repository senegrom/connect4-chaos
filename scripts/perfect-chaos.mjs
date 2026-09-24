#!/usr/bin/env node

import process from 'node:process';

import { CHAOS_DRAW, CHAOS_WIN, solveChaosPosition } from '../src/chaos-solver.js';
import { ACTION_ROTATE_CW, RED, createBoard } from '../src/engine.js';

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

const command = process.argv[2] ?? 'verify';
if (command !== 'verify') throw new RangeError(`Unknown command: ${command}`);
process.stdout.write(`${JSON.stringify(verify(), null, 2)}\n`);
