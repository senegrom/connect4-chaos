#!/usr/bin/env node

import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

import {
  CHAOS_DRAW,
  CHAOS_LOSS,
  CHAOS_WIN,
  buildChaosGraph,
  solveChaosGraph,
} from '../src/chaos-solver.js';
import { EMPTY, RED } from '../src/engine.js';

const FORMAT = 'connect4-chaos-closed-wdl-graph-v1';
const OBJECTIVE = 'maximize-win-then-draw-then-loss';
const REPOSITORY = resolve(import.meta.dirname, '..');
const PYTHON_SOLVER = join(REPOSITORY, 'scripts', 'perfect-chaos-wdl.py');

function valueName(value) {
  if (value === CHAOS_WIN) return 'win';
  if (value === CHAOS_DRAW) return 'draw';
  if (value === CHAOS_LOSS) return 'loss';
  throw new Error(`Unexpected Chaos value ${value}.`);
}

function terminalForAi(terminal, aiTurn) {
  if (terminal === CHAOS_DRAW) return 'draw';
  if (terminal === CHAOS_WIN) return aiTurn ? 'win' : 'loss';
  if (terminal === CHAOS_LOSS) return aiTurn ? 'loss' : 'win';
  throw new Error(`Unexpected terminal ${terminal}.`);
}

function fixedRoleGraph(base, aiStarts) {
  const pairs = [];
  const nodes = [];
  const indices = new Map();
  const queue = [];

  function add(baseNode, aiTurn) {
    const key = `${baseNode}:${aiTurn ? 1 : 0}`;
    let index = indices.get(key);
    if (index !== undefined) return index;
    index = pairs.length;
    indices.set(key, index);
    pairs.push({ baseNode, aiTurn });
    nodes.push(null);
    queue.push(index);
    return index;
  }

  const root = add(base.root, aiStarts);
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const index = queue[cursor];
    const pair = pairs[index];
    const baseNode = base.nodes[pair.baseNode];
    nodes[index] = {
      aiTurn: pair.aiTurn,
      edges: baseNode.edges.map((edge) => {
        if (edge.terminal !== null) {
          return {
            terminal: terminalForAi(edge.terminal, pair.aiTurn),
            action: edge.action,
          };
        }
        return {
          next: add(edge.next, !pair.aiTurn),
          action: edge.action,
        };
      }),
    };
  }

  return {
    document: {
      format: FORMAT,
      objective: OBJECTIVE,
      role: aiStarts ? 'red' : 'yellow',
      roots: [root],
      nodes,
    },
    pairs,
  };
}

async function solveWithPython(directory, label, document) {
  const input = join(directory, `${label}.graph.json`);
  const output = join(directory, `${label}.solution.json`);
  await writeFile(input, `${JSON.stringify(document, null, 2)}\n`);
  const result = spawnSync(
    'python3',
    [PYTHON_SOLVER, 'solve', '--input', input, '--output', output],
    { cwd: REPOSITORY, encoding: 'utf8' },
  );
  if (result.status !== 0) {
    throw new Error(`Python W/D/L solver failed for ${label}.\n${result.stderr}`);
  }
  return JSON.parse(result.stdout);
}

async function verifyCase(directory, name, board, connect, expectedBaseStates) {
  const base = buildChaosGraph(
    { board, currentPlayer: RED, connect },
    { maximumStates: 100_000 },
  );
  if (base.nodes.length !== expectedBaseStates) {
    throw new Error(`${name}: base graph has ${base.nodes.length}, expected ${expectedBaseStates}.`);
  }
  const solvedBase = solveChaosGraph(base);
  const summaries = [];

  for (const aiStarts of [true, false]) {
    const fixed = fixedRoleGraph(base, aiStarts);
    const label = `${name}-${aiStarts ? 'ai-first' : 'ai-second'}`;
    const solved = await solveWithPython(directory, label, fixed.document);
    if (solved.values.length !== fixed.pairs.length) {
      throw new Error(`${label}: fixed-role solution length mismatch.`);
    }

    for (let index = 0; index < fixed.pairs.length; index += 1) {
      const { baseNode, aiTurn } = fixed.pairs[index];
      const relative = solvedBase.values[baseNode];
      const expected = valueName(aiTurn ? relative : -relative);
      if (solved.values[index] !== expected) {
        throw new Error(
          `${label}: node ${index} value ${solved.values[index]} does not match `
          + `the independent relative solver value ${expected}.`,
        );
      }
    }

    const relativeRoot = solvedBase.values[base.root];
    const expectedRoot = valueName(aiStarts ? relativeRoot : -relativeRoot);
    if (solved.rootValues[0] !== expectedRoot) {
      throw new Error(`${label}: root value mismatch.`);
    }
    if (!solved.allChosenActionsOptimal
        || !solved.rankedWinningProgressVerified
        || !solved.drawRegionClosedVerified) {
      throw new Error(`${label}: exact policy verification flags are incomplete.`);
    }
    summaries.push({
      role: fixed.document.role,
      baseStates: base.nodes.length,
      fixedRoleStates: fixed.pairs.length,
      rootValue: solved.rootValues[0],
      values: solved.counts,
      policyEntries: solved.policy.length,
    });
  }
  return { name, summaries };
}

async function main() {
  const temporary = await mkdtemp(join(tmpdir(), 'connect4-chaos-wdl-crosscheck-'));
  try {
    const cases = [];
    cases.push(await verifyCase(
      temporary,
      '2x2-connect2',
      Array.from({ length: 2 }, () => Array(2).fill(EMPTY)),
      2,
      6,
    ));
    cases.push(await verifyCase(
      temporary,
      '3x3-connect3',
      Array.from({ length: 3 }, () => Array(3).fill(EMPTY)),
      3,
      628,
    ));
    process.stdout.write(`${JSON.stringify({
      format: 'connect4-chaos-wdl-crosscheck-v1',
      objective: OBJECTIVE,
      cases,
    }, null, 2)}\n`);
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

await main();
