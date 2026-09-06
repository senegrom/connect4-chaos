// Headless Neural versus the app's prepared Brutal opponent. Random openings
// do not follow the certified Chaos policy, so use its general mid-game route
// in that case. Classic books and bounded Chaos proofs are still prepared by
// the same function as the browser worker. This is not a proof of playing strength.

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import {
  ACTION_DROP, RED, YELLOW, applyAction, createBoard, legalActions,
  otherPlayer, positionKey, resolveActionOutcome, sameAction,
} from '../src/engine.js';
import { CANVAS, PLANES, planeBuffer, writePlanes } from '../src/neural-planes.js';
import { bestAction, searchPosition } from '../src/neural-search.js';
import { choosePreparedMove } from '../src/ai-worker.js';

const REPO = join(dirname(fileURLToPath(import.meta.url)), '..');
const BOARDS = [
  { rows: 6, cols: 7, connect: 4, chaosMode: false },
  { rows: 6, cols: 7, connect: 4, chaosMode: true },
  { rows: 7, cols: 7, connect: 4, chaosMode: true },
  { rows: 8, cols: 8, connect: 5, chaosMode: true },
  { rows: 9, cols: 7, connect: 4, chaosMode: false },
  { rows: 10, cols: 10, connect: 4, chaosMode: false },
  { rows: 5, cols: 10, connect: 4, chaosMode: true },
  { rows: 10, cols: 9, connect: 5, chaosMode: true },
];

/** A serial evaluator with the same repetition planes as the browser runtime. */
export function createEvaluator(ort, session) {
  const input = planeBuffer(1);
  return async (board, mover, _actions, connect, chaosMode, repeated = 0) => {
    const rows = board.length;
    const cols = board[0].length;
    writePlanes(input, 0, rows, cols, connect, chaosMode, (row, column) => {
      const cell = board[rows - 1 - row][column];
      return cell === 0 ? 0 : (cell === mover ? 1 : 2);
    }, repeated >= 1, repeated >= 2);
    const outputs = await session.run({
      planes: new ort.Tensor('float32', input, [1, PLANES, CANVAS, CANVAS]),
    });
    return { policy: outputs.policy.data, value: outputs.value.data, q: outputs.q.data };
  };
}

function randomAction(board, chaosMode) {
  const actions = legalActions(board, chaosMode);
  return actions[Math.floor(Math.random() * actions.length)];
}

/** Play one match. Injected opponents/openings keep rule regressions model-free.
 * Returns a neural-relative outcome; null means the safety ply cap, NOT a draw.
 */
export async function playGame(board0, connect, chaosMode, neuralPlays, {
  evaluate, simulations = 64, openingPlies = 4, maxPlies = 300,
  neuralSearch = searchPosition, brutalMove = choosePreparedMove, chooseOpening = randomAction,
} = {}) {
  if (![RED, YELLOW].includes(neuralPlays)
      || !Number.isSafeInteger(simulations) || simulations < 1
      || !Number.isSafeInteger(openingPlies) || openingPlies < 0
      || !Number.isSafeInteger(maxPlies) || maxPlies < 1) {
    throw new RangeError('Invalid benchmark player, simulations or ply limits.');
  }
  let board = board0;
  let mover = RED;
  // Match the app: the initial position is occurrence one, and keys identify
  // the NEXT player to move, with dimensions and rules included.
  const seen = new Map([[positionKey(board, mover, connect, chaosMode), 1]]);
  for (let ply = 0; ply < maxPlies; ply += 1) {
    const actions = legalActions(board, chaosMode);
    if (actions.length === 0) return 0;
    const position = {
      board, currentPlayer: mover, connect, chaosMode, startingPlayer: RED,
      repetitionCounts: [...seen.entries()],
    };
    let action;
    if (ply < openingPlies) {
      action = await chooseOpening(board, chaosMode);
    } else if (mover === neuralPlays) {
      action = bestAction(await neuralSearch(position, evaluate, { simulations }));
    } else {
      const move = await brutalMove(position, {
        difficulty: 'brutal', aiPlayer: mover, useChaosPolicy: openingPlies === 0,
      });
      action = move?.action;
    }
    // Never disguise a broken comparator as a random move or a drawn game.
    if (!actions.some((candidate) => sameAction(candidate, action))) {
      throw new Error(`Benchmark opponent returned an illegal move at ply ${ply + 1}.`);
    }
    const applied = applyAction(board, action, mover);
    if (!applied) throw new Error(`Benchmark could not apply move at ply ${ply + 1}.`);
    const lastDrop = action.type === ACTION_DROP
      ? { row: applied.row, column: applied.column } : null;
    const outcome = resolveActionOutcome(applied.board, connect, mover, action.type, lastDrop);
    board = applied.board;
    if (outcome.status === 'won') return outcome.winner === neuralPlays ? 1 : -1;
    if (outcome.status === 'draw') return 0;
    mover = otherPlayer(mover);
    const key = positionKey(board, mover, connect, chaosMode);
    const count = (seen.get(key) ?? 0) + 1;
    seen.set(key, count);
    if (count >= 3) return 0;
  }
  return null;
}

function tallyText(tally) {
  const completed = tally[0] + tally[1] + tally[2];
  const score = completed ? `${((tally[0] + 0.5 * tally[1]) * 100 / completed).toFixed(1)}%` : 'n/a';
  return `${score} on completed games (${tally[0]}W/${tally[1]}D/${tally[2]}L, ${tally[3]} capped)`;
}

async function main() {
  const simulations = Number(process.argv[2] ?? 64);
  const games = Number(process.argv[3] ?? 12);
  if (!Number.isSafeInteger(simulations) || simulations <= 0 || !Number.isSafeInteger(games) || games <= 0) {
    throw new Error('usage: node scripts/neural-vs-brutal.mjs <simulations> <games per board> [model.onnx]');
  }
  // Keep importing this module safe for tests; load the native runtime only for CLI runs.
  const ort = await import('onnxruntime-node');
  const model = process.argv[4] ?? join(REPO, 'assets', 'neural', 'model.onnx');
  const session = await ort.InferenceSession.create(await readFile(model));
  try {
    const evaluate = createEvaluator(ort, session);
    console.log(`Neural (${simulations} simulations, ${model.split(/[\\/]/).pop()}) vs prepared Brutal, ${games} games per board`);
    console.log('4 randomized opening plies; general Chaos handoff (not certified opening policy), Classic book and bounded proofs enabled.\n');
    let totals = [0, 0, 0, 0];
    for (const { rows, cols, connect, chaosMode } of BOARDS) {
      const tally = [0, 0, 0, 0];
      for (let game = 0; game < games; game += 1) {
        const neuralPlays = game % 2 === 0 ? RED : YELLOW;
        const started = Date.now();
        const result = await playGame(createBoard(rows, cols), connect, chaosMode, neuralPlays, { evaluate, simulations });
        tally[result === null ? 3 : (result === 1 ? 0 : (result === 0 ? 1 : 2))] += 1;
        console.log(`    ${rows}x${cols} ${chaosMode ? 'chaos' : 'classic'} game ${game + 1}/${games}: `
          + `${result === null ? 'ply cap (unresolved)' : (result === 1 ? 'neural won' : (result === 0 ? 'draw' : 'brutal won'))} `
          + `(${((Date.now() - started) / 1000).toFixed(0)}s)`);
      }
      totals = totals.map((value, index) => value + tally[index]);
      console.log(`  ${rows}x${cols} c${connect} ${chaosMode ? 'chaos' : 'classic'}: ${tallyText(tally)}`);
    }
    console.log(`\n  overall for neural: ${tallyText(totals)}`);
  } finally {
    await session.release();
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
