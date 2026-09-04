// Plays the shipped neural player against the site's Brutal opponent.
//
// Brutal is the strongest non-exact opponent the game already offers, so
// this says whether the network is worth choosing over it. Colours
// alternate and the openings are sampled, since both players are otherwise
// deterministic and every game of a colour would be identical.

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  ACTION_DROP, RED, YELLOW, applyAction, createBoard, legalActions,
  otherPlayer, resolveActionOutcome,
} from '../src/engine.js';
import { CANVAS, PLANES, planeBuffer, writePlanes }
  from '../src/neural-planes.js';
import { bestAction, searchPosition }
  from '../src/neural-search.js';
import { chooseMove } from '../src/ai.js';

const here = dirname(fileURLToPath(import.meta.url));
const REPO = join(here, '..');
const SIMULATIONS = Number(process.argv[2] ?? 64);
const GAMES = Number(process.argv[3] ?? 12);
if (!Number.isInteger(SIMULATIONS) || SIMULATIONS <= 0 || !Number.isInteger(GAMES) || GAMES <= 0) {
  throw new Error('usage: node scripts/neural-vs-brutal.mjs <simulations> <games per board> [model.onnx]');
}
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
const OPENING_PLIES = 4;
const MAX_PLIES = 300;

const ort = await import('onnxruntime-node');
// A third argument names a different model, so a candidate can be tried
// before it replaces the one on the site.
const MODEL = process.argv[4] ?? join(REPO, 'assets', 'neural', 'model.onnx');
const session = await ort.InferenceSession.create(await readFile(MODEL));
const input = planeBuffer(1);

async function evaluate(board, mover, _actions, connect, chaosMode) {
  const rows = board.length;
  const cols = board[0].length;
  writePlanes(input, 0, rows, cols, connect, chaosMode, (row, column) => {
    const cell = board[rows - 1 - row][column];
    if (cell === 0) return 0;
    return cell === mover ? 1 : 2;
  });
  const outputs = await session.run(
    { planes: new ort.Tensor('float32', input, [1, PLANES, CANVAS, CANVAS]) });
  return { policy: outputs.policy.data, value: outputs.value.data, q: outputs.q.data };
}

function randomAction(board, chaosMode) {
  const actions = legalActions(board, chaosMode);
  return actions[Math.floor(Math.random() * actions.length)];
}

async function playGame(board0, connect, chaosMode, neuralPlays) {
  let board = board0;
  let mover = RED;
  const seen = new Map();
  for (let ply = 0; ply < MAX_PLIES; ply += 1) {
    const actions = legalActions(board, chaosMode);
    if (actions.length === 0) return 0;
    let action;
    if (ply < OPENING_PLIES) {
      action = randomAction(board, chaosMode);
    } else if (mover === neuralPlays) {
      const result = await searchPosition({ board, currentPlayer: mover, connect, chaosMode },
        evaluate, { simulations: SIMULATIONS });
      action = bestAction(result);
    } else {
      const move = chooseMove(
        { board, currentPlayer: mover, connect, chaosMode, startingPlayer: RED },
        { difficulty: 'brutal', aiPlayer: mover },
      );
      action = move?.action ?? randomAction(board, chaosMode);
    }
    const applied = applyAction(board, action, mover);
    if (!applied) return 0;
    const lastDrop = action.type === ACTION_DROP
      ? { row: applied.row, column: applied.column } : null;
    const outcome = resolveActionOutcome(applied.board, connect, mover, action.type, lastDrop);
    board = applied.board;
    if (outcome.status === 'won') return outcome.winner === neuralPlays ? 1 : -1;
    if (outcome.status === 'draw') return 0;
    const key = `${mover}:${board.map((row) => row.join('')).join('/')}`;
    const count = (seen.get(key) ?? 0) + 1;
    if (count >= 3) return 0;                    // threefold repetition
    seen.set(key, count);
    mover = otherPlayer(mover);
  }
  return 0;
}

console.log(`neural (${SIMULATIONS} simulations, ${MODEL.split(/[\\/]/).pop()}) vs brutal, ${GAMES} games per board\n`);
let totals = [0, 0, 0];
for (const { rows, cols, connect, chaosMode } of BOARDS) {
  const tally = [0, 0, 0];
  for (let game = 0; game < GAMES; game += 1) {
    const neuralPlays = game % 2 === 0 ? RED : YELLOW;
    // eslint-disable-next-line no-await-in-loop
    const started = Date.now();
    const result = await playGame(createBoard(rows, cols), connect, chaosMode, neuralPlays);
    tally[result === 1 ? 0 : (result === 0 ? 1 : 2)] += 1;
    console.log(`    ${rows}x${cols} ${chaosMode ? 'chaos' : 'classic'} game ${game + 1}/${GAMES}: `
      + `${result === 1 ? 'neural won' : (result === 0 ? 'draw' : 'brutal won')} `
      + `(${((Date.now() - started) / 1000).toFixed(0)}s)`);
  }
  totals = totals.map((value, index) => value + tally[index]);
  const score = (tally[0] + 0.5 * tally[1]) / GAMES;
  console.log(`  ${rows}x${cols} c${connect} ${chaosMode ? 'chaos' : 'classic'}: `
    + `${(score * 100).toFixed(1)}%  (${tally[0]}W/${tally[1]}D/${tally[2]}L)`);
}
const played = totals[0] + totals[1] + totals[2];
console.log(`\n  overall: ${(((totals[0] + 0.5 * totals[1]) / played) * 100).toFixed(1)}% `
  + `for neural (${totals[0]}W/${totals[1]}D/${totals[2]}L)`);
