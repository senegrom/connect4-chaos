#!/usr/bin/env node
// Positions in which exactly one move keeps a forced win, with the exact
// value of every legal move, for the neural strength test
// (tests/strength/neural-strength.mjs).
//
// The positions come from "sensible random" games played from the empty
// board: a player takes an immediate win when there is one, otherwise avoids
// the moves that hand the opponent one, and otherwise plays uniformly at
// random - in Chaos choosing a transform some of the time. Games like that
// wander into lopsided middlegames where one precise move decides the game,
// which is what a strength test needs.
//
// Every legal action of a candidate is applied with the engine. A move that
// ends the game is scored by the rules; any other is solved exactly with the
// opponent to move - classic boards by src/classic-solver.js, Chaos boards by
// the whole reachable graph in src/chaos-solver.js - and the child's value is
// negated. A position is kept only when the mover wins, exactly one action
// wins, that action is not an immediate win (finding it takes lookahead), at
// least one other action loses, and at least one other action does not lose
// at once - otherwise the win is merely the only parry of a threat.
//
// The mix is weighted to standard 6x7 Connect Four, with a few smaller
// classic boards, and Chaos on boards whose whole reachable graph the solver
// can hold: early in the game on 4x4 Connect 3, late on the Connect 4
// boards, where the graphs of earlier positions outgrow SOLVER_LIMITS.
//
// The PRNG is seeded and nothing reads the clock or Math.random to decide
// anything, so the same source always writes the same file. Regenerate with
//
//   node scripts/neural-strength-positions.mjs [--out <path>] [--verbose]
//
// which rewrites tests/fixtures/neural-strength-positions.json in about a
// quarter of an hour on one core. New positions need a new calibration of
// tests/strength/neural-strength.mjs on the shipped network.

import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import {
  ACTION_DROP, ACTION_FLIP, ACTION_ROTATE_CCW, ACTION_ROTATE_CW, EMPTY, RED, YELLOW,
  applyAction, createBoard, immediateWinningActions, legalActions, otherPlayer,
  positionKey, resolveActionOutcome, sameAction,
} from '../src/engine.js';
import { solveClassicPosition } from '../src/classic-solver.js';
import { solveChaosPosition } from '../src/chaos-solver.js';
import { isEntryPoint } from './entry-point.mjs';

export const FIXTURE = fileURLToPath(new URL('../tests/fixtures/neural-strength-positions.json', import.meta.url));

export const WIN = 1;
export const DRAW = 0;
export const LOSS = -1;

const SEED = 0x5eed504;
const TRANSFORM_CHANCE = 0.25;

// Fail-closed bounds on one exact solve: about five seconds of classic
// search, or one second of building a Chaos graph. A candidate that exceeds
// one is skipped, never guessed, and both counts are deterministic, so the
// skip is too.
export const SOLVER_LIMITS = Object.freeze({ maximumNodes: 500_000, maximumStates: 60_000 });

// ---- actions ----------------------------------------------------------------

/** A stable name for an action: `drop 3`, `flip`, `rotateCW`, `rotateCCW`. */
export function actionLabel(action) {
  return action.type === ACTION_DROP ? `drop ${action.column}` : action.type;
}

export function parseActionLabel(label) {
  const drop = /^drop (\d)$/.exec(label);
  if (drop) return { type: ACTION_DROP, column: Number(drop[1]) };
  if ([ACTION_FLIP, ACTION_ROTATE_CW, ACTION_ROTATE_CCW].includes(label)) return { type: label };
  throw new RangeError(`Unknown action label: ${label}`);
}

// ---- positions --------------------------------------------------------------

function encodeBoard(board) {
  return board.map((row) => row.join(''));
}

/** The position a fixture entry describes, in the engine's shape. */
export function decodePosition(entry) {
  const board = entry.board.map((row) => [...row].map(Number));
  return {
    board,
    currentPlayer: entry.currentPlayer,
    connect: entry.connect,
    chaosMode: entry.chaosMode,
  };
}

export function pieceCount(board) {
  let pieces = 0;
  for (const row of board) for (const cell of row) if (cell !== EMPTY) pieces += 1;
  return pieces;
}

/**
 * One key for every position the network cannot tell apart. Its input is
 * mover-relative, so a position and its colour swap with the other player
 * to move are the same test, and a mirror image differs only by symmetry.
 */
export function equivalenceKey({ board, currentPlayer, connect, chaosMode }) {
  const relative = board.map((row) => row.map((cell) => {
    if (cell === EMPTY) return EMPTY;
    return cell === currentPlayer ? RED : YELLOW;
  }));
  const mirrored = relative.map((row) => [...row].reverse());
  const keys = [relative, mirrored].map((candidate) => positionKey(candidate, RED, connect, chaosMode));
  return keys[0] < keys[1] ? keys[0] : keys[1];
}

// ---- exact values -------------------------------------------------------------

function solveExact(position, limits = SOLVER_LIMITS) {
  return position.chaosMode
    ? solveChaosPosition(position, { maximumStates: limits.maximumStates })
    : solveClassicPosition(position, { maximumNodes: limits.maximumNodes });
}

function isLimitError(error) {
  return error?.code === 'CHAOS_GRAPH_LIMIT' || error?.code === 'CLASSIC_EXACT_NODE_LIMIT';
}

/**
 * The exact value of the position and of every legal action, all for the
 * player to move. `immediate` marks an action the rules decide at once;
 * `losesAtOnce` one that loses by the rules or leaves the opponent an
 * immediate win. `nodes` is the size of the whole proof - classic search
 * nodes or Chaos graph states, root and children together - which is
 * deterministic, so it can choose what is cheap to re-check.
 */
export function exactValues(position, limits = SOLVER_LIMITS, root = solveExact(position, limits)) {
  const { board, currentPlayer, connect, chaosMode } = position;
  const opponent = otherPlayer(currentPlayer);
  let nodes = root.nodes;
  const actions = legalActions(board, chaosMode).map((action) => {
    const applied = applyAction(board, action, currentPlayer);
    const lastDrop = action.type === ACTION_DROP ? { row: applied.row, column: applied.column } : null;
    const outcome = resolveActionOutcome(applied.board, connect, currentPlayer, action.type, lastDrop);
    if (outcome.status === 'won') {
      const value = outcome.winner === currentPlayer ? WIN : LOSS;
      return { action, value, immediate: true, losesAtOnce: value === LOSS };
    }
    if (outcome.status === 'draw') return { action, value: DRAW, immediate: true, losesAtOnce: false };
    const child = solveExact({ board: applied.board, currentPlayer: opponent, connect, chaosMode }, limits);
    nodes += child.nodes;
    // A draw stays a draw; `-0` would not compare equal in a fixture.
    const value = child.value === DRAW ? DRAW : -child.value;
    const losesAtOnce = immediateWinningActions(applied.board, opponent, connect, chaosMode).length > 0;
    return { action, value, immediate: false, losesAtOnce };
  });
  // The solver's own answer for the root must agree with the best action, so
  // an orientation slip in the child solves fails here instead of in the file.
  const best = Math.max(...actions.map((entry) => entry.value));
  if (best !== root.value) {
    throw new Error(`Root value ${root.value} disagrees with the best action value ${best} `
      + `at ${positionKey(board, currentPlayer, connect, chaosMode)}.`);
  }
  return { value: root.value, action: root.action, depth: chaosMode ? root.depth : null, nodes, actions };
}

/** Why a position is not a strength test, or null when it is one. */
export function rejection(position, solved) {
  if (solved.value !== WIN) return 'the mover does not win';
  const wins = solved.actions.filter((entry) => entry.value === WIN);
  if (wins.length !== 1) return `${wins.length} actions win`;
  if (!sameAction(wins[0].action, solved.action)) {
    throw new Error(`The solver chose ${actionLabel(solved.action)} but only ${actionLabel(wins[0].action)} wins.`);
  }
  if (wins[0].immediate) return 'the only win is immediate';
  if (!solved.actions.some((entry) => entry.value === LOSS)) return 'no action loses';
  // When every other action loses at once, the winning move is simply the
  // only one that parries a threat: the root's own policy finds it with no
  // search at all, so such a position cannot tell a working search from a
  // broken one.
  if (solved.actions.every((entry) => entry.value === WIN || entry.losesAtOnce)) {
    return 'every other action loses at once';
  }
  return null;
}

// ---- sampling -----------------------------------------------------------------

/** mulberry32, as in tests/classic-solver.test.js. */
function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function pick(items, random) {
  return items[Math.floor(random() * items.length)];
}

function play(board, mover, connect, action) {
  const applied = applyAction(board, action, mover);
  const lastDrop = action.type === ACTION_DROP ? { row: applied.row, column: applied.column } : null;
  return { board: applied.board, outcome: resolveActionOutcome(applied.board, connect, mover, action.type, lastDrop) };
}

/** Whether an action loses outright or leaves the opponent an immediate win. */
function handsOverWin(board, mover, connect, chaosMode, action) {
  const { board: next, outcome } = play(board, mover, connect, action);
  if (outcome.status === 'won') return outcome.winner !== mover;
  if (outcome.status === 'draw') return false;
  return immediateWinningActions(next, otherPlayer(mover), connect, chaosMode).length > 0;
}

function sensibleAction(board, mover, connect, chaosMode, random) {
  const wins = immediateWinningActions(board, mover, connect, chaosMode);
  if (wins.length > 0) return pick(wins, random);
  const actions = legalActions(board, chaosMode);
  const safe = actions.filter((action) => !handsOverWin(board, mover, connect, chaosMode, action));
  const pool = safe.length > 0 ? safe : actions;
  if (!chaosMode) return pick(pool, random);
  // Three transforms among up to ten drops would make a uniform choice drop
  // almost every time; choosing the kind first keeps transforms in play.
  const drops = pool.filter((action) => action.type === ACTION_DROP);
  const transforms = pool.filter((action) => action.type !== ACTION_DROP);
  if (drops.length === 0) return pick(transforms, random);
  if (transforms.length === 0) return pick(drops, random);
  return pick(random() < TRANSFORM_CHANCE ? transforms : drops, random);
}

/**
 * Every position one sensible random game reaches with the game still
 * running, in order, each with its mover to play.
 */
export function* sampledGame({ rows, cols, connect, chaosMode }, random) {
  let board = createBoard(rows, cols);
  let mover = RED;
  // Counted as the game counts them: the start is the first occurrence.
  const seen = new Map([[positionKey(board, mover, connect, chaosMode), 1]]);
  for (;;) {
    yield { board, currentPlayer: mover, connect, chaosMode };
    const action = sensibleAction(board, mover, connect, chaosMode, random);
    const { board: next, outcome } = play(board, mover, connect, action);
    if (outcome.status !== 'playing') return;
    board = next;
    mover = otherPlayer(mover);
    // The game's threefold-repetition draw ends a sampled game too, and
    // keeps a game of transforms alone from running forever.
    const key = positionKey(board, mover, connect, chaosMode);
    const count = (seen.get(key) ?? 0) + 1;
    seen.set(key, count);
    if (count >= 3) return;
  }
}

// ---- the mix ------------------------------------------------------------------

export const BOARDS = Object.freeze([
  { rows: 6, cols: 7, connect: 4, chaosMode: false, count: 24, pieces: [18, 30] },
  { rows: 5, cols: 6, connect: 4, chaosMode: false, count: 4, pieces: [10, 24] },
  { rows: 6, cols: 5, connect: 4, chaosMode: false, count: 4, pieces: [10, 24] },
  { rows: 5, cols: 5, connect: 4, chaosMode: false, count: 4, pieces: [8, 20] },
  { rows: 4, cols: 4, connect: 3, chaosMode: true, count: 6, pieces: [3, 10] },
  { rows: 4, cols: 4, connect: 4, chaosMode: true, count: 4, pieces: [11, 14] },
  { rows: 4, cols: 5, connect: 4, chaosMode: true, count: 5, pieces: [14, 19] },
  { rows: 5, cols: 4, connect: 4, chaosMode: true, count: 5, pieces: [14, 19] },
  { rows: 5, cols: 5, connect: 4, chaosMode: true, count: 4, pieces: [18, 24] },
]);

/** The rules a position is played under, as its fixture id begins. */
export function ruleName({ rows, cols, connect, chaosMode }) {
  return `${chaosMode ? 'chaos' : 'classic'}-${rows}x${cols}-c${connect}`;
}

/** Each board draws from its own stream, so adding or reordering boards
 * leaves every other board's positions - and their calibration - alone. */
function boardSeed(spec) {
  let hash = SEED;
  for (const character of ruleName(spec)) hash = Math.imul(hash ^ character.charCodeAt(0), 0x01000193);
  return hash >>> 0;
}

/** Positions for one board, half with each colour to move. */
export function collect(spec, { seed = boardSeed(spec), maximumGames = 5_000, log = () => {} } = {}) {
  const random = seededRandom(seed);
  const kept = [];
  const seen = new Set();
  const byColour = { [RED]: 0, [YELLOW]: 0 };
  const perColour = Math.ceil(spec.count / 2);
  const stats = { games: 0, candidates: 0, limited: 0 };
  const [minimum, maximum] = spec.pieces;
  for (let game = 0; game < maximumGames && kept.length < spec.count; game += 1) {
    stats.games += 1;
    for (const position of sampledGame(spec, random)) {
      const pieces = pieceCount(position.board);
      if (pieces > maximum) break;
      // A rotation transposes the board; the other shape is another spec's.
      if (pieces < minimum || position.board.length !== spec.rows) continue;
      if (byColour[position.currentPlayer] >= perColour) continue;
      // An available immediate win would be the only one the fixture could
      // record, and it needs no lookahead.
      if (immediateWinningActions(position.board, position.currentPlayer, position.connect,
        position.chaosMode).length > 0) continue;
      const key = equivalenceKey(position);
      if (seen.has(key)) continue;
      seen.add(key);
      stats.candidates += 1;
      const started = performance.now();
      let solved;
      try {
        // Most candidates are not won for the mover at all; one solve of the
        // root settles that before any action is solved.
        const root = solveExact(position);
        solved = root.value === WIN ? exactValues(position, SOLVER_LIMITS, root) : { value: root.value };
      } catch (error) {
        if (!isLimitError(error)) throw error;
        stats.limited += 1;
        log(`${ruleName(spec)} game ${game} pieces ${pieces}: beyond the solver limits`);
        continue;
      }
      const reason = rejection(position, solved);
      log(`${ruleName(spec)} game ${game} pieces ${pieces}: ${reason ?? 'kept'} `
        + `(${(performance.now() - started).toFixed(0)} ms)`);
      if (reason) continue;
      byColour[position.currentPlayer] += 1;
      kept.push({ position, solved });
      // One position per game keeps the fixture from following one line.
      break;
    }
  }
  return { kept, stats };
}

function entryFor(spec, index, { position, solved }) {
  const win = solved.actions.find((entry) => entry.value === WIN);
  return {
    id: `${ruleName(spec)}-${String(index + 1).padStart(2, '0')}`,
    rows: spec.rows,
    cols: spec.cols,
    connect: spec.connect,
    chaosMode: spec.chaosMode,
    currentPlayer: position.currentPlayer,
    pieces: pieceCount(position.board),
    board: encodeBoard(position.board),
    win: actionLabel(win.action),
    depth: solved.depth,
    nodes: solved.nodes,
    values: Object.fromEntries(solved.actions.map((entry) => [actionLabel(entry.action), entry.value])),
  };
}

function formatFixture(fixture) {
  // One position per line: readable, and a regenerated file diffs by position.
  const { positions, ...header } = fixture;
  const head = JSON.stringify(header, null, 2).replace(/\n}$/, '');
  const lines = positions.map((entry) => `    ${JSON.stringify(entry)}`);
  return `${head},\n  "positions": [\n${lines.join(',\n')}\n  ]\n}\n`;
}

export async function main(argv = process.argv.slice(2)) {
  const out = argv.includes('--out') ? argv[argv.indexOf('--out') + 1] : FIXTURE;
  const verbose = argv.includes('--verbose');
  const log = verbose ? (line) => process.stderr.write(`${line}\n`) : () => {};
  const positions = [];
  for (const spec of BOARDS) {
    const started = performance.now();
    const { kept, stats } = collect(spec, { log });
    if (kept.length < spec.count) {
      throw new Error(`${ruleName(spec)}: found ${kept.length} of ${spec.count} positions in ${stats.games} games.`);
    }
    positions.push(...kept.map((found, at) => entryFor(spec, at, found)));
    process.stderr.write(`${ruleName(spec)}: ${kept.length} kept of ${stats.candidates} candidates `
      + `(${stats.limited} beyond the solver limits) from ${stats.games} games, `
      + `${((performance.now() - started) / 1000).toFixed(1)} s\n`);
  }
  const fixture = {
    description: 'Positions in which exactly one legal action keeps a forced win for the player to move, '
      + 'with the exact value of every legal action (1 win, 0 draw, -1 loss, for the mover). '
      + 'Boards list rows from the top as in src/engine.js: 0 empty, 1 Red, 2 Yellow. '
      + '`depth` counts the plies to the forced win where the solver reports it (Chaos only); '
      + '`nodes` is the size of the exact proof (classic search nodes or Chaos graph states). '
      + 'Written by scripts/neural-strength-positions.mjs; do not edit by hand.',
    seed: SEED,
    solverLimits: SOLVER_LIMITS,
    positions,
  };
  await writeFile(out, formatFixture(fixture));
  process.stderr.write(`${positions.length} positions written to ${out}\n`);
}

if (isEntryPoint(import.meta.url)) await main();
