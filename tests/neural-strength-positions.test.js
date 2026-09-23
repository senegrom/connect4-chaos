import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import {
  ACTION_DROP, EMPTY, RED, YELLOW, applyAction, isBoardFull, legalActions, otherPlayer,
  resolveActionOutcome, sameAction, winningCells,
} from '../src/engine.js';
import {
  DRAW, FIXTURE, LOSS, WIN, actionLabel, decodePosition, equivalenceKey, exactValues,
  parseActionLabel, pieceCount, rejection, ruleName,
} from '../scripts/neural-strength-positions.mjs';

// The strength test (tests/strength/neural-strength.mjs) trusts this fixture
// completely: a position whose recorded move is not the only win would score
// a correct answer as a miss, or reward a wrong one. These checks keep it
// honest without the model, so they run with every `npm test`.

const fixture = JSON.parse(await readFile(FIXTURE, 'utf8'));
const { positions } = fixture;

test('every fixture entry is well formed', () => {
  assert.ok(Array.isArray(positions) && positions.length > 0);
  const ids = new Set();
  for (const entry of positions) {
    assert.ok(Number.isInteger(entry.rows) && entry.rows >= 4 && entry.rows <= 10, entry.id);
    assert.ok(Number.isInteger(entry.cols) && entry.cols >= 4 && entry.cols <= 10, entry.id);
    assert.ok([3, 4, 5].includes(entry.connect), entry.id);
    assert.equal(typeof entry.chaosMode, 'boolean', entry.id);
    // The strength test groups its report by the rules an id names.
    assert.match(entry.id, new RegExp(`^${ruleName(entry)}-\\d{2}$`));
    assert.ok(!ids.has(entry.id), `${entry.id} is listed twice`);
    ids.add(entry.id);
    assert.ok([RED, YELLOW].includes(entry.currentPlayer), entry.id);
    assert.equal(entry.board.length, entry.rows, entry.id);
    for (const row of entry.board) assert.match(row, new RegExp(`^[012]{${entry.cols}}$`), entry.id);
    assert.equal(pieceCount(decodePosition(entry).board), entry.pieces, entry.id);
    // Chaos proofs report plies to the win; the classic search does not.
    assert.equal(entry.depth === null, !entry.chaosMode, entry.id);
    if (entry.chaosMode) assert.ok(Number.isInteger(entry.depth) && entry.depth >= 3, entry.id);
    assert.ok(Number.isInteger(entry.nodes) && entry.nodes > 0, entry.id);
    for (const [label, value] of Object.entries(entry.values)) {
      assert.equal(actionLabel(parseActionLabel(label)), label, entry.id);
      assert.ok([WIN, DRAW, LOSS].includes(value), `${entry.id} ${label}`);
    }
  }
});

test('every position is a legal, unfinished game with its mover to play', () => {
  for (const entry of positions) {
    const { board, currentPlayer, connect, chaosMode } = decodePosition(entry);
    // Gravity: nothing floats above an empty cell.
    for (let column = 0; column < entry.cols; column += 1) {
      for (let row = 1; row < entry.rows; row += 1) {
        if (board[row - 1][column] === EMPTY) continue;
        assert.notEqual(board[row][column], EMPTY, `${entry.id}: a piece floats in column ${column}`);
      }
    }
    assert.equal(winningCells(board, RED, connect).length, 0, `${entry.id}: Red already has a line`);
    assert.equal(winningCells(board, YELLOW, connect).length, 0, `${entry.id}: Yellow already has a line`);
    assert.equal(isBoardFull(board), false, entry.id);
    if (!chaosMode) {
      // Classic moves alternate, so the mover has as many pieces as the
      // opponent, or one fewer.
      const mine = board.flat().filter((cell) => cell === currentPlayer).length;
      const theirs = board.flat().filter((cell) => cell === otherPlayer(currentPlayer)).length;
      assert.ok(theirs - mine === 0 || theirs - mine === 1, `${entry.id}: ${mine} against ${theirs}`);
    }
  }
});

test('the recorded move is legal, the only win, and not an immediate one', () => {
  for (const entry of positions) {
    const { board, currentPlayer, connect, chaosMode } = decodePosition(entry);
    const actions = legalActions(board, chaosMode);
    assert.deepEqual(Object.keys(entry.values), actions.map(actionLabel), `${entry.id} lists every legal action`);
    const win = parseActionLabel(entry.win);
    assert.ok(actions.some((action) => sameAction(action, win)), `${entry.id}: ${entry.win} is not legal`);
    assert.deepEqual(Object.entries(entry.values).filter(([, value]) => value === WIN).map(([label]) => label),
      [entry.win], `${entry.id}: exactly the recorded move wins`);
    assert.ok(Object.values(entry.values).includes(LOSS), `${entry.id}: some action loses`);
    // Whatever the rules decide at once must match the recorded value.
    for (const action of actions) {
      const applied = applyAction(board, action, currentPlayer);
      const lastDrop = action.type === ACTION_DROP ? { row: applied.row, column: applied.column } : null;
      const outcome = resolveActionOutcome(applied.board, connect, currentPlayer, action.type, lastDrop);
      if (sameAction(action, win)) assert.equal(outcome.status, 'playing', `${entry.id}: the win needs lookahead`);
      if (outcome.status === 'won') {
        assert.equal(entry.values[actionLabel(action)], outcome.winner === currentPlayer ? WIN : LOSS,
          `${entry.id} ${actionLabel(action)}`);
      } else if (outcome.status === 'draw') {
        assert.equal(entry.values[actionLabel(action)], DRAW, `${entry.id} ${actionLabel(action)}`);
      }
    }
  }
});

test('no two positions are the same test for the network', () => {
  // Its input is mover-relative, so a colour swap is the same position, and
  // a mirror image differs only by symmetry.
  const keys = new Map();
  for (const entry of positions) {
    const key = equivalenceKey(decodePosition(entry));
    assert.ok(!keys.has(key), `${entry.id} repeats ${keys.get(key)}`);
    keys.set(key, entry.id);
  }
});

test('each rule set asks both colours', () => {
  const movers = new Map();
  for (const entry of positions) {
    const counts = movers.get(ruleName(entry)) ?? { [RED]: 0, [YELLOW]: 0 };
    counts[entry.currentPlayer] += 1;
    movers.set(ruleName(entry), counts);
  }
  for (const [rules, counts] of movers) {
    assert.ok(Math.abs(counts[RED] - counts[YELLOW]) <= 1, `${rules}: ${counts[RED]} Red, ${counts[YELLOW]} Yellow`);
  }
});

// Re-proving all of them takes over a minute. Every position was proved in
// full when the fixture was written; here the cheapest proof of each rule
// set is redone, then as many of the next cheapest as fit in this many
// solver nodes (about a second per 60 000).
const RESOLVE_NODES = 400_000;

function cheapestProofs() {
  const byNodes = [...positions].sort((a, b) => a.nodes - b.nodes || a.id.localeCompare(b.id));
  const chosen = new Map();
  for (const entry of byNodes) if (!chosen.has(ruleName(entry))) chosen.set(ruleName(entry), entry);
  const everyRuleSet = [...chosen.values()];
  let spent = everyRuleSet.reduce((sum, entry) => sum + entry.nodes, 0);
  const rest = [];
  for (const entry of byNodes) {
    if (everyRuleSet.includes(entry)) continue;
    if (spent + entry.nodes > RESOLVE_NODES) break;
    spent += entry.nodes;
    rest.push(entry);
  }
  return { everyRuleSet, chosen: [...everyRuleSet, ...rest] };
}

test('the exact solvers still prove the recorded values of the cheapest positions', () => {
  const { everyRuleSet, chosen } = cheapestProofs();
  // A regenerated fixture whose cheapest proofs alone overrun the budget
  // would quietly slow every `npm test`; say so instead.
  assert.ok(everyRuleSet.reduce((sum, entry) => sum + entry.nodes, 0) <= RESOLVE_NODES,
    'the cheapest proof of each rule set no longer fits in RESOLVE_NODES');
  for (const entry of chosen) {
    const position = decodePosition(entry);
    const solved = exactValues(position);
    assert.equal(rejection(position, solved), null, `${entry.id} is no longer a unique-win position`);
    assert.deepEqual(Object.fromEntries(solved.actions.map((found) => [actionLabel(found.action), found.value])),
      entry.values, entry.id);
    assert.equal(solved.depth, entry.depth, entry.id);
  }
});
