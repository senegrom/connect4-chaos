import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import {
  ACTION_DROP, EMPTY, RED, YELLOW, applyAction, createBoard, otherPlayer, positionKey, resolveActionOutcome,
} from '../src/engine.js';
import { bestAction, searchPosition } from '../src/neural-search.js';

// A fixed stand-in for the network: its logits depend only on what the
// network reads - the board, the mover and the repetition count - drawn from
// a hash of them, so a position always gets the same answer. Some outcome
// logits are masked, as a network certain of a result masks them.
function standIn(board, mover, repeated) {
  let seed = 0x811c9dc5;
  const mix = (value) => { seed = Math.imul(seed ^ value, 0x01000193) >>> 0; };
  mix(board.length);
  mix(board[0].length);
  for (const row of board) for (const cell of row) mix(cell);
  mix(mover);
  mix(repeated);
  const hash = seed;
  const next = () => {                        // mulberry32, in [-3, 3)
    seed = (seed + 0x6d2b79f5) >>> 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (((t ^ (t >>> 14)) >>> 0) / 2 ** 32) * 6 - 3;
  };
  const policy = Float32Array.from({ length: 13 }, next);
  const value = Float32Array.from({ length: 3 }, next);
  const q = Float32Array.from({ length: 39 }, next);
  if (hash % 5 === 0) value[hash % 3] = -Infinity;
  for (let action = 0; action < 13; action += 1) {
    if (((hash >>> action) & 3) === 0) q[action * 3 + (action % 3)] = -Infinity;
  }
  return { policy, value, q };
}

const evaluate = (board, mover, _actions, _connect, _chaosMode, repeated) => Promise.resolve(standIn(board, mover, repeated));
const evaluateMany = (items) => Promise.resolve(items.map(({ board, mover, repeated }) => standIn(board, mover, repeated)));

// Plays the leftmost open column so surely that every simulation follows one
// line, 64 plies deep and on, and every mover expects to win half its games,
// draw 30% and lose 20%: [loss, draw, win] in the network's order.
function leftmostLine(board) {
  const policy = new Float32Array(13).fill(-20);
  policy[board[0].indexOf(EMPTY)] = 20;
  return { policy, value: Float32Array.from([Math.log(0.2), Math.log(0.3), Math.log(0.5)]), q: new Float32Array(39) };
}

/** The position `moves` lead to from an empty board, with the repetition
 * counts the page keeps. A number is a drop in that column. */
function play(rows, cols, connect, chaosMode, moves) {
  let board = createBoard(rows, cols);
  let currentPlayer = RED;
  const counts = new Map([[positionKey(board, currentPlayer, connect, chaosMode), 1]]);
  for (const move of moves) {
    const action = typeof move === 'number' ? { type: ACTION_DROP, column: move } : { type: move };
    const applied = applyAction(board, action, currentPlayer);
    const drop = action.type === ACTION_DROP ? applied : null;
    assert.equal(resolveActionOutcome(applied.board, connect, currentPlayer, action.type, drop).status, 'playing');
    board = applied.board;
    currentPlayer = otherPlayer(currentPlayer);
    const key = positionKey(board, currentPlayer, connect, chaosMode);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return { board, currentPlayer, connect, chaosMode, repetitionCounts: [...counts] };
}

// Between them they take every path a simulation can: leaves one at a time
// and in batches, with the virtual losses and blocked descents batches bring,
// terminal wins and losses, repetition draws, transposed Chaos boards, a stop
// part-way and the depth cutoff.
const SCENARIOS = {
  'classic 6x7 opening': () => ({ position: play(6, 7, 4, false, []), options: { simulations: 96 } }),
  'classic 6x7 tactics in batches of eight': () => ({
    position: play(6, 7, 4, false, [0, 1, 0, 1, 2, 1, 2]), options: { simulations: 128, evaluateMany } }),
  'classic 6x7 tactics stopped after three rounds': () => {
    let checks = 0;
    return { position: play(6, 7, 4, false, [0, 1, 0, 1, 2, 1, 2]),
      options: { simulations: 128, evaluateMany, shouldStop: () => ++checks >= 3 } };
  },
  'Chaos 4x4 connect 3 with a position seen twice, in batches of eight': () => ({
    position: play(4, 4, 3, true, [0, 3, 'flip', 'flip']), options: { simulations: 160, evaluateMany } }),
  'Chaos 4x5 connect 4 turned on its side': () => ({
    position: play(4, 5, 4, true, [2, 2, 1, 'rotateCW', 0]), options: { simulations: 120 } }),
  'a 64-ply line reaches the depth cutoff': () => ({
    position: play(10, 10, 10, false, []), options: { simulations: 80 }, network: leftmostLine }),
};

async function runScenario(name) {
  const { position, options, network } = SCENARIOS[name]();
  const progress = [];
  const result = await searchPosition(position, network ? async (board) => network(board) : evaluate, {
    ...options, onProgress: (...args) => progress.push(args) });
  return { result, progress };
}

// What the search decided before it kept win/draw/loss splits beside its
// values: recorded from src/neural-search.js at be70bff. A split is only
// reported, so every one of these must come out exactly as it did.
const scalars = JSON.parse(readFileSync(new URL('./fixtures/neural-search-scalars.json', import.meta.url), 'utf8'));

for (const name of Object.keys(SCENARIOS)) {
  test(`the search decides as it did before it kept outcomes: ${name}`, async () => {
    const { result, progress } = await runScenario(name);
    const expected = scalars[name];
    // Through JSON, as the recording went, so a value compares bit for bit.
    const same = (value) => JSON.parse(JSON.stringify(value));
    for (const field of ['actions', 'visits', 'actionValues', 'terminalValues', 'completedSimulations', 'evaluations',
      'policy', 'value']) {
      assert.deepEqual(same(result[field]), expected[field], field);
    }
    assert.deepEqual(same(bestAction(result)), expected.best, 'the move');
    assert.deepEqual(progress.map(([completed, simulations]) => [completed, simulations]), expected.progress,
      'the progress reports');
  });
}

// The network's [loss, draw, win] logits at `at`, as the [win, draw, loss]
// split the search reports.
function splitOf(logits, at = 0) {
  const weights = [logits[at + 2], logits[at + 1], logits[at]].map((logit) => Math.exp(logit));
  const total = weights[0] + weights[1] + weights[2];
  return weights.map((weight) => weight / total);
}

function assertNear(actual, expected, label, tolerance = 1e-6) {
  assert.equal(actual.length, expected.length, label);
  actual.forEach((share, index) => assert.ok(Math.abs(share - expected[index]) <= tolerance,
    `${label}: [${actual}] is not [${expected}]`));
}

function assertSplit(split, label) {
  assert.equal(split.length, 3, label);
  assert.ok(split.every((share) => Number.isFinite(share) && share >= 0 && share <= 1), `${label}: [${split}]`);
  assert.ok(Math.abs(split[0] + split[1] + split[2] - 1) < 1e-12, `${label} adds up to ${split[0] + split[1] + split[2]}`);
}

for (const name of Object.keys(SCENARIOS)) {
  test(`every split adds up to one and says what its value says: ${name}`, async () => {
    const { result, progress } = await runScenario(name);
    assertSplit(result.outcome, 'the root');
    assertSplit(result.networkOutcome, 'the value head');
    assert.equal(result.actionOutcomes.length, result.actions.length);
    result.actionOutcomes.forEach((split, index) => {
      assertSplit(split, `action ${index}`);
      // A value is the split's win minus its loss.
      if (result.visits[index] > 0) {
        assert.ok(Math.abs(split[0] - split[2] - result.actionValues[index]) < 1e-9, `action ${index}`);
      }
    });
    assert.ok(Math.abs(result.outcome[0] - result.outcome[2] - result.value) < 1e-9, 'the root');
    // The page shows the estimate while the search runs; the last report is the result's.
    for (const [completed, , outcome] of progress) assertSplit(outcome, `after ${completed} simulations`);
    assert.deepEqual(progress.at(-1)[2], result.outcome);
  });
}

test('each ply between the root and the position a simulation ends in swaps win and loss', async () => {
  // Only column 0 is ever played: Red there, then Yellow on top. Yellow, to
  // move after one ply, expects [win 0.7, draw 0.2, loss 0.1]; Red, to move
  // again after two, [0.6, 0.3, 0.1]. Every other move looks lost.
  const expected = { 1: [0.7, 0.2, 0.1], 2: [0.6, 0.3, 0.1] };
  const network = async (board) => {
    const [win, draw, loss] = expected[board.flat().filter((cell) => cell !== EMPTY).length] ?? [0.4, 0.2, 0.4];
    const policy = new Float32Array(13).fill(-20);
    policy[0] = 20;
    const q = new Float32Array(39);
    for (let action = 1; action < 13; action += 1) q.set([0, -Infinity, -Infinity], action * 3);
    return { policy, value: Float32Array.of(Math.log(loss), Math.log(draw), Math.log(win)), q };
  };
  const result = await searchPosition({ board: createBoard(4, 4), currentPlayer: RED, connect: 3, chaosMode: false },
    network, { simulations: 2 });
  assert.deepEqual(result.visits, [2, 0, 0, 0]);
  // For Red, Yellow's loss of 0.1 is a win and Yellow's win a loss; Red's own
  // split after two plies comes back as it is.
  const columnZero = [(0.1 + 0.6) / 2, (0.2 + 0.3) / 2, (0.7 + 0.1) / 2];
  assertNear(result.actionOutcomes[0], columnZero, 'column 0');
  assertNear(result.outcome, columnZero, 'the root');
  assertNear([result.value], [-0.05], 'the value it agrees with');
  assertNear(result.networkOutcome, [0.4, 0.2, 0.4], 'the value head');
  // The network alone after column 0 is Yellow's value head, for Red; the
  // other columns' positions were never evaluated.
  assertNear(result.actionNetworkOutcomes[0], [0.1, 0.2, 0.7], 'column 0 alone');
  assert.deepEqual(result.actionNetworkOutcomes.slice(1), [null, null, null]);
});

test('a simulation cut off 64 plies down counts the position it stopped in, for its mover', async () => {
  const { result } = await runScenario('a 64-ply line reaches the depth cutoff');
  // Each mover expects [0.5, 0.3, 0.2]. Simulations 1-64 end one ply deeper
  // each and the other 16 at ply 64, so 48 end where Red, the root's mover,
  // is to move and 32 where Yellow is.
  assertNear(result.actionOutcomes[0], [(48 * 0.5 + 32 * 0.2) / 80, 0.3, (48 * 0.2 + 32 * 0.5) / 80], 'column 0');
});

test('a result the rules decided comes back as that result for certain', async () => {
  const blind = async () => ({ policy: new Float32Array(13), value: new Float32Array(3), q: new Float32Array(39) });
  // Yellow wins in column 1.
  const win = await searchPosition(play(6, 7, 4, false, [0, 1, 0, 1, 2, 1, 2]), blind, { simulations: 16 });
  assert.equal(win.terminalValues[1], 1);
  assert.deepEqual(win.actionOutcomes[1], [1, 0, 0]);
  // The network never evaluates a position the rules decided.
  assert.equal(win.actionNetworkOutcomes[1], null);
  // On this 4x4 Connect 4 Chaos board the flip loses, and the one drop left
  // fills the board (tests/neural-terminal-selection.test.js).
  const lossOrDraw = play(4, 4, 4, true, [1, 0, 0, 2, 2, 0, 2, 2, 1, 1, 0, 1, 3, 3, 3]);
  const decided = await searchPosition(lossOrDraw, blind, { simulations: 16 });
  const flip = decided.actions.findIndex((action) => action.type === 'flip');
  const drop = decided.actions.findIndex((action) => action.type === ACTION_DROP);
  assert.deepEqual([decided.terminalValues[flip], decided.terminalValues[drop]], [-1, 0]);
  assert.deepEqual(decided.actionOutcomes[flip], [0, 0, 1]);
  assert.deepEqual(decided.actionOutcomes[drop], [0, 1, 0]);
  assert.deepEqual([decided.actionNetworkOutcomes[flip], decided.actionNetworkOutcomes[drop]], [null, null]);
  // A third occurrence draws: Yellow's flip on this empty board brings back
  // Red to move, which has occurred twice.
  const board = createBoard(4, 4);
  const repeated = await searchPosition({ board, currentPlayer: YELLOW, connect: 4, chaosMode: true,
    repetitionCounts: [[positionKey(board, RED, 4, true), 2], [positionKey(board, YELLOW, 4, true), 2]] },
  blind, { simulations: 16 });
  const again = repeated.actions.findIndex((action) => action.type === 'flip');
  assert.equal(repeated.terminalValues[again], 0);
  assert.deepEqual(repeated.actionOutcomes[again], [0, 1, 0]);
  assert.equal(repeated.actionNetworkOutcomes[again], null);
});

test('a move no simulation tried reports the per-action head, and the root the value head', async () => {
  const { result } = await runScenario('classic 6x7 opening');
  const root = standIn(createBoard(6, 7), RED, 0);
  assert.equal(result.visits[0], 0, 'column 0 was never tried');
  assertNear(result.actionOutcomes[0], splitOf(root.q, 0), 'column 0', 1e-12);
  assertNear(result.networkOutcome, splitOf(root.value), 'the value head', 1e-12);
});

test('a position already drawn reports a draw and no network estimate', async () => {
  const board = createBoard(4, 4);
  const result = await searchPosition({ board, currentPlayer: RED, connect: 4, chaosMode: true,
    repetitionCounts: [[positionKey(board, RED, 4, true), 3]] }, () => assert.fail('the network is not asked'));
  assert.deepEqual([result.outcome, result.actionOutcomes, result.networkOutcome, result.actionNetworkOutcomes],
    [[0, 1, 0], [], null, []]);
});

// What the network says alone, beside what the search makes of it: for the
// root, with every progress report, and for the position each action leads
// to. A child's split is for its own mover, so the root's win is its loss.
for (const name of Object.keys(SCENARIOS)) {
  test(`the network alone, for the root and where each action leads: ${name}`, async () => {
    const { position, network } = SCENARIOS[name]();
    const { result, progress } = await runScenario(name);
    for (const [completed, , , networkOutcome] of progress) {
      assert.deepEqual(networkOutcome, result.networkOutcome, `after ${completed} simulations`);
    }
    assert.ok(Array.isArray(progress[0][3]), 'an array of its own, not the node\'s');
    assert.equal(new Set(progress.map((report) => report[3])).size, progress.length, 'a fresh one each time');

    const mover = otherPlayer(position.currentPlayer);
    const counts = new Map(position.repetitionCounts);
    assert.equal(result.actionNetworkOutcomes.length, result.actions.length);
    result.actions.forEach((action, index) => {
      // Never tried, or decided by the rules: the network never saw it.
      if (result.visits[index] === 0 || result.terminalValues[index] !== null) {
        assert.equal(result.actionNetworkOutcomes[index], null, `action ${index}`);
        return;
      }
      const { board } = applyAction(position.board, action, position.currentPlayer);
      const repeated = counts.get(positionKey(board, mover, position.connect, position.chaosMode)) ?? 0;
      const { value } = network ? network(board) : standIn(board, mover, repeated);
      assertNear(result.actionNetworkOutcomes[index], splitOf(value).reverse(), `action ${index}`, 1e-12);
    });
  });
}
