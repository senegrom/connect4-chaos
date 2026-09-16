import assert from 'node:assert/strict';
import test from 'node:test';

import {
  ACTION_DROP, ACTION_FLIP, RED, applyAction, createBoard, immediateWinningActions,
  otherPlayer, resolveActionOutcome,
} from '../src/engine.js';
import { bestAction, searchPosition } from '../src/neural-search.js';

function resultWith(terminalValues, policy, actionValues = []) {
  return {
    actions: policy.map((_, column) => ({ type: ACTION_DROP, column })),
    terminalValues,
    policy,
    visits: policy.map((share) => share * 100),
    actionValues,
  };
}

test('a confirmed loss cannot beat a draw or an unknown move, even with all the policy mass', () => {
  for (const alternative of [0, null, undefined]) {
    for (const lossIndex of [0, 1]) {
      const terminals = [alternative, alternative];
      const policy = [0, 0];
      terminals[lossIndex] = -1;
      policy[lossIndex] = 1;
      const result = resultWith(terminals, policy);
      assert.equal(bestAction(result), result.actions[1 - lossIndex]);
    }
  }
});

test('an absent terminal entry is unknown, not evidence that all moves lose', () => {
  const result = resultWith([-1], [1, 0]);
  assert.equal(bestAction(result), result.actions[1]);
});

test('the highest policy among non-losses wins, not the highest value or stale visit count', () => {
  const result = resultWith([-1, 0, null], [0.8, 0.05, 0.15], [-1, 0, -0.9]);
  result.visits = [90, 9, 1];
  assert.equal(bestAction(result), result.actions[2]);
});

test('a pessimistic estimate is not a confirmed loss', () => {
  const result = resultWith([null, null], [0.7, 0.3], [-1, 0.9]);
  assert.equal(bestAction(result), result.actions[0]);
});

test('all confirmed losses still return the highest-policy legal move', () => {
  const result = resultWith([-1, -1, -1], [0.2, 0.5, 0.3], [-1, -1, -1]);
  result.visits = [100, 1, 10];
  assert.equal(bestAction(result), result.actions[1]);
  const only = resultWith([-1], [1], [-1]);
  assert.equal(bestAction(only), only.actions[0]);
});

test('confirmed wins retain priority, ranked by policy among the winning moves', () => {
  const result = resultWith([null, 1, -1, 1], [0.6, 0.05, 0.25, 0.1]);
  assert.equal(bestAction(result), result.actions[3]);
});

test('tied eligible policies keep the searched-value tiebreak and stable ordering', () => {
  const result = resultWith([-1, null, 0], [0.8, 0.1, 0.1], [-1, -0.5, 0]);
  assert.equal(bestAction(result), result.actions[2]);
  result.actionValues = [-1, 0, 0];
  assert.equal(bestAction(result), result.actions[1]);
});

test('missing terminal metadata leaves all moves eligible', () => {
  const result = resultWith(undefined, [0.1, 0.9]);
  assert.equal(bestAction(result), result.actions[1]);
});

test('legacy visit-only results use the same eligibility filter', () => {
  const result = resultWith([-1, null, 0], [0.8, 0.15, 0.05]);
  delete result.policy;
  assert.equal(bestAction(result), result.actions[1]);
});

test('policy-only results and an empty result need no visit array', () => {
  const result = resultWith([-1, null], [1, 0]);
  delete result.visits;
  assert.equal(bestAction(result), result.actions[1]);
  assert.equal(bestAction({ actions: [], policy: [], terminalValues: [] }), null);
});

test('selection does not mutate the search result', () => {
  const result = resultWith([-1, null, 0], [0.8, 0.15, 0.05], [-1, -0.2, 0]);
  const before = structuredClone(result);
  for (const value of Object.values(result)) Object.freeze(value);
  Object.freeze(result);
  assert.equal(bestAction(result), result.actions[1]);
  assert.deepEqual(result, before);
});

test('every ordering of three terminal states obeys the loss filter and policy ranking', () => {
  const values = [-1, 0, 1, null, undefined];
  const policies = [[0.6, 0.3, 0.1], [0.6, 0.1, 0.3], [0.3, 0.6, 0.1],
    [0.1, 0.6, 0.3], [0.3, 0.1, 0.6], [0.1, 0.3, 0.6]];
  for (const first of values) for (const second of values) for (const third of values) {
    const terminals = [first, second, third];
    for (const policy of policies) {
      const result = resultWith(terminals, policy);
      const index = result.actions.indexOf(bestAction(result));
      let eligible = [0, 1, 2].filter((i) => terminals[i] === 1);
      if (!eligible.length) eligible = [0, 1, 2].filter((i) => terminals[i] !== -1);
      if (!eligible.length) eligible = [0, 1, 2];
      assert.ok(eligible.includes(index));
      assert.equal(policy[index], Math.max(...eligible.map((i) => policy[i])));
    }
  }
});

function drawOrLossPosition() {
  let board = createBoard(4, 4);
  let currentPlayer = RED;
  // A legal history: no win or full board before Yellow's final drop.
  for (const column of [1, 0, 0, 2, 2, 0, 2, 2, 1, 1, 0, 1, 3, 3, 3]) {
    const applied = applyAction(board, { type: ACTION_DROP, column }, currentPlayer);
    assert.equal(resolveActionOutcome(applied.board, 4, currentPlayer, ACTION_DROP, applied).status, 'playing');
    board = applied.board;
    currentPlayer = otherPlayer(currentPlayer);
  }
  assert.deepEqual(immediateWinningActions(board, currentPlayer, 4, true), []);
  return { board, currentPlayer, connect: 4, chaosMode: true };
}

function misleadingNetwork() {
  // Strongly prefer the losing flip. The actual engine, not this evaluator,
  // proves that the flip loses and the drop draws during the search.
  const policy = new Float32Array(13).fill(-40);
  policy[3] = Math.log(0.01);
  policy[10] = Math.log(0.99);
  const q = new Float32Array(39);
  for (let i = 0; i < 13; i += 1) q.set([0, -40, -40], i * 3);
  q.set([Math.log(0.8), -40, Math.log(0.2)], 3 * 3);
  q.set([Math.log(0.05), -40, Math.log(0.95)], 10 * 3);
  return { policy, value: new Float32Array(3), q };
}

for (const [name, simulations, batchSize, stopAfter] of [
  ['single', 16, 1, null],
  ['batched', 16, 8, null],
  ['Move now', 128, 1, 16],
  ['batched Move now', 128, 8, 16],
  ['minimum budget with unexplored alternatives', 2, 1, null],
]) {
  test(`${name}: actual search never selects the confirmed losing flip`, async () => {
    const position = drawOrLossPosition();
    const result = await searchPosition(position, async () => misleadingNetwork(), {
      simulations, batchSize,
      evaluateMany: batchSize > 1 ? async (items) => items.map(() => misleadingNetwork()) : null,
      shouldStop: (done) => stopAfter !== null && done >= stopAfter,
    });
    assert.equal(result.completedSimulations, stopAfter ?? simulations);
    const drop = result.actions.findIndex((action) => action.type === ACTION_DROP);
    const flip = result.actions.findIndex((action) => action.type === ACTION_FLIP);
    assert.equal(result.terminalValues[flip], -1);
    assert.equal(result.terminalValues[drop], simulations === 2 ? null : 0);
    assert.ok(result.policy[flip] > result.policy[drop], 'the losing move really has higher policy');
    const action = bestAction(result);
    assert.equal(action, result.actions[drop]);
    const applied = applyAction(position.board, action, position.currentPlayer);
    assert.equal(resolveActionOutcome(applied.board, 4, position.currentPlayer, action.type, applied).status, 'draw');
  });
}
