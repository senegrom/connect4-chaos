import assert from 'node:assert/strict';
import test from 'node:test';

import {
  ACTION_DROP, ACTION_FLIP, ACTION_ROTATE_CCW, ACTION_ROTATE_CW, RED, YELLOW,
  applyAction, createBoard, immediateWinningActions, resolveActionOutcome, sameAction,
} from '../src/engine.js';
import { chooseMove, preferImmediateWin } from '../src/ai.js';
import { chooseMoveWithPerfectClassic } from '../src/ai-worker.js';

// A player that can end the game now should end it. Winning in five instead
// is worth the same game-theoretically, but it leaves the person opposite
// believing they still have a game.

/** Red to move on 6x7 with three in a row on the bottom, at columns 0-2. */
function redWinsAtThree() {
  const board = createBoard(6, 7);
  for (const column of [0, 1, 2]) {
    board[5][column] = RED;
    board[5][column + 4] = YELLOW;      // keep the piece counts equal
  }
  board[4][0] = YELLOW;
  board[4][1] = RED;
  return { board, currentPlayer: RED, connect: 4, chaosMode: false };
}

function wins(position, action) {
  const applied = applyAction(position.board, action, position.currentPlayer);
  const outcome = resolveActionOutcome(applied.board, position.connect, position.currentPlayer,
    action.type, action.type === ACTION_DROP ? { row: applied.row, column: applied.column } : null);
  return outcome.status === 'won' && outcome.winner === position.currentPlayer;
}

test('the winning drop is available and really wins', () => {
  const position = redWinsAtThree();
  const winning = immediateWinningActions(position.board, RED, 4, false);
  assert.deepEqual(winning, [{ type: ACTION_DROP, column: 3 }]);
  assert.ok(wins(position, winning[0]));
});

for (const difficulty of ['easy', 'medium', 'hard', 'brutal']) {
  test(`${difficulty} plays the win in one rather than any other move`, () => {
    const position = redWinsAtThree();
    const result = chooseMove(position, { difficulty, random: () => 0 });
    assert.ok(wins(position, result.action),
      `${difficulty} played ${JSON.stringify(result.action)} with a mate in one available`);
  });
}

test('the prepared route (Perfect and the certified policies) plays the win in one', () => {
  const position = redWinsAtThree();
  const result = chooseMoveWithPerfectClassic(position, { difficulty: 'brutal' });
  assert.ok(wins(position, result.action));
});

// The override must leave a route's own reporting alone: only the move and
// the line change, never the solver, the proof or the statistics.
test('preferring the win keeps every other field the route reported', () => {
  const position = redWinsAtThree();
  const reported = {
    action: { type: ACTION_DROP, column: 6 }, score: 1234, depth: 9, nodes: 42,
    solver: 'chaos-exact-graph', solved: true, principalVariation: [{ type: ACTION_DROP, column: 6 }],
  };
  const result = preferImmediateWin(position, reported);
  assert.deepEqual(result.action, { type: ACTION_DROP, column: 3 });
  assert.deepEqual(result.principalVariation, [{ type: ACTION_DROP, column: 3 }]);
  for (const key of ['score', 'depth', 'nodes', 'solver', 'solved']) {
    assert.deepEqual(result[key], reported[key], key);
  }
});

test('a winning move already chosen is left exactly as it was', () => {
  const position = redWinsAtThree();
  const chosen = { action: { type: ACTION_DROP, column: 3 }, score: 7, principalVariation: [] };
  assert.equal(preferImmediateWin(position, chosen), chosen);
});

test('a position with no win in one is not touched', () => {
  const position = { board: createBoard(6, 7), currentPlayer: RED, connect: 4, chaosMode: false };
  const chosen = { action: { type: ACTION_DROP, column: 0 }, score: 0, principalVariation: [] };
  assert.equal(preferImmediateWin(position, chosen), chosen);
});

// In Chaos a transform can complete a line, so it counts as a winning move -
// but one that completes the opponent's line instead must never be offered.
test('a transform that wins counts, and one that loses does not', () => {
  const board = createBoard(4, 4);
  board[3][0] = RED; board[3][1] = RED; board[3][2] = RED;
  board[3][3] = YELLOW; board[2][0] = YELLOW; board[2][1] = YELLOW;
  const position = { board, currentPlayer: RED, connect: 4, chaosMode: true };
  const winning = immediateWinningActions(board, RED, 4, true);
  for (const action of winning) {
    assert.ok(wins(position, action), `${JSON.stringify(action)} does not actually win`);
  }
  for (const action of [{ type: ACTION_FLIP }, { type: ACTION_ROTATE_CW },
    { type: ACTION_ROTATE_CCW }]) {
    if (wins(position, action)) {
      assert.ok(winning.some((win) => sameAction(win, action)),
        `${JSON.stringify(action)} wins but was not offered`);
    }
  }
});
