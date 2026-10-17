import assert from 'node:assert/strict';
import test from 'node:test';

import { chooseMove, exactChaosEndgame } from '../src/ai.js';
import { choosePreparedMove } from '../src/ai-worker.js';
import {
  ACTION_DROP, RED, YELLOW, applyAction, legalActions, otherPlayer, positionKey,
  resolveActionOutcome,
} from '../src/engine.js';

// Easy plays at random, but it takes a win when it has one, and otherwise it
// never plays a move that lets the opponent win at once while another move is
// safe. In Chaos a transform can complete the opponent's line itself, and a
// drop can leave the opponent a transform that completes it. ai.test.js checks
// Easy's safety only on a classic board, where it blocks a drop.

/** 6x7 Connect 4 Chaos, Yellow to move, as the page posts a position for its
 * AI, here the round's only position so far. `top` has the top piece of
 * columns 1 to 4, at alternating heights, so they make no line. A flip turns
 * the board upside down and drops each column's top piece to the bottom row:
 * whoever flips, `top` wins. */
function flipLine(top) {
  const under = otherPlayer(top);
  const board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, top, 0, top, 0, 0],
    [0, top, under, top, under, 0, 0],
  ];
  return {
    board, currentPlayer: YELLOW, connect: 4, chaosMode: true, startingPlayer: RED,
    repetitionCounts: [[positionKey(board, YELLOW, 4, true), 1]],
  };
}

function label(action) {
  return action.type === ACTION_DROP ? `drop ${action.column}` : action.type;
}

/** The board and the engine's outcome once `player` plays `action`. */
function play(board, action, player) {
  const result = applyAction(board, action, player);
  const outcome = resolveActionOutcome(result.board, 4, player, action.type,
    action.type === ACTION_DROP ? { row: result.row, column: result.column } : null);
  return { board: result.board, outcome };
}

/** What each of Yellow's moves leads to: the winner when it ends the game,
 * otherwise Red's replies that win at once, found by playing every one. */
function consequences(board) {
  return Object.fromEntries(legalActions(board, true).map((action) => {
    const { board: after, outcome } = play(board, action, YELLOW);
    if (outcome.status === 'won') {
      return [label(action), outcome.winner === RED ? 'Red wins' : 'Yellow wins'];
    }
    return [label(action), legalActions(after, true).filter((reply) => {
      const replied = play(after, reply, RED).outcome;
      return replied.status === 'won' && replied.winner === RED;
    }).map(label)];
  }));
}

// Easy draws one entry of a list that holds each move it may play: on seven
// columns a drop four to seven times, more the nearer the centre, and a
// transform twice, so 43 entries at most. A hundred draws spread evenly over
// [0, 1) reach every entry, so the sweep sees every move Easy can play.
const DRAWS = 100;

/** Easy's move at each draw, through the worker's choosePreparedMove, which
 * plays the page's AI moves, and through chooseMove, which the page calls
 * itself when the worker fails. */
async function easyMoves(position) {
  const moves = { choosePreparedMove: [], chooseMove: [] };
  for (let index = 0; index < DRAWS; index += 1) {
    const options = { difficulty: 'easy', aiPlayer: YELLOW, random: () => (index + 0.5) / DRAWS };
    moves.choosePreparedMove.push(label((await choosePreparedMove(position, options)).action));
    moves.chooseMove.push(label(chooseMove(position, options).action));
  }
  return moves;
}

test('easy AI never plays a Chaos move that hands Red a win, at any draw', async () => {
  const position = flipLine(RED);
  // Yellow's own flip completes Red's line, and drops 0, 5 and 6 leave Red
  // the flip and no winning drop. A drop on columns 1 to 4 covers one of
  // Red's tops, and neither rotation leaves Red a win. No move wins for
  // Yellow, so Easy has no win to take instead.
  assert.deepEqual(consequences(position.board), {
    'drop 0': ['flip'], 'drop 1': [], 'drop 2': [], 'drop 3': [], 'drop 4': [],
    'drop 5': ['flip'], 'drop 6': ['flip'], flip: 'Red wins', rotateCW: [], rotateCCW: [],
  });
  // Easy never asks the exact Chaos endgame, which with 37 empty cells would
  // not take this position anyway: every move below is Easy's own pick.
  assert.equal(exactChaosEndgame(position, {}, YELLOW).eligible, false);

  const safe = ['drop 1', 'drop 2', 'drop 3', 'drop 4', 'rotateCW', 'rotateCCW'];
  for (const [entry, moves] of Object.entries(await easyMoves(position))) {
    for (const move of moves) {
      assert.ok(safe.includes(move), `${entry} played ${move}, which hands Red a win`);
    }
    // Each safe move is drawn, so the test cannot pass on one lucky pick.
    assert.deepEqual(safe.filter((move) => moves.includes(move)), safe, entry);
  }
});

// ai.test.js has Easy flip to win on a board where nothing else is safe, so
// it passes whether Easy takes its win or only avoids losing.
test('easy AI flips to win at every draw, though nine other moves are safe', async () => {
  const position = flipLine(YELLOW);
  // Only the flip wins, and no other move lets Red win at once.
  assert.deepEqual(consequences(position.board), {
    'drop 0': [], 'drop 1': [], 'drop 2': [], 'drop 3': [], 'drop 4': [],
    'drop 5': [], 'drop 6': [], flip: 'Yellow wins', rotateCW: [], rotateCCW: [],
  });
  for (const [entry, moves] of Object.entries(await easyMoves(position))) {
    assert.deepEqual([...new Set(moves)], ['flip'], entry);
  }
});
