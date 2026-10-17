import assert from 'node:assert/strict';
import test from 'node:test';

import { chooseMove, exactChaosEndgame } from '../src/ai.js';
import { choosePreparedMove } from '../src/ai-worker.js';
import {
  ACTION_DROP, ACTION_FLIP, RED, YELLOW, applyAction, legalActions, otherPlayer, positionKey,
  resolveActionOutcome,
} from '../src/engine.js';

// Easy plays at random, but it takes a win when it has one, and otherwise it
// never plays a move that lets the opponent win at once while another move is
// safe. In Chaos a transform can complete the opponent's line itself, and a
// drop can leave the opponent a transform that completes it. ai.test.js checks
// Easy's safety only on a classic board, where it blocks a drop.

/** 6x7 Connect 4 Chaos, Yellow to move, as the page posts a position for its
 * AI, here the round's only position so far. `top` has the top piece of
 * columns 0 to 3, at heights 1, 2, 2 and 1, so they make no line. A flip turns
 * the board upside down and drops each column's top piece to the bottom row:
 * whoever flips, `top` wins. */
function flipLine(top) {
  const under = otherPlayer(top);
  const board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, top, top, 0, under, top, 0],
    [top, under, under, top, under, under, 0],
  ];
  return {
    board, currentPlayer: YELLOW, connect: 4, chaosMode: true, startingPlayer: RED,
    repetitionCounts: [[positionKey(board, YELLOW, 4, true), 1]],
  };
}

/** A 6x7 Chaos board with one piece in a column at most, Yellow to move here
 * for the second time in the round. Red has three of columns 0 to 3 on the
 * bottom row and wins with drop 2. The same board with Red to move, which a
 * flip gives, has been seen `flippedSeen` times. */
function flipRepeat(flippedSeen) {
  const board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [RED, RED, 0, RED, 0, YELLOW, YELLOW],
  ];
  return {
    board, currentPlayer: YELLOW, connect: 4, chaosMode: true, startingPlayer: RED,
    repetitionCounts: [
      [positionKey(board, YELLOW, 4, true), 2],
      [positionKey(board, RED, 4, true), flippedSeen],
    ],
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

/** `player`'s moves that win at once, found by playing every one. */
function winningMoves(board, player) {
  return legalActions(board, true).filter((action) => {
    const { outcome } = play(board, action, player);
    return outcome.status === 'won' && outcome.winner === player;
  }).map(label);
}

/** What each of Yellow's moves leads to: the winner when it ends the game,
 * otherwise Red's replies that win at once. */
function consequences(board) {
  return Object.fromEntries(legalActions(board, true).map((action) => {
    const { board: after, outcome } = play(board, action, YELLOW);
    if (outcome.status === 'won') {
      return [label(action), outcome.winner === RED ? 'Red wins' : 'Yellow wins'];
    }
    return [label(action), winningMoves(after, RED)];
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

/** Every move Easy plays must be one of `safe`, at every draw and on both
 * routes, and each safe move must be drawn, so a test cannot pass on one
 * lucky pick. */
async function assertEasyPlaysOnly(position, safe) {
  for (const [entry, moves] of Object.entries(await easyMoves(position))) {
    for (const move of moves) {
      assert.ok(safe.includes(move), `${entry} played ${move}, which hands Red a win`);
    }
    assert.deepEqual(safe.filter((move) => moves.includes(move)), safe, entry);
  }
}

test('easy AI never plays a Chaos move that hands Red a win, at any draw', async () => {
  const position = flipLine(RED);
  // Yellow's own flip completes Red's line, and drops 4, 5 and 6 leave Red
  // the flip and no winning drop. A drop on columns 0 to 3 covers one of
  // Red's tops, and neither rotation leaves Red a win. No move wins for
  // Yellow, so Easy has no win to take instead.
  assert.deepEqual(consequences(position.board), {
    'drop 0': [], 'drop 1': [], 'drop 2': [], 'drop 3': [], 'drop 4': ['flip'],
    'drop 5': ['flip'], 'drop 6': ['flip'], flip: 'Red wins', rotateCW: [], rotateCCW: [],
  });
  // Once Yellow's flip has made Red's line, Red has no win to play: the edge
  // and column 4's Yellow pair close the line at both ends, so no drop
  // lengthens it, Red's own flip undoes it, and either rotation makes
  // Yellow's line too, which loses for Red, the mover. So only Easy's check
  // of the move's own outcome can see that the flip loses, not its check of
  // Red's replies.
  const flipped = play(position.board, { type: ACTION_FLIP }, YELLOW).board;
  assert.deepEqual(winningMoves(flipped, RED), []);
  // Easy never asks the exact Chaos endgame, which with 32 empty cells would
  // not take this position anyway: every move below is Easy's own pick.
  assert.equal(exactChaosEndgame(position, {}, YELLOW).eligible, false);

  const safe = ['drop 0', 'drop 1', 'drop 2', 'drop 3', 'rotateCW', 'rotateCCW'];
  await assertEasyPlaysOnly(position, safe);
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

// A transform adds no piece, so in Chaos a position can come back, and its
// third occurrence draws the round. A move that draws hands Red no win, but
// a second occurrence plays on.
test('easy AI counts a Chaos repetition as a draw only at its third occurrence', async () => {
  const position = flipRepeat(1);
  // A flip drops each column's one piece back where it was, so it changes
  // nothing but the side to move, and Red then wins with drop 2. On the
  // board alone, only drop 2 and the anticlockwise rotation leave Red no
  // win, and no move wins for Yellow.
  assert.deepEqual(play(position.board, { type: ACTION_FLIP }, YELLOW).board, position.board);
  assert.deepEqual(consequences(position.board), {
    'drop 0': ['drop 2'], 'drop 1': ['drop 2'], 'drop 2': [], 'drop 3': ['drop 2'],
    'drop 4': ['drop 2'], 'drop 5': ['drop 2'], 'drop 6': ['drop 2'], flip: ['drop 2'],
    rotateCW: ['drop 0'], rotateCCW: [],
  });

  // Yellow flipped and Red flipped back, so the board with Red to move has
  // been seen once: a flip brings it back a second time, which is no draw,
  // and Red wins.
  await assertEasyPlaysOnly(position, ['drop 2', 'rotateCCW']);
  // Seen twice, it comes back a third time and the round is drawn, so the
  // flip is safe too.
  await assertEasyPlaysOnly(flipRepeat(2), ['drop 2', 'flip', 'rotateCCW']);
});
