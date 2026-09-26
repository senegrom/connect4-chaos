import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

import {
  ACTION_DROP, ACTION_FLIP, ACTION_ROTATE_CCW, ACTION_ROTATE_CW, RED,
  applyAction, createBoard, positionKey,
} from '../src/engine.js';

const app = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');

function definition(name) {
  const start = app.search(new RegExp(`^function ${name}\\(`, 'm'));
  assert.notEqual(start, -1, `missing ${name} in app.js`);
  return app.slice(start, app.indexOf('\n}\n', start) + 2);
}

function load(names, globals) {
  const context = vm.createContext(globals);
  vm.runInContext(names.map(definition).join('\n'), context);
  return context;
}

// score-store.js accepts result ids of this shape only.
const RESULT_ID = /^[a-zA-Z0-9._:-]{1,128}$/;

test('a finished round is recorded under the round and where it ended', () => {
  const state = {
    roundId: 'round-1', moveCount: 7, board: createBoard(6, 7), currentPlayer: RED,
    config: { connect: 4, chaosMode: false },
  };
  const context = load(['fnv1a', 'finishedResultId'], { state, positionKey });
  const first = context.finishedResultId();
  assert.match(first, RESULT_ID);
  assert.equal(context.finishedResultId(), first, 'retrying the same ending is the same result');
  // A second tab that resumed the same round, or a duplicated tab, shares the
  // round id: ending somewhere else must be a result of its own.
  state.board = applyAction(state.board, { type: ACTION_DROP, column: 3 }, RED).board;
  assert.notEqual(context.finishedResultId(), first);
  state.roundId = 'x'.repeat(128);
  assert.match(context.finishedResultId(), RESULT_ID, 'a long round id still yields a valid result id');
});

test('the AI move is announced to screen readers', () => {
  const elements = { selectedColumnStatus: { textContent: '' } };
  const context = load(['announceAiAction'], {
    elements, ACTION_DROP, ACTION_FLIP, ACTION_ROTATE_CW, ACTION_ROTATE_CCW,
  });
  for (const [action, text] of [
    [{ type: ACTION_DROP, column: 3 }, 'AI dropped a disc in column 4.'],
    [{ type: ACTION_FLIP }, 'AI flipped the board.'],
    [{ type: ACTION_ROTATE_CW }, 'AI rotated the board clockwise.'],
    [{ type: ACTION_ROTATE_CCW }, 'AI rotated the board counter-clockwise.'],
  ]) {
    context.announceAiAction(action);
    assert.equal(elements.selectedColumnStatus.textContent, text);
  }
});
