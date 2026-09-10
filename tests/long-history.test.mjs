import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { applyAction, createBoard, otherPlayer, positionKey, resolveActionOutcome } from '../src/engine.js';

test('the long Python search fixture is a legal game ending in third repetition', async () => {
  const fixture = JSON.parse(await readFile(new URL('./fixtures/long-transform-era.json', import.meta.url), 'utf8'));
  const transforms = ['flip', 'rotateCW', 'rotateCCW'];
  let board = createBoard(fixture.rows, fixture.cols);
  let mover = 1;
  const key = () => positionKey(board, mover, fixture.connect, true);
  const counts = new Map([[key(), 1]]);
  const actions = [...fixture.drops.map(column => ({ type: 'drop', column })),
    ...fixture.transforms.map(i => ({ type: transforms[i] }))];
  for (const [i, action] of actions.entries()) {
    const applied = applyAction(board, action, mover);
    assert.ok(applied, `legal move ${i + 1}`);
    const outcome = resolveActionOutcome(applied.board, fixture.connect, mover, action.type,
      action.type === 'drop' ? { row: applied.row, column: applied.column } : null);
    assert.equal(outcome.status, 'playing', `no terminal result at move ${i + 1}`);
    board = applied.board;
    mover = otherPlayer(mover);
    counts.set(key(), (counts.get(key()) ?? 0) + 1);
    assert.ok(counts.get(key()) < 3, `no earlier repetition draw at move ${i + 1}`);
  }
  assert.equal(actions.length, 245);
  assert.deepEqual(board, fixture.board);
  assert.equal(mover, fixture.mover);
  const applied = applyAction(board, { type: fixture.nextAction }, mover);
  assert.equal(resolveActionOutcome(applied.board, fixture.connect, mover, fixture.nextAction).status, 'playing');
  board = applied.board;
  mover = otherPlayer(mover);
  assert.equal(counts.get(key()), 2, 'the next flip is the third occurrence');
});
