import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { chooseMove } from '../src/ai.js';
import { RED } from '../src/engine.js';
import { decodePerfectBook, loadPerfectBook } from '../src/perfect-book.js';

function emptyBoard() {
  return Array.from({ length: 6 }, () => Array(7).fill(0));
}

test('the committed perfect-play book is valid and contains the solved root move', async () => {
  const book = await loadPerfectBook();
  assert.equal(book.version, 1);
  assert.ok(book.entryCount >= 1);

  const root = book.lookup(0n);
  assert.ok(root);
  assert.equal(root.outcome, 1);
  assert.notEqual(root.moveMask & (1 << 3), 0);
});

test('standard play uses an exact book move before allocating a search table', async () => {
  const book = await loadPerfectBook();
  const result = chooseMove({
    board: emptyBoard(),
    currentPlayer: RED,
    connect: 4,
    chaosMode: false,
  }, {
    difficulty: 'brutal',
    aiPlayer: RED,
    perfectBook: book,
  });

  assert.equal(result.solver, 'perfect-book');
  assert.equal(result.solved, true);
  assert.deepEqual(result.action, { type: 'drop', column: 3 });
  assert.equal(result.nodes, 0);
});

test('the decoder rejects malformed books instead of making an unverified move', async () => {
  const bytes = new Uint8Array(
    await readFile(new URL('../assets/perfect-book.bin', import.meta.url)),
  );

  assert.throws(() => decodePerfectBook(bytes.slice(0, bytes.length - 1)), /length mismatch/);

  const invalidOutcome = bytes.slice();
  invalidOutcome[invalidOutcome.length - 1] = 9;
  assert.throws(() => decodePerfectBook(invalidOutcome), /outcomes must be/);
});
