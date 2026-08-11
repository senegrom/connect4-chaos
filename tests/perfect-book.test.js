import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { chooseMove } from '../src/ai.js';
import { RED } from '../src/engine.js';
import { decodePerfectBook, loadPerfectBook } from '../src/perfect-book.js';

function emptyBoard() {
  return Array.from({ length: 6 }, () => Array(7).fill(0));
}

test('the committed perfect-play book matches its manifest and contains the solved root move', async () => {
  const bookUrl = new URL('../assets/perfect-book.bin', import.meta.url);
  const manifestUrl = new URL('../data/perfect-book.manifest.json', import.meta.url);
  const [book, bytes, manifestText] = await Promise.all([
    loadPerfectBook(bookUrl),
    readFile(bookUrl),
    readFile(manifestUrl, 'utf8'),
  ]);
  const manifest = JSON.parse(manifestText);
  assert.equal(book.version, manifest.format);
  assert.equal(book.maxPly, manifest.maxPly);
  assert.equal(book.entryCount, manifest.entryCount);
  assert.equal(book.byteLength, manifest.byteLength);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), manifest.sha256);

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

  const invalidReservedByte = bytes.slice();
  invalidReservedByte[7] = 1;
  assert.throws(() => decodePerfectBook(invalidReservedByte), /reserved header byte/);

  const invalidKey = bytes.slice();
  new DataView(invalidKey.buffer).setBigUint64(12, 1n << 49n, true);
  assert.throws(() => decodePerfectBook(invalidKey), /outside the standard board/);
});
