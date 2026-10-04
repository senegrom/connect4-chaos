import assert from 'node:assert/strict';
import test from 'node:test';

import { parseScoredLine } from '../scripts/standard-board.mjs';

test('a scored line is the root or a move sequence, then one score per column', () => {
  // The book's pack and the strategy oracle read the exact solver's output
  // through this parser, and only the root's line has no sequence.
  assert.deepEqual(parseScoredLine('-2 -1 0 1 0 -1 -2', 1), { sequence: '', scores: [-2, -1, 0, 1, 0, -1, -2] });
  assert.deepEqual(parseScoredLine('44 -3 -2 -1 0 -1 -2 -3', 2),
    { sequence: '44', scores: [-3, -2, -1, 0, -1, -2, -3] });
  assert.deepEqual(parseScoredLine('﻿  1234567\t0 0 0 0 0 0 0 \r', 3),
    { sequence: '1234567', scores: [0, 0, 0, 0, 0, 0, 0] });
  // Columns are 1-7, and a line has seven scores.
  for (const line of ['04 1 2 3 4 5 6 7', '40 1 2 3 4 5 6 7', '48 1 2 3 4 5 6 7', '4a 1 2 3 4 5 6 7', '1 2 3 4 5 6', '4 1 2 3 4 5 6 7 8']) {
    assert.throws(() => parseScoredLine(line, 9, 'oracle output line'),
      (error) => error.message === `Invalid oracle output line 9: ${line}`, line);
  }
  assert.throws(() => parseScoredLine('4 1 2 x 4 5 6 7', 9, 'oracle output line'),
    (error) => error.message === 'Non-integer score in oracle output line 9.');
  assert.throws(() => parseScoredLine('1 2 3 4 5 6', 5), (error) => error.message === 'Invalid scored line 5: 1 2 3 4 5 6');
});
