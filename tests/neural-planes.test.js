import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { CANVAS, PLANES, planeBuffer, writePlanes } from '../src/neural-planes.js';

const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(join(here, 'fixtures', 'neural-planes.json'), 'utf8'));

// The engine numbers rows from the top and the network from the bottom, so
// the encoder has to flip. These positions and their planes come from the
// Python side (scripts/neural-plane-fixture.py); if the two ever disagree
// the browser player is quietly playing a mirrored board.
test('board encoding matches the Python network input', () => {
  const area = CANVAS * CANVAS;
  for (const kase of fixture.cases) {
    const { rows, columns, connect, chaos, grid, planes } = kase;
    const buffer = planeBuffer(1);
    // grid[0] is the top row; the encoder is asked for rows counted from
    // the bottom, which is what the network expects.
    writePlanes(buffer, 0, rows, columns, connect, chaos,
      (row, column) => grid[rows - 1 - row][column]);

    for (let plane = 0; plane < PLANES; plane += 1) {
      for (let row = 0; row < CANVAS; row += 1) {
        for (let column = 0; column < CANVAS; column += 1) {
          const got = buffer[plane * area + row * CANVAS + column];
          const want = planes[plane][row][column];
          // Float32Array rounds, so connect/10 is compared with tolerance.
          assert.ok(Math.abs(got - want) < 1e-6,
            `${rows}x${columns} c${connect} ${chaos ? 'chaos' : 'classic'}: `
            + `plane ${plane} at row ${row}, column ${column}: ${got} != ${want}`);
        }
      }
    }
  }
});

test('a piece of the player to move lands on the first plane', () => {
  const buffer = planeBuffer(1);
  const area = CANVAS * CANVAS;
  // A single mover piece in the bottom-left corner of a 4x5 board.
  writePlanes(buffer, 0, 4, 5, 4, false, (row, column) => (row === 0 && column === 0 ? 1 : 0));
  assert.equal(buffer[0], 1, 'mover plane, bottom-left');
  assert.equal(buffer[area], 0, 'opponent plane stays empty');
  assert.equal(buffer[2 * area], 1, 'the cell is on the board');
  assert.equal(buffer[2 * area + 4 * CANVAS], 0, 'row 4 is outside a 4-row board');
  assert.ok(Math.abs(buffer[3 * area] - 0.4) < 1e-6,
    'connect length is carried as connect/10');
  assert.equal(buffer[4 * area], 0, 'classic boards carry a zero chaos flag');
});
