import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildStrategy,
  decodeStrategy,
  verifyClosure,
  STRATEGY_CONSTANTS,
} from '../scripts/perfect-strategy.mjs';

function neutralScores(sequences) {
  return new Map(sequences.map((sequence) => [sequence, Array(7).fill(0)]));
}

test('the deterministic strategy format covers both starting roles to its handoff', async () => {
  const { bytes, manifest } = await buildStrategy({
    handoffRemaining: 40,
    roles: 'both',
    source: 'deterministic test oracle',
    scoreBatch: neutralScores,
  });
  const decoded = decodeStrategy(bytes);

  assert.equal(decoded.version, 1);
  assert.equal(decoded.handoffRemaining, 40);
  assert.equal(decoded.roleFlags, STRATEGY_CONSTANTS.ROLE_FIRST | STRATEGY_CONSTANTS.ROLE_SECOND);
  assert.equal(decoded.entries.length, 5);
  assert.equal(manifest.entryCount, 5);
  assert.deepEqual(verifyClosure(decoded), manifest.closure);
});

// An oracle that knows only immediate wins: 1 for a column that wins at once
// for the side to move, `otherwise` for every other column.
function immediateWinScores(otherwise) {
  const winsAtOnce = (sequence, column) => {
    const heights = Array(7).fill(0);
    const owner = new Map();
    [...sequence].forEach((digit, ply) => {
      const played = Number(digit) - 1;
      owner.set(`${played},${heights[played]}`, ply % 2);
      heights[played] += 1;
    });
    if (heights[column] >= 6) return false;
    const row = heights[column];
    const mine = (c, r) => (c === column && r === row) || owner.get(`${c},${r}`) === sequence.length % 2;
    return [[1, 0], [0, 1], [1, 1], [1, -1]].some(([dc, dr]) => {
      let count = 1;
      for (const sign of [1, -1]) {
        for (let step = 1; step < 4 && mine(column + sign * dc * step, row + sign * dr * step); step += 1) {
          count += 1;
        }
      }
      return count >= 4;
    });
  };
  return (sequences) => new Map(sequences.map((sequence) => [
    sequence,
    Array.from({ length: 7 }, (_, column) => (winsAtOnce(sequence, column) ? 1 : otherwise)),
  ]));
}

test('the closure refuses a stored draw whose move lets the opponent win at once', async () => {
  // Claiming a draw everywhere, the frontier-minimising choice happily picks
  // moves that hand the opponent an immediate win; those entries lie.
  await assert.rejects(
    buildStrategy({
      handoffRemaining: 33, roles: 'first', source: 'draw-claiming test oracle',
      scoreBatch: immediateWinScores(0),
    }),
    /stores outcome 0, but its move lets the opponent win with column \d/,
  );
  // The same moves stored as losses are consistent, and are counted apart
  // from the non-losing terminals.
  const { manifest } = await buildStrategy({
    handoffRemaining: 33, roles: 'first', source: 'loss-claiming test oracle',
    scoreBatch: immediateWinScores(-1),
  });
  assert.ok(manifest.closure.first.aiLosses > 0);
});

test('the closure refuses a stored outcome that the move itself contradicts', async () => {
  // Neutral scores store 0 even for a move that wins on the spot.
  await assert.rejects(
    buildStrategy({
      handoffRemaining: 35, roles: 'first', source: 'deterministic test oracle',
      scoreBatch: neutralScores,
    }),
    /stores outcome 0, but its move ends the game with 1/,
  );
});

test('strategy decoding rejects truncation and multi-move entries', async () => {
  const { bytes } = await buildStrategy({
    handoffRemaining: 41,
    roles: 'first',
    source: 'deterministic test oracle',
    scoreBatch: neutralScores,
  });

  assert.throws(() => decodeStrategy(bytes.subarray(0, bytes.length - 1)), /length/);
  const invalid = bytes.slice();
  invalid[20] = 3;
  assert.throws(() => decodeStrategy(invalid), /exactly one/);
});
