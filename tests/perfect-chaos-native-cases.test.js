import assert from 'node:assert/strict';
import test from 'node:test';

import { validateNativeCases } from '../scripts/perfect-chaos-native.mjs';

const CASES = [
  { name: '2x2-connect2', value: 1, states: 6 },
  { name: '3x3-connect3', value: 0, states: 628 },
  { name: '6x7-endgame-fixture', value: 1, states: 2585, action: { type: 'rotateCW' } },
];

test('the native Chaos self-test accepts exactly its three reference cases', () => {
  validateNativeCases(CASES);
  validateNativeCases([...CASES].reverse());
});

test('a case reported twice cannot stand in for a missing one', () => {
  // Right count, every name known, every record correct - and the 6x7
  // fixture never ran.
  assert.throws(
    () => validateNativeCases([CASES[0], CASES[1], CASES[0]]),
    /Native verifier repeated case 2x2-connect2/,
  );
  assert.throws(() => validateNativeCases(CASES.slice(0, 2)), /returned 2 cases; expected 3/);
  assert.throws(
    () => validateNativeCases([CASES[0], CASES[1], { ...CASES[2], states: 2586 }]),
    /mismatch for 6x7-endgame-fixture/,
  );
});
