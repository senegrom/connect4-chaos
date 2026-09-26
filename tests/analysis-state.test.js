import assert from 'node:assert/strict';
import test from 'node:test';

import { searchSummary } from '../src/analysis-state.js';

test('Perfect policies report their verified decisions like the 6x7 strategy', () => {
  // The classic policies and complete Chaos certificates name the count
  // policyEntryCount, and the page showed nothing for them.
  assert.equal(searchSummary({ strategyEntryCount: 227_455 }).strategyEntryCount, 227_455);
  assert.equal(searchSummary({ policyEntryCount: 470_494 }).strategyEntryCount, 470_494);
  assert.equal(searchSummary({}).strategyEntryCount, null);
});
