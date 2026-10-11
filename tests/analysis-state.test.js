import assert from 'node:assert/strict';
import test from 'node:test';

import { currentNeuralEstimate, neuralEstimateCopy, neuralOutcome, searchSummary } from '../src/analysis-state.js';

test('Perfect policies report their verified decisions like the 6x7 strategy', () => {
  // The classic policies and complete Chaos certificates name the count
  // policyEntryCount, and the page showed nothing for them.
  assert.equal(searchSummary({ strategyEntryCount: 227_455 }).strategyEntryCount, 227_455);
  assert.equal(searchSummary({ policyEntryCount: 470_494 }).strategyEntryCount, 470_494);
  assert.equal(searchSummary({}).strategyEntryCount, null);
});

test('only a win/draw/loss split that adds up to one is kept as an estimate', () => {
  assert.deepEqual(neuralOutcome([0.5, 0.2, 0.3]), [0.5, 0.2, 0.3]);
  assert.deepEqual(neuralOutcome(Float64Array.of(1, 0, 0)), [1, 0, 0]);
  // The search's own sums can be a rounding error away from one.
  assert.deepEqual(neuralOutcome([0.1, 0.2, 0.7 + 1e-9]), [0.1, 0.2, 0.7 + 1e-9]);
  for (const bad of [undefined, null, 0.4, 'win', [0.5, 0.5], [0.25, 0.25, 0.25, 0.25], [0.5, 0.2, 0.2],
    [1.2, -0.2, 0], [0.5, NaN, 0.5], [Infinity, 0, 0], ['0.5', '0.2', '0.3'], { 0: 1, 1: 0, 2: 0, length: 3 }]) {
    assert.equal(neuralOutcome(bad), null, JSON.stringify(bad));
  }
  // What the page stores of a result: a corrupt split is no estimate at all.
  assert.deepEqual(searchSummary({ solver: 'neural', outcome: [0.6, 0.3, 0.1] }).outcome, [0.6, 0.3, 0.1]);
  assert.equal(searchSummary({ solver: 'neural', outcome: [0.6, 0.6, 0.1] }).outcome, null);
  assert.equal(searchSummary({ solver: 'bitboard', score: 1 }).outcome, null);
});

test('the panel reads a network estimate as whole percentages that add up to 100', () => {
  assert.deepEqual(neuralEstimateCopy({ outcome: [0.62, 0.1, 0.28], simulations: 128 }), {
    label: 'AI 62% · Draw 10% · You 28%',
    description: 'Neural network estimate after 128 simulations',
    // The bar is the human's share, as for the heuristic, with half the draws.
    youShare: 33,
    text: 'Neural network estimate after 128 simulations: an AI win 62 percent, a draw 10 percent, your win 28 percent',
  });
  // An even split leaves a point over, and it goes to neither side.
  const even = neuralEstimateCopy({ outcome: [1 / 3, 1 / 3, 1 / 3], simulations: 1 });
  assert.equal(even.label, 'AI 33% · Draw 34% · You 33%');
  assert.equal(even.description, 'Neural network estimate after 1 simulation');
  assert.equal(even.youShare, 50);
  // Rounded down 41 + 18 + 40 = 99; the largest remainder, the human's .7, takes the point.
  assert.equal(neuralEstimateCopy({ outcome: [0.413, 0.18, 0.407], simulations: 2 }).label, 'AI 41% · Draw 18% · You 41%');
  assert.equal(neuralEstimateCopy({ outcome: [1, 0, 0], simulations: 0 }).label, 'AI 100% · Draw 0% · You 0%');
  assert.equal(neuralEstimateCopy({ outcome: [0, 0, 1], simulations: 0 }).description, 'Neural network estimate');
  let seed = 7;
  const random = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  for (let trial = 0; trial < 2_000; trial += 1) {
    const weights = [random(), random() ** 3, random()];
    const total = weights[0] + weights[1] + weights[2];
    const outcome = weights.map((weight) => weight / total);
    const { label, youShare } = neuralEstimateCopy({ outcome, simulations: 64 });
    const [ai, draw, you] = label.match(/\d+/g).map(Number);
    assert.equal(ai + draw + you, 100, label);
    outcome.forEach((share, index) => assert.ok(Math.abs([ai, draw, you][index] - share * 100) < 1, `${outcome}: ${label}`));
    assert.equal(youShare, you + draw / 2);
  }
});

// An estimate describes one position: the one on the board when the round
// (state.version) and its move count are those it was made for.
test('an estimate made for another round or move is not shown', () => {
  const search = { solver: 'neural', nodes: 96, outcome: [0.5, 0.3, 0.2] };
  const madeFor = { version: 4, moveCount: 7 };
  assert.deepEqual(currentNeuralEstimate(search, madeFor, { version: 4, moveCount: 7 }),
    { outcome: [0.5, 0.3, 0.2], simulations: 96 });
  // Undo, a new round, a restored round and new rules all begin a new version.
  assert.equal(currentNeuralEstimate(search, madeFor, { version: 5, moveCount: 7 }), null);
  // A move since, either side's.
  assert.equal(currentNeuralEstimate(search, madeFor, { version: 4, moveCount: 8 }), null);
  // A page that never made one: a reload restores lastSearch, not the estimate.
  assert.equal(currentNeuralEstimate(search, null, { version: 4, moveCount: 7 }), null);
  // Before the network has spoken, or with nothing usable to say.
  assert.equal(currentNeuralEstimate({ solver: 'neural-searching', note: 'Neural search' }, madeFor,
    { version: 4, moveCount: 7 }), null);
  assert.equal(currentNeuralEstimate(null, madeFor, { version: 4, moveCount: 7 }), null);
  assert.equal(currentNeuralEstimate({ ...search, outcome: [0.5, 0.5, 0.5] }, madeFor, { version: 4, moveCount: 7 }), null);
});
