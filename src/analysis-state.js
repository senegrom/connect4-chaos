/** A solver's name describes its algorithm, not whether it finished a proof. */
export function searchIsExact(search) {
  return search?.solved === true && search.proofScope !== 'board-only' && Number.isFinite(search.score);
}

export function searchUsesExactSolver(search) {
  return ['bitboard-exact', 'classic-exact', 'chaos-exact-graph', 'perfect-strategy',
    'perfect-book', 'perfect-classic-policy', 'perfect-chaos-complete'].includes(search?.solver);
}

export function exactAnalysisCopy({ status, winner, search, thinking = false }) {
  if (status === 'won' || status === 'draw') {
    return { label: 'Round result', description: 'Final position', badge: 'Final',
      text: status === 'draw' ? 'The position ended in a draw'
        : winner === 1 ? 'You won this position' : 'AI won this position' };
  }
  if (search?.proofScope === 'board-only') {
    return { label: 'Certified policy', description: 'Repetition history can change the certificate’s board-only value',
      badge: 'Conditional', text: 'The certified move is shown; no history-aware outcome has been proved' };
  }
  if (searchIsExact(search)) {
    return { label: 'Exact result', description: 'Proved by exact analysis', badge: 'Proved',
      text: search.score > 0 ? 'AI can force a win' : search.score < 0
        ? 'You can force a win' : 'Best play leads to a draw' };
  }
  return { label: thinking ? 'Solving exactly…' : 'Perfect play',
    description: 'No completed proof for this position yet', badge: thinking ? 'Searching' : 'Ready',
    text: thinking ? 'The result is not known yet' : 'The AI will choose only game-theoretically optimal moves' };
}

/**
 * A network estimate as the page keeps one: [win, draw, loss], three finite
 * shares of at least 0 that add up to 1. Anything else - no estimate yet, a
 * stand-in's result, a corrupt save - is null.
 */
export function neuralOutcome(value) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value.length !== 3) return null;
  const shares = Array.from(value);
  if (!shares.every((share) => Number.isFinite(share) && share >= 0)) return null;
  return Math.abs(shares[0] + shares[1] + shares[2] - 1) <= 1e-6 ? shares : null;
}

/**
 * The network's estimate for the position on the board, as `{ outcome,
 * simulations, networkOutcome }`, or null. `estimateFor` holds the round
 * (`version`) and the move count the latest estimate was made for. Undo, a
 * new or restored round, new rules and every move change one of them, and
 * the panel then shows its heuristic until the network has spoken about the
 * new position - or until the page tags the estimate a restored snapshot
 * kept for it (app.js). `networkOutcome` is the network's own output for the
 * position, beside the search's figure, or null when there is none to show.
 */
export function currentNeuralEstimate(search, estimateFor, { version, moveCount }) {
  if (estimateFor?.version !== version || estimateFor.moveCount !== moveCount) return null;
  const outcome = neuralOutcome(search?.outcome);
  return outcome && { outcome, simulations: search.nodes ?? 0, networkOutcome: neuralOutcome(search.networkOutcome) };
}

/** Whole percentages that add up to 100: each share rounded down, and the
 * points left over to the largest remainders. Equal remainders give the draw
 * its point first, so an even split reads 33, 34, 33 and favours no side. */
function wholePercents(shares) {
  const total = shares[0] + shares[1] + shares[2];
  const exact = shares.map((share) => (share / total) * 100);
  const percents = exact.map((percent) => Math.floor(percent));
  let left = 100 - percents[0] - percents[1] - percents[2];
  for (const index of [1, 0, 2].sort((a, b) => (exact[b] - percents[b]) - (exact[a] - percents[a]))) {
    if (left <= 0) break;
    percents[index] += 1;
    left -= 1;
  }
  return percents;
}

/** Whole percentages as the panel writes them. Only the space after each dot
 * can break: a narrow panel wraps them into whole shares, never "Draw" from
 * its number. */
function percentsLabel([ai, draw, you]) {
  return `AI\u00a0${ai}%\u00a0· Draw\u00a0${draw}%\u00a0· You\u00a0${you}%`;
}

/**
 * What the analysis panel says about a network estimate. The outcome is the
 * AI's, so its loss is the human's win. The bar shows the human's share, as
 * it does for the heuristic, and gives each side half of the draws. `network`
 * is a smaller line under the bar: the network's own output for the same
 * position, written as the label is, or null when there is none.
 */
export function neuralEstimateCopy({ outcome, simulations, networkOutcome = null }) {
  const percents = wholePercents(outcome);
  const [ai, draw, you] = percents;
  const description = simulations > 0
    ? `Neural network estimate after ${simulations} simulation${simulations === 1 ? '' : 's'}`
    : 'Neural network estimate';
  return {
    label: percentsLabel(percents),
    description,
    youShare: you + draw / 2,
    text: `${description}: an AI win ${ai} percent, a draw ${draw} percent, your win ${you} percent`,
    network: networkOutcome ? `Network alone: ${percentsLabel(wholePercents(networkOutcome))}` : null,
  };
}

export function searchSummary(result) {
  return {
    score: Number.isFinite(result.score) ? result.score : null,
    depth: result.depth ?? 0,
    nodes: result.nodes ?? 0,
    evaluations: result.evaluations ?? null,
    elapsedMs: result.elapsedMs ?? 0,
    solved: result.solved === true,
    proofScope: result.proofScope ?? null,
    solver: result.solver ?? 'general',
    bookEntryCount: result.bookEntryCount ?? null,
    // The 6x7 strategy reports strategyEntryCount; the classic policies and
    // complete Chaos certificates report policyEntryCount. The page shows
    // either as "N verified decisions".
    strategyEntryCount: result.strategyEntryCount ?? result.policyEntryCount ?? null,
    certifiedFromPieces: result.certifiedFromPieces ?? null,
    certifiedThroughPieces: result.certifiedThroughPieces ?? null,
    backend: result.backend ?? null,
    // The neural opponent's [win, draw, loss] for the AI in the position its
    // move leads to: the search's, and the network's own output alone.
    outcome: neuralOutcome(result.outcome),
    networkOutcome: neuralOutcome(result.networkOutcome),
  };
}
