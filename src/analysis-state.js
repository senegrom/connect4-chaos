/** A solver's name describes its algorithm, not whether it finished a proof. */
export function searchIsExact(search) {
  return search?.solved === true && Number.isFinite(search.score);
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
  if (searchIsExact(search)) {
    return { label: 'Exact result', description: 'Proved by exact analysis', badge: 'Proved',
      text: search.score > 0 ? 'AI can force a win' : search.score < 0
        ? 'You can force a win' : 'Best play leads to a draw' };
  }
  return { label: thinking ? 'Solving exactly…' : 'Perfect play',
    description: 'No completed proof for this position yet', badge: thinking ? 'Searching' : 'Ready',
    text: thinking ? 'The result is not known yet' : 'The AI will choose only game-theoretically optimal moves' };
}

export function searchSummary(result) {
  return {
    score: Number.isFinite(result.score) ? result.score : null,
    depth: result.depth ?? 0,
    nodes: result.nodes ?? 0,
    evaluations: result.evaluations ?? null,
    elapsedMs: result.elapsedMs ?? 0,
    solved: result.solved === true,
    solver: result.solver ?? 'general',
    bookEntryCount: result.bookEntryCount ?? null,
    strategyEntryCount: result.strategyEntryCount ?? null,
    certifiedFromPieces: result.certifiedFromPieces ?? null,
    certifiedThroughPieces: result.certifiedThroughPieces ?? null,
    backend: result.backend ?? null,
  };
}
