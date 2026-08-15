import { applyAction, boardDimensions, resolveActionOutcome } from './engine.js';
import { isExactClassicPosition, solveClassicPosition } from './classic-solver.js';
import { perfectClassicRole } from './perfect-classic-policy.js';

const MATE_SCORE = 1_000_000;

function now() {
  return globalThis.performance?.now?.() ?? Date.now();
}

function pieceCount(board) {
  let count = 0;
  for (const row of board) {
    for (const cell of row) if (cell !== 0) count += 1;
  }
  return count;
}

export function isPerfectClassicVariant(position) {
  if (!position || position.chaosMode === true || position.connect !== 4) return false;
  const { rows, cols } = boardDimensions(position.board ?? []);
  return rows >= 4 && rows <= 7 && cols >= 4 && cols <= 7
    && isExactClassicPosition(position);
}

/**
 * Selects a game-theoretically exact move on configurable non-Chaos boards.
 * A verified policy is used above its handoff boundary; the exact solver owns
 * every reached endgame. Missing policy data never falls back heuristically.
 */
export function choosePerfectClassicMove(position, options = {}) {
  const difficulty = options.difficulty ?? position?.difficulty ?? 'medium';
  if (difficulty !== 'perfect' || !isPerfectClassicVariant(position)) return null;

  const aiPlayer = options.aiPlayer ?? position.currentPlayer;
  if (aiPlayer !== position.currentPlayer) {
    throw new RangeError('Perfect classic AI can only choose a move for the side to move.');
  }
  if (options.maximumDepth !== undefined) {
    throw new RangeError('Perfect classic AI does not accept a bounded-depth override.');
  }

  const start = now();
  const policy = options.perfectClassicPolicy ?? null;
  if (policy) {
    const rows = position.board.length;
    const columns = position.board[0]?.length ?? 0;
    const role = perfectClassicRole(position.startingPlayer, aiPlayer);
    if (policy.rows !== rows || policy.columns !== columns
        || policy.connect !== position.connect || policy.role !== role) {
      throw new Error('Perfect classic policy metadata does not match the current round.');
    }

    const entry = policy.lookup(
      position.board,
      position.currentPlayer,
      aiPlayer,
      position.startingPlayer,
    );
    if (entry?.action) {
      const applied = applyAction(position.board, entry.action, position.currentPlayer);
      if (!applied) throw new Error('Perfect classic policy returned an illegal action.');
      const outcome = resolveActionOutcome(
        applied.board,
        position.connect,
        position.currentPlayer,
        entry.action.type,
        { row: applied.row, column: applied.column },
      );
      const terminalValue = outcome.status === 'draw'
        ? 0
        : outcome.status === 'won'
          ? outcome.winner === aiPlayer ? 1 : -1
          : null;
      if (terminalValue !== null && terminalValue !== entry.outcome) {
        throw new Error('Perfect classic policy outcome conflicts with its terminal move.');
      }
      const action = { ...entry.action };
      const result = {
        action,
        value: entry.outcome,
        score: entry.outcome === 0
          ? 0
          : entry.outcome * (MATE_SCORE - pieceCount(position.board)),
        depth: 0,
        nodes: 0,
        elapsedMs: now() - start,
        tableHits: 0,
        cutoffs: 0,
        tableStores: 0,
        tableCollisions: 0,
        principalVariation: [action],
        solved: true,
        solver: 'perfect-classic-policy',
        policyRole: policy.role,
        policyHandoffRemaining: policy.handoffRemaining,
        policyEntryCount: policy.entryCount,
        policyClosureStates: policy.closureStates,
      };
      if (typeof options.onIteration === 'function') {
        try {
          options.onIteration(result);
        } catch {
          // Search telemetry must never affect an exact policy result.
        }
      }
      return result;
    }

    const remaining = policy.rows * policy.columns - pieceCount(position.board);
    if (remaining > policy.handoffRemaining) {
      throw new Error(
        `Perfect classic policy coverage gap with ${remaining} cells remaining.`,
      );
    }
  } else if (options.requirePerfectClassicPolicy === true) {
    throw new Error('The verified perfect classic policy could not be loaded.');
  }

  return solveClassicPosition(position, {
    ...options,
    aiPlayer,
    maximumNodes: options.classicMaximumNodes ?? Infinity,
  });
}
