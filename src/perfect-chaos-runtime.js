import {
  applyAction,
  boardDimensions,
  resolveActionOutcome,
  supportsPerfectChaosConfig,
} from './engine.js';
import { perfectChaosCompleteRole } from './perfect-chaos-complete.js';

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

export function isPerfectChaosVariant(position) {
  if (!position || position.chaosMode !== true || !Array.isArray(position.board)) return false;
  const { rows, cols } = boardDimensions(position.board);
  return supportsPerfectChaosConfig(rows, cols, position.connect, true);
}

/**
 * Plays the committed complete Chaos solution. The certificate covers every
 * position reachable from the empty board under its own policy, so there is no
 * handoff to search: a missing record means the certificate does not match this
 * round, and that fails closed rather than quietly reverting to bounded search.
 */
export function choosePerfectChaosMove(position, options = {}) {
  const difficulty = options.difficulty ?? position?.difficulty ?? 'medium';
  if (difficulty !== 'perfect' || !isPerfectChaosVariant(position)) return null;

  const aiPlayer = options.aiPlayer ?? position.currentPlayer;
  if (aiPlayer !== position.currentPlayer) {
    throw new RangeError('Perfect Chaos AI can only choose a move for the side to move.');
  }
  if (options.maximumDepth !== undefined) {
    throw new RangeError('Perfect Chaos AI does not accept a bounded-depth override.');
  }

  const policy = options.perfectChaosCompletePolicy ?? null;
  if (!policy) {
    throw new Error('The verified complete Chaos policy could not be loaded.');
  }

  const start = now();
  const { rows, cols: columns } = boardDimensions(position.board);
  const role = perfectChaosCompleteRole(position.startingPlayer, aiPlayer);
  // The board may currently be rotated, so the policy's shape is checked
  // against both orientations of its orbit.
  const shapeMatches = (policy.rows === rows && policy.columns === columns)
    || (policy.rows === columns && policy.columns === rows);
  if (!shapeMatches || policy.connect !== position.connect || policy.role !== role) {
    throw new Error('Perfect Chaos policy metadata does not match the current round.');
  }

  const entry = policy.lookup(
    position.board,
    position.currentPlayer,
    aiPlayer,
    position.startingPlayer,
  );
  if (!entry?.action) {
    throw new Error('The complete Chaos policy does not cover this reachable position.');
  }

  const applied = applyAction(position.board, entry.action, position.currentPlayer);
  if (!applied) throw new Error('The complete Chaos policy returned an illegal action.');
  const outcome = resolveActionOutcome(
    applied.board,
    position.connect,
    position.currentPlayer,
    entry.action.type,
    entry.action.type === 'drop' ? { row: applied.row, column: applied.column } : null,
  );
  const terminalValue = outcome.status === 'draw'
    ? 0
    : outcome.status === 'won'
      ? outcome.winner === aiPlayer ? 1 : -1
      : null;
  if (terminalValue !== null && terminalValue !== entry.outcome) {
    throw new Error('Perfect Chaos policy outcome conflicts with its terminal move.');
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
    tableResets: 0,
    principalVariation: [action],
    solved: true,
    solver: 'perfect-chaos-complete',
    policyRole: policy.role,
    policyRootValue: policy.rootValue,
    policyEntryCount: policy.entryCount,
    policyClosureStates: policy.closureStates,
  };
  if (typeof options.onIteration === 'function') {
    try {
      options.onIteration(result);
    } catch {
      // Telemetry must never affect a certified result.
    }
  }
  return result;
}
