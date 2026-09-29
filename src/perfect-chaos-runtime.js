import { boardPieceCount, copyRepetitionCounts, repetitionHistoryIsFresh } from './ai.js';
import {
  applyAction,
  boardDimensions,
  legalActions,
  otherPlayer,
  positionKey,
  resolveActionOutcome,
  supportsPerfectChaosConfig,
} from './engine.js';
import { perfectChaosCompleteRole } from './perfect-chaos-complete.js';

const MATE_SCORE = 1_000_000;

function now() {
  return globalThis.performance?.now?.() ?? Date.now();
}

/** A legal action that brings a position round for the third time, which
 * ends the round in a draw, or null. Only a transform can: a drop adds a
 * piece, and pieces are never removed. */
function repetitionEscape(position, occurrences) {
  for (const action of legalActions(position.board, true)) {
    if (action.type === 'drop') continue;
    const applied = applyAction(position.board, action, position.currentPlayer);
    const outcome = resolveActionOutcome(applied.board, position.connect, position.currentPlayer, action.type);
    if (outcome.status === 'playing' && occurrences(applied.board) >= 2) return action;
  }
  return null;
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

  const history = copyRepetitionCounts(position.repetitionCounts);
  const occurrences = (board) => history.get(positionKey(board, otherPlayer(position.currentPlayer),
    position.connect, position.chaosMode)) ?? 0;
  // Board wins/full-board draws take precedence over repetition, just as in app.js.
  const repetitionDraw = outcome.status === 'playing' && occurrences(applied.board) >= 2;
  // The certificate knows no history. Lost on the board, the AI can still
  // draw when some action makes a third occurrence: that beats the certified
  // loss, and it used to play the loss.
  const escape = entry.outcome === -1 && !repetitionDraw ? repetitionEscape(position, occurrences) : null;
  const drawn = repetitionDraw || escape !== null;
  const value = drawn ? 0 : entry.outcome;

  // A board-only certificate cannot prove a value the history can still
  // change. That takes a position of this piece layer seen twice already
  // (repetitionHistoryIsFresh): a certified line never repeats a position, so
  // one earlier occurrence cannot, and repetition only ever draws, so a
  // certified draw stands whatever the history. Keep the certified move and
  // qualify its value rather than inventing a proof.
  const historyUnproved = terminalValue === null && !drawn && entry.outcome !== 0
    && !repetitionHistoryIsFresh(history, position.board);
  const action = { ...(escape ?? entry.action) };
  const result = {
    action,
    value,
    score: value === 0
      ? 0
      : value * (MATE_SCORE - boardPieceCount(position.board)),
    depth: 0,
    nodes: 0,
    elapsedMs: now() - start,
    tableHits: 0,
    cutoffs: 0,
    tableResets: 0,
    principalVariation: [action],
    solved: !historyUnproved,
    proofScope: historyUnproved ? 'board-only' : 'history-aware',
    drawReason: drawn ? 'repetition' : null,
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
