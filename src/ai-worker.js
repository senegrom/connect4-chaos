import {
  boardHasWin,
  boardPieceCount,
  chooseMove,
  copyRepetitionCounts,
  exactChaosEndgame,
  preferImmediateWin,
  repetitionHistoryIsFresh,
} from './ai.js';
import { PERFECT_STRATEGY_HANDOFF, isBitboardPosition } from './bitboard.js';
import { solveChaosProofPosition } from './chaos-proof.js';
import { CHAOS_CELL_LIMIT, CHAOS_LOSS } from './chaos-solver.js';
import { perfectClassicRole } from './perfect-classic-policy.js';
import { loadVerifiedPerfectClassicPolicy } from './perfect-classic-verified.js';
import {
  choosePerfectClassicMove,
  isPerfectClassicVariant,
  usesPerfectClassicPolicy,
} from './perfect-classic-runtime.js';
import {
  choosePerfectChaosMove,
  isPerfectChaosVariant,
} from './perfect-chaos-runtime.js';
import {
  authorizeChaosPolicy,
  loadVerifiedPerfectChaosCompletePolicy,
  perfectChaosCompleteRole,
} from './perfect-chaos-complete.js';
// These load with the worker, right after the page checked that the site was
// not redeployed; only their tables wait until a move needs them. A worker
// outlives a deploy, and a module it imported on first use came from whatever
// the site served by then.
import { loadPerfectBook as loadBook, PERFECT_BOOK_CERTIFICATE } from './perfect-book.js';
import { loadPerfectChaosPolicy as loadChaosPolicy } from './perfect-chaos-prefix.js';
import { loadPerfectStrategy as loadStrategy } from './perfect-strategy.js';
import {
  EMPTY,
  RED,
  YELLOW,
  createBoard,
  positionKey,
  sameAction,
} from './engine.js';

const BOOK_DIFFICULTIES = new Set(['medium', 'hard', 'brutal']);
const CHAOS_PROOF_DEFAULTS = Object.freeze({
  medium: { dropDepth: 1, maximumStates: 10_000 },
  hard: { dropDepth: 2, maximumStates: 50_000 },
  brutal: { dropDepth: 2, maximumStates: 100_000 },
});

async function loadPerfectStrategy(options) {
  return loadStrategy(undefined, options);
}

async function loadPerfectBook(options) {
  try {
    return await loadBook(undefined, options);
  } catch {
    return null;
  }
}

async function loadConfiguredPerfectClassicPolicy(position, aiPlayer, options) {
  const rows = position?.board?.length ?? 0;
  const columns = position?.board?.[0]?.length ?? 0;
  const role = perfectClassicRole(position?.startingPlayer, aiPlayer);
  if (role === null) return null;
  try {
    return await loadVerifiedPerfectClassicPolicy(rows, columns, position.connect, role, options);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`Could not load the verified Perfect classic policy: ${detail}`);
  }
}

export function requireCertifiedChaosPolicy(policy) {
  if (policy === null) return null;
  if (!policy || typeof policy.lookup !== 'function') {
    throw new TypeError('Certified Brutal Chaos policy data is invalid.');
  }

  const lookup = policy.lookup;
  return Object.freeze({
    ...policy,
    lookup(...args) {
      const entry = lookup.apply(policy, args);
      if (!entry?.action) {
        throw new Error(
          'The certified Brutal Chaos policy does not cover this reachable position.',
        );
      }
      return entry;
    },
  });
}

async function loadPerfectChaosPolicy(role, pieceCount, options) {
  try {
    return requireCertifiedChaosPolicy(await loadChaosPolicy(role, pieceCount, null, options));
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`Could not load the certified Brutal Chaos policy: ${detail}`);
  }
}

function standardChaosPosition(position) {
  const rows = position?.board?.length ?? 0;
  const columns = position?.board?.[0]?.length ?? 0;
  return position?.chaosMode === true
    && position?.connect === 4
    && ((rows === 6 && columns === 7) || (rows === 7 && columns === 6));
}

function certifiedStartingPlayer(position) {
  const keys = new Set(copyRepetitionCounts(position?.repetitionCounts).keys());
  const initialBoard = createBoard(6, 7);
  const candidates = [RED, YELLOW].filter((player) => keys.has(
    positionKey(initialBoard, player, 4, true),
  ));
  const declared = position?.startingPlayer;
  if (declared === RED || declared === YELLOW) {
    return candidates.includes(declared) ? declared : null;
  }
  return candidates.length === 1 ? candidates[0] : null;
}

function chaosPolicyIdentity(position, aiPlayer) {
  if (position?.currentPlayer !== aiPlayer || !Array.isArray(position?.board)) return null;
  const startingPlayer = certifiedStartingPlayer(position);
  if (startingPlayer === null) return null;
  let pieceCount = 0;
  for (const row of position.board) {
    if (!Array.isArray(row)) return null;
    for (const cell of row) {
      if (cell === RED || cell === YELLOW) pieceCount += 1;
      else if (cell !== EMPTY) return null;
    }
  }
  return {
    role: aiPlayer === startingPlayer ? 1 : 2,
    pieceCount,
  };
}

function positionAlreadyTerminal(position) {
  if (!Array.isArray(position?.board) || !Number.isInteger(position?.connect)) return true;
  return position.board.every((row) => row.every((cell) => cell !== EMPTY))
    || boardHasWin(position.board, RED, position.connect)
    || boardHasWin(position.board, YELLOW, position.connect);
}

function chaosProofConfiguration(position, options, difficulty) {
  if (options.useChaosProof !== undefined && typeof options.useChaosProof !== 'boolean') {
    throw new TypeError('Chaos proof option must be a boolean.');
  }
  if (options.useChaosProof === false) return null;
  if (options.maximumDepth !== undefined && options.useChaosProof !== true) return null;
  const defaults = CHAOS_PROOF_DEFAULTS[difficulty];
  if (!defaults) return null;

  const boardCells = position.board.length * position.board[0].length;
  const dropDepth = options.chaosProofDropDepth ?? defaults.dropDepth;
  if (!Number.isInteger(dropDepth) || dropDepth < 0 || dropDepth > boardCells) {
    throw new RangeError(`Chaos proof drop depth must be an integer from 0 through ${boardCells}.`);
  }
  if (dropDepth === 0) return null;

  const maximumStates = options.chaosProofMaximumStates ?? defaults.maximumStates;
  if (!Number.isInteger(maximumStates) || maximumStates < 1 || maximumStates > 2_000_000) {
    throw new RangeError('Chaos proof state limit must be an integer from 1 through 2,000,000.');
  }
  return { dropDepth, maximumStates };
}

function proofTelemetry(proof) {
  return {
    dropDepth: proof.depth,
    states: proof.nodes,
    lowerValue: proof.lowerValue,
    upperValue: proof.upperValue,
    actionBounds: proof.actionBounds,
    provenLosingActions: proof.provenLosingActions,
    graph: proof.graph,
  };
}

function reportProof(options, result) {
  if (typeof options.onIteration !== 'function') return;
  try {
    options.onIteration(result);
  } catch {
    // Progress reporting must never affect the selected action.
  }
}

function combineProofWork(result, proof) {
  return {
    ...result,
    nodes: (result.nodes ?? 0) + proof.nodes,
    elapsedMs: (result.elapsedMs ?? 0) + proof.elapsedMs,
    chaosProof: proofTelemetry(proof),
  };
}

/**
 * Runs a sound loopy-game proof before ordinary bounded Chaos search. Exact
 * proof results are returned directly. Otherwise a heuristic move is replaced
 * only when the optimistic proof still classifies that action as losing.
 */
export function chooseMoveWithChaosProof(position, options = {}) {
  if (position?.chaosMode !== true
      || !Array.isArray(position.board)
      || !Array.isArray(position.board[0])) {
    return chooseMove(position, options);
  }
  const columns = position.board[0].length;
  if (columns === 0
      || position.board.some((row) => !Array.isArray(row) || row.length !== columns)
      || (position.currentPlayer !== RED && position.currentPlayer !== YELLOW)) {
    return chooseMove(position, options);
  }

  const difficulty = options.difficulty ?? position.difficulty ?? 'medium';
  const aiPlayer = options.aiPlayer ?? position.currentPlayer;
  const configuration = chaosProofConfiguration(position, options, difficulty);
  if (!configuration
      || aiPlayer !== position.currentPlayer
      || options.perfectChaosPolicy
      || difficulty === 'perfect'
      // Larger boards go to the ordinary search rather than failing the move.
      || position.board.length * columns > CHAOS_CELL_LIMIT
      || !repetitionHistoryIsFresh(position.repetitionCounts, position.board)
      || positionAlreadyTerminal(position)) {
    return chooseMove(position, options);
  }

  let searched = null;
  if (exactChaosEndgame(position, options, aiPlayer).eligible) {
    searched = chooseMove(position, options);
    // Terminal positions and certified policies went to chooseMove above,
    // so here only the exact graph answers without a proof.
    if (searched.solver === 'chaos-exact-graph') return searched;
  }

  let proof;
  try {
    proof = solveChaosProofPosition(position, configuration);
  } catch (error) {
    if (error?.code !== 'CHAOS_PROOF_GRAPH_LIMIT') throw error;
    const fallback = searched ?? chooseMove(position, options);
    return {
      ...fallback,
      chaosProof: {
        dropDepth: configuration.dropDepth,
        stateLimit: configuration.maximumStates,
        aborted: 'state-limit',
        states: error.states,
      },
    };
  }

  if (proof.solved) {
    const result = {
      ...proof,
      nodes: proof.nodes + (searched?.nodes ?? 0),
      elapsedMs: proof.elapsedMs + (searched?.elapsedMs ?? 0),
      tableHits: searched?.tableHits ?? 0,
      cutoffs: searched?.cutoffs ?? 0,
      tableResets: searched?.tableResets ?? 0,
      chaosProof: proofTelemetry(proof),
    };
    reportProof(options, result);
    return result;
  }

  searched ??= chooseMove(position, options);
  if (!searched.action) return combineProofWork(searched, proof);
  const searchedBound = proof.actionBounds.find((entry) => (
    sameAction(entry.action, searched.action)
  ));
  if (searchedBound?.upper !== CHAOS_LOSS
      || !proof.action
      || sameAction(proof.action, searched.action)) {
    return combineProofWork(searched, proof);
  }

  const result = combineProofWork({
    ...searched,
    action: { ...proof.action },
    score: 0,
    solved: false,
    solver: 'chaos-search+bounded-proof',
    principalVariation: [{ ...proof.action }],
  }, proof);
  reportProof(options, result);
  return result;
}

/** Every prepared route in order: a verified classic policy, the certified
 * Chaos policy, then the bounded proof or the search. */
export function choosePreparedAction(position, options = {}) {
  // The verified policies and the bounded proof answer without going
  // through chooseMove, so the immediate win is preferred here too: a
  // player who can win now should never play on instead.
  return preferImmediateWin(position, choosePerfectClassicMove(position, options)
    ?? choosePerfectChaosMove(position, options)
    ?? chooseMoveWithChaosProof(position, options));
}

async function loadConfiguredPerfectChaosPolicy(position, aiPlayer, options) {
  const rows = position?.board?.length ?? 0;
  const columns = position?.board?.[0]?.length ?? 0;
  const role = perfectChaosCompleteRole(position?.startingPlayer, aiPlayer);
  if (role === null) return null;
  try {
    const entry = authorizeChaosPolicy(options.authorizedChaosPolicy, rows, columns, position.connect, role);
    return await loadVerifiedPerfectChaosCompletePolicy(
      rows,
      columns,
      position.connect,
      role,
      { ...options, manifest: { policies: [entry] }, bytes: options.policyBytes },
    );
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`Could not load the verified complete Chaos policy: ${detail}`);
  }
}

/** Whether a classic 6x7 move loads the opening book. Past the book's last
 * ply no position the search reaches can be in it - every node has at least
 * the root's pieces - so waiting for it there, even through a stalled
 * download, only delayed a move it could not help. */
export function usesOpeningBook(position, difficulty, options) {
  return BOOK_DIFFICULTIES.has(difficulty) && options?.maximumDepth === undefined
    && boardPieceCount(position.board) <= PERFECT_BOOK_CERTIFICATE.maxPly;
}

/** Whether a Perfect 6x7 move loads the 4.7 MB strategy. It stores only
 * positions with more than PERFECT_STRATEGY_HANDOFF empty cells; at or below
 * that bitboard.js solves the move exactly with no strategy, and a stalled
 * download there, after a worker restart, failed a move the exact solver
 * answers alone. */
export function usesPerfectStrategy(position) {
  const { board } = position;
  return board.length * board[0].length - boardPieceCount(board) > PERFECT_STRATEGY_HANDOFF;
}

async function exactDataFor(position, options) {
  const difficulty = options?.difficulty ?? position?.difficulty ?? 'medium';
  if (isBitboardPosition(position)) {
    if (difficulty === 'perfect') {
      return {
        perfectBook: null,
        perfectStrategy: usesPerfectStrategy(position) ? await loadPerfectStrategy(options) : null,
        perfectClassicPolicy: null,
        perfectChaosPolicy: null,
      };
    }
    return {
      perfectBook: usesOpeningBook(position, difficulty, options) ? await loadPerfectBook(options) : null,
      perfectStrategy: null,
      perfectClassicPolicy: null,
      perfectChaosPolicy: null,
    };
  }

  const aiPlayer = options?.aiPlayer ?? position?.currentPlayer;
  if (difficulty === 'perfect' && isPerfectClassicVariant(position)) {
    return {
      perfectBook: null,
      perfectStrategy: null,
      perfectClassicPolicy: usesPerfectClassicPolicy(position)
        ? await loadConfiguredPerfectClassicPolicy(position, aiPlayer, options)
        : null,
      perfectChaosPolicy: null,
    };
  }

  // Chaos boards with a committed complete solution never fall back to search.
  if (difficulty === 'perfect' && isPerfectChaosVariant(position)) {
    return {
      perfectBook: null,
      perfectStrategy: null,
      perfectClassicPolicy: null,
      perfectChaosPolicy: null,
      perfectChaosCompletePolicy: await loadConfiguredPerfectChaosPolicy(position, aiPlayer, options),
    };
  }

  const identity = chaosPolicyIdentity(position, aiPlayer);
  const useChaosPolicy = standardChaosPosition(position)
    && difficulty === 'brutal'
    && options?.maximumDepth === undefined
    && options?.useChaosPolicy !== false
    && identity !== null;
  return {
    perfectBook: null,
    perfectStrategy: null,
    perfectClassicPolicy: null,
    perfectChaosPolicy: useChaosPolicy
      ? await loadPerfectChaosPolicy(identity.role, identity.pieceCount, options)
      : null,
  };
}

/** Shared preparation for the browser worker and headless matches. */
export async function choosePreparedMove(position, options = {}) {
  const exactData = await exactDataFor(position, options);
  options.signal?.throwIfAborted();
  // Telemetry cannot alter move selection, just like onIteration reporting.
  try { options.onSearchStart?.(); } catch { /* ignore progress observer errors */ }
  return choosePreparedAction(position, { ...options, ...exactData });
}

const workerScope = globalThis.self;
if (workerScope?.addEventListener && workerScope?.postMessage) {
  let activeLoad = null;
  workerScope.addEventListener('message', async (event) => {
    activeLoad?.abort();
    const controller = new AbortController();
    activeLoad = controller;
    const { requestId, position, options, policyBytes } = event.data ?? {};
    try {
      workerScope.postMessage({ requestId, kind: 'phase', phase: 'loading' });
      let reported = 0;
      const result = await choosePreparedMove(position, {
        ...options,
        policyBytes,
        signal: controller.signal,
        // The page's loading watchdog is bounded by silence: each report that
        // table bytes are still arriving re-arms it and shows the progress.
        onDataProgress(loaded, total) {
          const now = Date.now();
          if (now - reported < 250 && loaded !== total) return;
          reported = now;
          workerScope.postMessage({ requestId, kind: 'phase', phase: 'loading', loaded, total });
        },
        onSearchStart() {
          workerScope.postMessage({ requestId, kind: 'phase', phase: 'searching' });
        },
        onIteration(progress) {
          workerScope.postMessage({ requestId, kind: 'progress', progress });
        },
      });
      if (!controller.signal.aborted) workerScope.postMessage({ requestId, kind: 'result', result });
    } catch (error) {
      if (controller.signal.aborted) return;
      workerScope.postMessage({
        requestId,
        kind: 'error',
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });
}
