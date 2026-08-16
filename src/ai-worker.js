import { chooseMove } from './ai.js';
import { isBitboardPosition } from './bitboard.js';
import { solveChaosProofPosition } from './chaos-proof.js';
import { CHAOS_LOSS } from './chaos-solver.js';
import { perfectClassicRole } from './perfect-classic-policy.js';
import { loadVerifiedPerfectClassicPolicy } from './perfect-classic-verified.js';
import {
  choosePerfectClassicMove,
  isPerfectClassicVariant,
} from './perfect-classic-runtime.js';
import {
  EMPTY,
  RED,
  YELLOW,
  createBoard,
  hasWinFrom,
  positionKey,
  sameAction,
} from './engine.js';

const BOOK_DIFFICULTIES = new Set(['medium', 'hard', 'brutal']);
const CHAOS_EXACT_EMPTY_THRESHOLD = 6;
const CHAOS_PROOF_DEFAULTS = Object.freeze({
  medium: { dropDepth: 1, maximumStates: 10_000 },
  hard: { dropDepth: 2, maximumStates: 50_000 },
  brutal: { dropDepth: 2, maximumStates: 100_000 },
});

async function loadPerfectStrategy() {
  const { loadPerfectStrategy: load } = await import('./perfect-strategy.js');
  return load();
}

async function loadPerfectBook() {
  try {
    const { loadPerfectBook: load } = await import('./perfect-book.js');
    return await load();
  } catch {
    return null;
  }
}

async function loadConfiguredPerfectClassicPolicy(position, aiPlayer) {
  const rows = position?.board?.length ?? 0;
  const columns = position?.board?.[0]?.length ?? 0;
  const role = perfectClassicRole(position?.startingPlayer, aiPlayer);
  if (role === null) return null;
  try {
    return await loadVerifiedPerfectClassicPolicy(rows, columns, position.connect, role);
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

async function loadPerfectChaosPolicy(role, pieceCount) {
  try {
    const { loadPerfectChaosPolicy: load } = await import('./perfect-chaos-prefix.js');
    return requireCertifiedChaosPolicy(await load(role, pieceCount));
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

function repetitionEntries(entries) {
  if (entries === undefined || entries === null) return [];
  if (entries instanceof Map) return [...entries];
  if (Array.isArray(entries)) return entries;
  if (typeof entries === 'object') return Object.entries(entries);
  return null;
}

function positiveRepetitionKeys(entries) {
  const pairs = repetitionEntries(entries);
  if (!pairs) return new Set();
  return new Set(
    pairs
      .filter((entry) => Array.isArray(entry) && Number.isInteger(entry[1]) && entry[1] > 0)
      .map((entry) => entry[0]),
  );
}

function repetitionHistoryIsFresh(entries) {
  const pairs = repetitionEntries(entries);
  if (!pairs) return false;
  return pairs.every((entry) => (
    Array.isArray(entry)
    && Number.isInteger(entry[1])
    && entry[1] >= 0
    && entry[1] <= 1
  ));
}

function certifiedStartingPlayer(position) {
  const keys = positiveRepetitionKeys(position?.repetitionCounts);
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

function emptyCellCount(board) {
  let count = 0;
  for (const row of board) {
    for (const cell of row) if (cell === EMPTY) count += 1;
  }
  return count;
}

function boardHasWin(board, player, connect) {
  for (let row = 0; row < board.length; row += 1) {
    for (let column = 0; column < board[row].length; column += 1) {
      if (board[row][column] === player
          && hasWinFrom(board, row, column, player, connect)) return true;
    }
  }
  return false;
}

function positionAlreadyTerminal(position) {
  if (!Array.isArray(position?.board) || !Number.isInteger(position?.connect)) return true;
  return position.board.every((row) => row.every((cell) => cell !== EMPTY))
    || boardHasWin(position.board, RED, position.connect)
    || boardHasWin(position.board, YELLOW, position.connect);
}

function exactChaosCandidate(position, options, aiPlayer) {
  if (aiPlayer !== position.currentPlayer || !repetitionHistoryIsFresh(position.repetitionCounts)) {
    return false;
  }
  const boardCells = position.board.length * position.board[0].length;
  const threshold = options.chaosExactEmptyThreshold ?? CHAOS_EXACT_EMPTY_THRESHOLD;
  if (!Number.isInteger(threshold) || threshold < 0 || threshold > boardCells) return true;
  const maximumStates = options.chaosMaximumStates;
  if (maximumStates !== undefined
      && (!Number.isInteger(maximumStates) || maximumStates < 1 || maximumStates > 2_000_000)) {
    return true;
  }
  return emptyCellCount(position.board) <= threshold;
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
      || !repetitionHistoryIsFresh(position.repetitionCounts)
      || positionAlreadyTerminal(position)) {
    return chooseMove(position, options);
  }

  let searched = null;
  if (exactChaosCandidate(position, options, aiPlayer)) {
    searched = chooseMove(position, options);
    if (searched.solver === 'chaos-exact-graph'
        || searched.solver === 'terminal'
        || searched.solver === 'chaos-certified-prefix') {
      return searched;
    }
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
    chaosProofOverride: true,
    searchedAction: { ...searched.action },
    searchedScore: searched.score,
  }, proof);
  reportProof(options, result);
  return result;
}

export function chooseMoveWithPerfectClassic(position, options = {}) {
  return choosePerfectClassicMove(position, options)
    ?? chooseMoveWithChaosProof(position, options);
}

async function exactDataFor(position, options) {
  const difficulty = options?.difficulty ?? position?.difficulty ?? 'medium';
  if (isBitboardPosition(position)) {
    if (difficulty === 'perfect') {
      return {
        perfectBook: null,
        perfectStrategy: await loadPerfectStrategy(),
        perfectClassicPolicy: null,
        perfectChaosPolicy: null,
      };
    }
    const useBook = BOOK_DIFFICULTIES.has(difficulty)
      && options?.maximumDepth === undefined
      && options?.useBook !== false;
    return {
      perfectBook: useBook ? await loadPerfectBook() : null,
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
      perfectClassicPolicy: await loadConfiguredPerfectClassicPolicy(position, aiPlayer),
      perfectChaosPolicy: null,
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
      ? await loadPerfectChaosPolicy(identity.role, identity.pieceCount)
      : null,
  };
}

const workerScope = globalThis.self;
if (workerScope?.addEventListener && workerScope?.postMessage) {
  workerScope.addEventListener('message', async (event) => {
    const { requestId, position, options } = event.data ?? {};
    try {
      const exactData = await exactDataFor(position, options);
      const result = chooseMoveWithPerfectClassic(position, {
        ...options,
        ...exactData,
        onIteration(progress) {
          workerScope.postMessage({ requestId, kind: 'progress', progress });
        },
      });
      workerScope.postMessage({ requestId, kind: 'result', result });
    } catch (error) {
      workerScope.postMessage({
        requestId,
        kind: 'error',
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });
}
