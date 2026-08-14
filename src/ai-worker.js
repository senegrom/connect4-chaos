import { chooseMove } from './ai.js';
import { isBitboardPosition } from './bitboard.js';
import { RED, YELLOW, createBoard, positionKey } from './engine.js';

const BOOK_DIFFICULTIES = new Set(['medium', 'hard', 'brutal']);

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

function positiveRepetitionKeys(entries) {
  if (entries instanceof Map) return new Set(
    [...entries].filter(([, count]) => Number.isInteger(count) && count > 0).map(([key]) => key),
  );
  if (Array.isArray(entries)) return new Set(
    entries
      .filter((entry) => Array.isArray(entry) && Number.isInteger(entry[1]) && entry[1] > 0)
      .map((entry) => entry[0]),
  );
  if (entries && typeof entries === 'object') return new Set(
    Object.entries(entries).filter(([, count]) => Number.isInteger(count) && count > 0)
      .map(([key]) => key),
  );
  return new Set();
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
      else if (cell !== 0) return null;
    }
  }
  return {
    role: aiPlayer === startingPlayer ? 1 : 2,
    pieceCount,
  };
}

async function exactDataFor(position, options) {
  const difficulty = options?.difficulty ?? position?.difficulty ?? 'medium';
  if (isBitboardPosition(position)) {
    if (difficulty === 'perfect') {
      return {
        perfectBook: null,
        perfectStrategy: await loadPerfectStrategy(),
        perfectChaosPolicy: null,
      };
    }
    const useBook = BOOK_DIFFICULTIES.has(difficulty)
      && options?.maximumDepth === undefined
      && options?.useBook !== false;
    return {
      perfectBook: useBook ? await loadPerfectBook() : null,
      perfectStrategy: null,
      perfectChaosPolicy: null,
    };
  }

  const aiPlayer = options?.aiPlayer ?? position?.currentPlayer;
  const identity = chaosPolicyIdentity(position, aiPlayer);
  const useChaosPolicy = standardChaosPosition(position)
    && difficulty === 'brutal'
    && options?.maximumDepth === undefined
    && options?.useChaosPolicy !== false
    && identity !== null;
  return {
    perfectBook: null,
    perfectStrategy: null,
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
      const result = chooseMove(position, {
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
