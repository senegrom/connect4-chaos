import { ACTION_DROP, EMPTY } from './engine.js';

const WIDTH = 7;
const HEIGHT = 6;
const STRIDE = HEIGHT + 1;
const CELL_COUNT = WIDTH * HEIGHT;
const MATE_SCORE = 1_000_000;
const HEURISTIC_LIMIT = 100_000;
const COLUMN_BITS = (1n << BigInt(HEIGHT)) - 1n;
const COLUMN_WITH_SENTINEL = (1n << BigInt(STRIDE)) - 1n;
const BOTTOM_MASKS = Array.from(
  { length: WIDTH },
  (_, column) => 1n << BigInt(column * STRIDE),
);
const COLUMN_MASKS = BOTTOM_MASKS.map((bottom) => bottom * COLUMN_BITS);
const BOTTOM_MASK = BOTTOM_MASKS.reduce((mask, bit) => mask | bit, 0n);
const BOARD_MASK = BOTTOM_MASK * COLUMN_BITS;
const CENTRE_MASK = COLUMN_MASKS[Math.floor(WIDTH / 2)];
const COLUMN_ORDER = Object.freeze([3, 2, 4, 1, 5, 0, 6]);

const DIFFICULTY = Object.freeze({
  medium: { depth: 10, exactThreshold: 16, tableBits: 16 },
  hard: { depth: 14, exactThreshold: 20, tableBits: 18 },
  brutal: { depth: 16, exactThreshold: 24, tableBits: 20 },
});

function now() {
  return globalThis.performance?.now?.() ?? Date.now();
}

function popcount(value) {
  let count = 0;
  while (value !== 0n) {
    value &= value - 1n;
    count += 1;
  }
  return count;
}

function possibleMoves(mask) {
  return (mask + BOTTOM_MASK) & BOARD_MASK;
}

function moveForColumn(mask, column) {
  return (mask + BOTTOM_MASKS[column]) & COLUMN_MASKS[column];
}

function play(position, move) {
  return {
    current: position.current ^ position.mask,
    mask: position.mask | move,
    moves: position.moves + 1,
  };
}

function hasAlignment(bits) {
  for (const direction of [1, HEIGHT, STRIDE, HEIGHT + 2]) {
    const shift = BigInt(direction);
    const pair = bits & (bits >> shift);
    if ((pair & (pair >> (2n * shift))) !== 0n) return true;
  }
  return false;
}

export function winningPosition(position, mask) {
  let result = (position << 1n) & (position << 2n) & (position << 3n);
  for (const direction of [HEIGHT, STRIDE, HEIGHT + 2]) {
    const shift = BigInt(direction);
    let pair = (position << shift) & (position << (2n * shift));
    result |= pair & (position << (3n * shift));
    result |= pair & (position >> shift);
    pair = (position >> shift) & (position >> (2n * shift));
    result |= pair & (position << shift);
    result |= pair & (position >> (3n * shift));
  }
  return result & (BOARD_MASK ^ mask);
}

export function possibleNonLosingMoves(position) {
  let possible = possibleMoves(position.mask);
  const opponentWinning = winningPosition(position.current ^ position.mask, position.mask);
  const forced = possible & opponentWinning;
  if (forced !== 0n) {
    if ((forced & (forced - 1n)) !== 0n) return 0n;
    possible = forced;
  }
  return possible & ~(opponentWinning >> 1n);
}

function mirrorBits(bits) {
  let mirrored = 0n;
  for (let column = 0; column < WIDTH; column += 1) {
    const columnBits = (bits >> BigInt(column * STRIDE)) & COLUMN_WITH_SENTINEL;
    mirrored |= columnBits << BigInt((WIDTH - 1 - column) * STRIDE);
  }
  return mirrored;
}

function canonicalPosition(position) {
  const normal = position.current + position.mask;
  const mirrored = mirrorBits(position.current) + mirrorBits(position.mask);
  return normal <= mirrored
    ? { key: normal, mirrored: false }
    : { key: mirrored, mirrored: true };
}

export function boardToBitboard(board, currentPlayer) {
  if (!Array.isArray(board)
      || board.length !== HEIGHT
      || board.some((row) => !Array.isArray(row) || row.length !== WIDTH)) {
    return null;
  }

  let current = 0n;
  let mask = 0n;
  let moves = 0;
  for (let column = 0; column < WIDTH; column += 1) {
    let emptyBelow = false;
    for (let row = HEIGHT - 1; row >= 0; row -= 1) {
      const cell = board[row][column];
      if (cell === EMPTY) {
        emptyBelow = true;
        continue;
      }
      if (emptyBelow || (cell !== 1 && cell !== 2)) return null;
      const bit = 1n << BigInt(column * STRIDE + HEIGHT - 1 - row);
      mask |= bit;
      if (cell === currentPlayer) current |= bit;
      moves += 1;
    }
  }
  return { current, mask, moves };
}

export function isBitboardPosition(position) {
  return Boolean(
    position
      && !position.chaosMode
      && position.connect === 4
      && boardToBitboard(position.board, position.currentPlayer),
  );
}

function evaluate(position) {
  const opponent = position.current ^ position.mask;
  const possible = possibleMoves(position.mask);
  const ownWinning = winningPosition(position.current, position.mask);
  const opponentWinning = winningPosition(opponent, position.mask);
  const ownPlayable = popcount(ownWinning & possible);
  const opponentPlayable = popcount(opponentWinning & possible);

  let score = (popcount(ownWinning) - popcount(opponentWinning)) * 24;
  score += ownPlayable * 420 - opponentPlayable * 520;
  if (ownPlayable >= 2) score += 2_400;
  if (opponentPlayable >= 2) score -= 2_900;
  score += (
    popcount(position.current & CENTRE_MASK)
    - popcount(opponent & CENTRE_MASK)
  ) * 10;
  return Math.max(-HEURISTIC_LIMIT, Math.min(HEURISTIC_LIMIT, score));
}

function orderedMoves(position, possible, preferredColumn = -1) {
  const moves = [];
  for (const column of COLUMN_ORDER) {
    const move = moveForColumn(position.mask, column);
    if ((possible & move) === 0n) continue;
    const potential = winningPosition(position.current | move, position.mask);
    moves.push({
      column,
      move,
      order: popcount(potential) * 64
        + (7 - Math.abs(3 - column))
        + (column === preferredColumn ? 1_000_000 : 0),
    });
  }
  moves.sort((first, second) => second.order - first.order);
  return moves;
}

class TranspositionTable {
  constructor(bits) {
    this.size = 1 << bits;
    this.indexMask = BigInt(this.size - 1);
    this.keys = new BigUint64Array(this.size);
    this.scores = new Int32Array(this.size);
    this.depths = new Uint8Array(this.size);
    this.flags = new Uint8Array(this.size);
    this.columns = new Uint8Array(this.size);
    this.stores = 0;
    this.collisions = 0;
  }

  index(key) {
    return Number((key ^ (key >> 23n) ^ (key >> 41n)) & this.indexMask);
  }

  probe(key) {
    const index = this.index(key);
    if (this.keys[index] !== key + 1n) return null;
    return {
      score: this.scores[index],
      depth: this.depths[index],
      flag: this.flags[index],
      column: this.columns[index] - 1,
    };
  }

  store(key, depth, score, flag, column) {
    const index = this.index(key);
    const storedKey = this.keys[index];
    if (storedKey !== 0n && storedKey !== key + 1n) this.collisions += 1;
    if (storedKey === key + 1n && this.depths[index] > depth) return;
    this.keys[index] = key + 1n;
    this.scores[index] = score;
    this.depths[index] = depth;
    this.flags[index] = flag;
    this.columns[index] = column + 1;
    this.stores += 1;
  }
}

class BitboardSearch {
  constructor(tableBits) {
    this.table = new TranspositionTable(tableBits);
    this.nodes = 0;
    this.tableHits = 0;
    this.cutoffs = 0;
  }

  search(position, depth, alpha, beta) {
    this.nodes += 1;
    const possible = possibleMoves(position.mask);
    if (possible === 0n) return 0;

    const winning = winningPosition(position.current, position.mask) & possible;
    if (winning !== 0n) return MATE_SCORE - position.moves;

    const safe = possibleNonLosingMoves(position);
    if (safe === 0n) return -MATE_SCORE + position.moves;

    const opponentWinning = winningPosition(
      position.current ^ position.mask,
      position.mask,
    ) & possible;
    if (depth <= 0 && opponentWinning === 0n) return evaluate(position);
    const effectiveDepth = Math.max(1, depth);

    const alphaOriginal = alpha;
    const betaOriginal = beta;
    const canonical = canonicalPosition(position);
    const cached = this.table.probe(canonical.key);
    let preferredColumn = -1;
    if (cached) {
      this.tableHits += 1;
      preferredColumn = canonical.mirrored ? WIDTH - 1 - cached.column : cached.column;
      if (cached.depth >= effectiveDepth) {
        if (cached.flag === 0) return cached.score;
        if (cached.flag === 1) alpha = Math.max(alpha, cached.score);
        else beta = Math.min(beta, cached.score);
        if (alpha >= beta) return cached.score;
      }
    }

    let bestScore = -Infinity;
    let bestColumn = preferredColumn >= 0 ? preferredColumn : 3;
    for (const candidate of orderedMoves(position, safe, preferredColumn)) {
      const score = -this.search(
        play(position, candidate.move),
        effectiveDepth - 1,
        -beta,
        -alpha,
      );
      if (score > bestScore) {
        bestScore = score;
        bestColumn = candidate.column;
      }
      alpha = Math.max(alpha, score);
      if (alpha >= beta) {
        this.cutoffs += 1;
        break;
      }
    }

    let flag = 0;
    if (bestScore <= alphaOriginal) flag = 2;
    else if (bestScore >= betaOriginal) flag = 1;
    const storedColumn = canonical.mirrored ? WIDTH - 1 - bestColumn : bestColumn;
    this.table.store(canonical.key, effectiveDepth, bestScore, flag, storedColumn);
    return bestScore;
  }

  root(position, depth, preferredColumn = -1) {
    const possible = possibleMoves(position.mask);
    if (possible === 0n) return { column: -1, score: 0 };

    const winning = winningPosition(position.current, position.mask) & possible;
    if (winning !== 0n) {
      for (const column of COLUMN_ORDER) {
        if ((winning & moveForColumn(position.mask, column)) !== 0n) {
          return { column, score: MATE_SCORE - position.moves };
        }
      }
    }

    let safe = possibleNonLosingMoves(position);
    if (safe === 0n) safe = possible;
    let alpha = -Infinity;
    let bestColumn = -1;
    let bestScore = -Infinity;
    for (const candidate of orderedMoves(position, safe, preferredColumn)) {
      const score = -this.search(
        play(position, candidate.move),
        depth - 1,
        -Infinity,
        -alpha,
      );
      if (score > bestScore) {
        bestScore = score;
        bestColumn = candidate.column;
      }
      alpha = Math.max(alpha, score);
    }
    return { column: bestColumn, score: bestScore };
  }
}

function principalVariation(position, firstColumn, depth, search) {
  if (firstColumn < 0) return [];
  const variation = [];
  let current = position;
  let column = firstColumn;
  for (let ply = 0; ply < depth && column >= 0; ply += 1) {
    const move = moveForColumn(current.mask, column);
    if (move === 0n) break;
    variation.push({ type: ACTION_DROP, column });
    current = play(current, move);
    if (hasAlignment(current.current ^ current.mask)) break;
    const possible = possibleMoves(current.mask);
    if (possible === 0n
        || (winningPosition(current.current, current.mask) & possible) !== 0n) break;
    const canonical = canonicalPosition(current);
    const cached = search.table.probe(canonical.key);
    if (!cached) break;
    column = canonical.mirrored ? WIDTH - 1 - cached.column : cached.column;
  }
  return variation;
}

export function chooseBitboardMove(position, options = {}) {
  const bitboard = boardToBitboard(position?.board, position?.currentPlayer);
  if (!bitboard || position.chaosMode || position.connect !== 4) return null;

  const difficulty = options.difficulty ?? position.difficulty ?? 'medium';
  const config = DIFFICULTY[difficulty] ?? DIFFICULTY.medium;
  const remaining = CELL_COUNT - bitboard.moves;
  const exactThreshold = options.exactThreshold ?? config.exactThreshold;
  const requestedDepth = options.maximumDepth
    ?? (remaining <= exactThreshold ? remaining : config.depth);
  const targetDepth = Math.max(1, Math.min(remaining, requestedDepth));
  const search = new BitboardSearch(options.tableBits ?? config.tableBits);
  const start = now();

  let completedDepth = 0;
  let best = { column: -1, score: evaluate(bitboard) };
  for (let depth = 1; depth <= targetDepth; depth += 1) {
    best = search.root(bitboard, depth, best.column);
    completedDepth = depth;
    const proven = Math.abs(best.score) >= MATE_SCORE - CELL_COUNT;
    const progress = {
      action: best.column < 0 ? null : { type: ACTION_DROP, column: best.column },
      score: position.currentPlayer === (options.aiPlayer ?? position.currentPlayer)
        ? best.score
        : -best.score,
      depth,
      nodes: search.nodes,
      elapsedMs: now() - start,
      tableHits: search.tableHits,
      cutoffs: search.cutoffs,
      tableStores: search.table.stores,
      tableCollisions: search.table.collisions,
      solved: depth >= remaining || proven,
      solver: 'bitboard',
    };
    if (typeof options.onIteration === 'function') options.onIteration(progress);
    if (proven) break;
  }

  const action = best.column < 0 ? null : { type: ACTION_DROP, column: best.column };
  const aiScore = position.currentPlayer === (options.aiPlayer ?? position.currentPlayer)
    ? best.score
    : -best.score;
  const solved = completedDepth >= remaining
    || Math.abs(best.score) >= MATE_SCORE - CELL_COUNT;
  return {
    action,
    score: aiScore,
    depth: completedDepth,
    nodes: search.nodes,
    elapsedMs: now() - start,
    tableHits: search.tableHits,
    cutoffs: search.cutoffs,
    tableResets: 0,
    tableStores: search.table.stores,
    tableCollisions: search.table.collisions,
    principalVariation: principalVariation(bitboard, best.column, completedDepth, search),
    solved,
    solver: 'bitboard',
  };
}

export const BITBOARD_DIMENSIONS = Object.freeze({ rows: HEIGHT, cols: WIDTH, connect: 4 });
