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

function reportProgress(callback, progress) {
  if (typeof callback !== 'function') return;
  try {
    callback(progress);
  } catch {
    // Search telemetry must never affect the selected move.
  }
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

function mirrorMoveMask(mask) {
  let mirrored = 0;
  for (let column = 0; column < WIDTH; column += 1) {
    if ((mask & (1 << column)) !== 0) mirrored |= 1 << (WIDTH - 1 - column);
  }
  return mirrored;
}

function columnFromMoveMask(position, moveMask) {
  const possible = possibleMoves(position.mask);
  for (const column of COLUMN_ORDER) {
    const move = moveForColumn(position.mask, column);
    if ((moveMask & (1 << column)) !== 0 && (possible & move) !== 0n) return column;
  }
  return -1;
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

class ExactOutcomeTable {
  constructor(bits) {
    this.size = 1 << bits;
    this.indexMask = BigInt(this.size - 1);
    this.keys = new BigUint64Array(this.size);
    this.lowerBounds = new Int8Array(this.size);
    this.upperBounds = new Int8Array(this.size);
    this.flags = new Uint8Array(this.size);
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
      lower: (this.flags[index] & 1) === 0 ? -2 : this.lowerBounds[index],
      upper: (this.flags[index] & 2) === 0 ? 2 : this.upperBounds[index],
    };
  }

  prepare(key) {
    const index = this.index(key);
    const storedKey = this.keys[index];
    if (storedKey !== 0n && storedKey !== key + 1n) this.collisions += 1;
    if (storedKey !== key + 1n) {
      this.keys[index] = key + 1n;
      this.flags[index] = 0;
    }
    return index;
  }

  storeLower(key, score) {
    const index = this.prepare(key);
    if ((this.flags[index] & 1) !== 0 && score <= this.lowerBounds[index]) return;
    this.lowerBounds[index] = score;
    this.flags[index] |= 1;
    this.stores += 1;
  }

  storeUpper(key, score) {
    const index = this.prepare(key);
    if ((this.flags[index] & 2) !== 0 && score >= this.upperBounds[index]) return;
    this.upperBounds[index] = score;
    this.flags[index] |= 2;
    this.stores += 1;
  }
}

class ExactOutcomeSearch {
  constructor(tableBits) {
    this.table = new ExactOutcomeTable(tableBits);
    this.nodes = 0;
    this.tableHits = 0;
    this.cutoffs = 0;
    this.onProgress = null;
    this.nextProgressNode = 65_536;
  }

  visitNode() {
    this.nodes += 1;
    if (this.onProgress && this.nodes >= this.nextProgressNode) {
      this.nextProgressNode += 65_536;
      this.onProgress(this);
    }
  }

  search(position, alpha, beta) {
    this.visitNode();
    const possible = possibleMoves(position.mask);
    if (possible === 0n) return 0;
    if ((winningPosition(position.current, position.mask) & possible) !== 0n) return 1;

    const nonLosing = possibleNonLosingMoves(position);
    if (nonLosing === 0n) return -1;
    if (position.moves >= CELL_COUNT - 2) return 0;

    const canonical = canonicalPosition(position);
    const cached = this.table.probe(canonical.key);
    if (cached) {
      this.tableHits += 1;
      if (cached.lower >= beta) return cached.lower;
      if (cached.upper <= alpha) return cached.upper;
      alpha = Math.max(alpha, cached.lower);
      beta = Math.min(beta, cached.upper);
      if (alpha >= beta) return alpha;
    }

    for (const candidate of orderedMoves(position, nonLosing)) {
      const score = -this.search(play(position, candidate.move), -beta, -alpha);
      if (score >= beta) {
        this.cutoffs += 1;
        this.table.storeLower(canonical.key, score);
        return score;
      }
      if (score > alpha) alpha = score;
    }

    this.table.storeUpper(canonical.key, alpha);
    return alpha;
  }

  solve(position) {
    const possible = possibleMoves(position.mask);
    if (possible === 0n) return 0;
    if ((winningPosition(position.current, position.mask) & possible) !== 0n) return 1;

    let minimum = -1;
    let maximum = 1;
    while (minimum < maximum) {
      const middle = minimum + Math.floor((maximum - minimum) / 2);
      const score = this.search(position, middle, middle + 1);
      if (score <= middle) maximum = score;
      else minimum = score;
    }
    return minimum;
  }

  root(position) {
    const possible = possibleMoves(position.mask);
    if (possible === 0n) return { column: -1, outcome: 0 };

    const winning = winningPosition(position.current, position.mask) & possible;
    if (winning !== 0n) {
      for (const column of COLUMN_ORDER) {
        if ((winning & moveForColumn(position.mask, column)) !== 0n) {
          return { column, outcome: 1 };
        }
      }
    }

    const nonLosing = possibleNonLosingMoves(position);
    if (nonLosing === 0n) {
      for (const column of COLUMN_ORDER) {
        if ((possible & moveForColumn(position.mask, column)) !== 0n) {
          return { column, outcome: -1 };
        }
      }
    }

    let bestColumn = -1;
    let bestOutcome = -2;
    for (const candidate of orderedMoves(position, nonLosing)) {
      const outcome = -this.solve(play(position, candidate.move));
      if (outcome > bestOutcome) {
        bestOutcome = outcome;
        bestColumn = candidate.column;
      }
      if (bestOutcome === 1) break;
    }
    return { column: bestColumn, outcome: bestOutcome };
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

function chooseExactOutcomeMove(bitboard, options, config, currentPlayer, aiPlayer, start) {
  const remaining = CELL_COUNT - bitboard.moves;
  reportProgress(options.onIteration, {
    action: null,
    score: 0,
    depth: remaining,
    nodes: 0,
    elapsedMs: 0,
    tableHits: 0,
    cutoffs: 0,
    tableResets: 0,
    tableStores: 0,
    tableCollisions: 0,
    principalVariation: [],
    solved: false,
    solver: 'bitboard-exact',
  });

  const search = new ExactOutcomeSearch(options.exactTableBits ?? config.tableBits);
  search.onProgress = () => reportProgress(options.onIteration, {
    action: null,
    score: 0,
    depth: remaining,
    nodes: search.nodes,
    elapsedMs: now() - start,
    tableHits: search.tableHits,
    cutoffs: search.cutoffs,
    tableResets: 0,
    tableStores: search.table.stores,
    tableCollisions: search.table.collisions,
    principalVariation: [],
    solved: false,
    solver: 'bitboard-exact',
  });
  const result = search.root(bitboard);
  const action = result.column < 0 ? null : { type: ACTION_DROP, column: result.column };
  const aiOutcome = currentPlayer === aiPlayer ? result.outcome : -result.outcome;
  const progress = {
    action,
    score: aiOutcome === 0 ? 0 : aiOutcome * (MATE_SCORE - bitboard.moves),
    depth: remaining,
    nodes: search.nodes,
    elapsedMs: now() - start,
    tableHits: search.tableHits,
    cutoffs: search.cutoffs,
    tableResets: 0,
    tableStores: search.table.stores,
    tableCollisions: search.table.collisions,
    principalVariation: action ? [{ ...action }] : [],
    solved: true,
    solver: 'bitboard-exact',
  };
  reportProgress(options.onIteration, progress);
  return progress;
}

export function chooseBitboardMove(position, options = {}) {
  const bitboard = boardToBitboard(position?.board, position?.currentPlayer);
  if (!bitboard || position.chaosMode || position.connect !== 4) return null;

  const difficulty = options.difficulty ?? position.difficulty ?? 'medium';
  const config = DIFFICULTY[difficulty] ?? DIFFICULTY.medium;
  const aiPlayer = options.aiPlayer ?? position.currentPlayer;
  const remaining = CELL_COUNT - bitboard.moves;
  const exactThreshold = options.exactThreshold ?? config.exactThreshold;
  const start = now();

  const perfectBook = options.perfectBook;
  if (options.useBook !== false
      && options.maximumDepth === undefined
      && perfectBook
      && typeof perfectBook.lookup === 'function') {
    const canonical = canonicalPosition(bitboard);
    const stored = perfectBook.lookup(canonical.key);
    if (stored) {
      const moveMask = canonical.mirrored
        ? mirrorMoveMask(stored.moveMask)
        : stored.moveMask;
      const column = columnFromMoveMask(bitboard, moveMask);
      if (column >= 0) {
        const action = { type: ACTION_DROP, column };
        const relativeOutcome = position.currentPlayer === aiPlayer
          ? stored.outcome
          : -stored.outcome;
        const result = {
          action,
          score: relativeOutcome === 0
            ? 0
            : relativeOutcome * (MATE_SCORE - bitboard.moves),
          depth: 0,
          nodes: 0,
          elapsedMs: now() - start,
          tableHits: 0,
          cutoffs: 0,
          tableResets: 0,
          tableStores: 0,
          tableCollisions: 0,
          principalVariation: [{ ...action }],
          solved: true,
          solver: 'perfect-book',
          bookPly: bitboard.moves,
          bookMaxPly: perfectBook.maxPly ?? null,
          bookEntryCount: perfectBook.entryCount ?? null,
        };
        reportProgress(options.onIteration, result);
        return result;
      }
    }
  }

  if (options.maximumDepth === undefined && remaining <= exactThreshold) {
    return chooseExactOutcomeMove(
      bitboard,
      options,
      config,
      position.currentPlayer,
      aiPlayer,
      start,
    );
  }

  const requestedDepth = options.maximumDepth ?? config.depth;
  const targetDepth = Math.max(1, Math.min(remaining, requestedDepth));
  const search = new BitboardSearch(options.tableBits ?? config.tableBits);
  let completedDepth = 0;
  let best = { column: -1, score: evaluate(bitboard) };
  for (let depth = 1; depth <= targetDepth; depth += 1) {
    best = search.root(bitboard, depth, best.column);
    completedDepth = depth;
    const proven = Math.abs(best.score) >= MATE_SCORE - CELL_COUNT;
    const progress = {
      action: best.column < 0 ? null : { type: ACTION_DROP, column: best.column },
      score: position.currentPlayer === aiPlayer ? best.score : -best.score,
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
    reportProgress(options.onIteration, progress);
    if (proven) break;
  }

  const action = best.column < 0 ? null : { type: ACTION_DROP, column: best.column };
  const aiScore = position.currentPlayer === aiPlayer ? best.score : -best.score;
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
