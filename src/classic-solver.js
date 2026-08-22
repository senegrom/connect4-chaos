import { ACTION_DROP, EMPTY, RED, YELLOW } from './engine.js';

export const CLASSIC_WIN = 1;
export const CLASSIC_DRAW = 0;
export const CLASSIC_LOSS = -1;

const MATE_SCORE = 1_000_000;
const DEFAULT_MAXIMUM_NODES = 50_000_000;
const GEOMETRIES = new Map();

function now() {
  return globalThis.performance?.now?.() ?? Date.now();
}

function reportProgress(callback, progress) {
  if (typeof callback !== 'function') return;
  try {
    callback(progress);
  } catch {
    // Search telemetry must never affect an exact result.
  }
}

function integerOption(value, fallback, minimum, maximum, label) {
  const selected = value ?? fallback;
  if (!Number.isInteger(selected) || selected < minimum || selected > maximum) {
    throw new RangeError(`${label} must be an integer from ${minimum} through ${maximum}.`);
  }
  return selected;
}

function popcount(value) {
  let count = 0;
  while (value !== 0n) {
    value &= value - 1n;
    count += 1;
  }
  return count;
}

function geometryKey(rows, columns, connect) {
  return `${rows}x${columns}:c${connect}`;
}

export function createClassicGeometry(rows, columns, connect = 4) {
  if (!Number.isInteger(rows) || !Number.isInteger(columns)
      || rows < 1 || columns < 1 || rows > 7 || columns > 7) {
    throw new RangeError('Exact classic boards must have 1 through 7 rows and columns.');
  }
  if (!Number.isInteger(connect) || connect < 1 || connect > Math.max(rows, columns)) {
    throw new RangeError('Connect length must be a positive integer that fits the board.');
  }

  const key = geometryKey(rows, columns, connect);
  let geometry = GEOMETRIES.get(key);
  if (geometry) return geometry;

  const stride = rows + 1;
  const cellCount = rows * columns;
  const columnBits = (1n << BigInt(rows)) - 1n;
  const columnWithSentinel = (1n << BigInt(stride)) - 1n;
  const bottomMasks = Array.from(
    { length: columns },
    (_, column) => 1n << BigInt(column * stride),
  );
  const columnMasks = bottomMasks.map((bottom) => bottom * columnBits);
  const bottomMask = bottomMasks.reduce((mask, bit) => mask | bit, 0n);
  const boardMask = bottomMask * columnBits;
  const centre = (columns - 1) / 2;
  const columnOrder = Array.from({ length: columns }, (_, column) => column)
    .sort((first, second) => (
      Math.abs(first - centre) - Math.abs(second - centre) || first - second
    ));

  geometry = Object.freeze({
    rows,
    columns,
    connect,
    stride,
    cellCount,
    columnBits,
    columnWithSentinel,
    bottomMasks: Object.freeze(bottomMasks),
    columnMasks: Object.freeze(columnMasks),
    bottomMask,
    boardMask,
    columnOrder: Object.freeze(columnOrder),
    directions: Object.freeze([1, stride - 1, stride, stride + 1]),
  });
  GEOMETRIES.set(key, geometry);
  return geometry;
}

function possibleMoves(geometry, mask) {
  return (mask + geometry.bottomMask) & geometry.boardMask;
}

function moveForColumn(geometry, mask, column) {
  if (!Number.isInteger(column) || column < 0 || column >= geometry.columns) return 0n;
  return (mask + geometry.bottomMasks[column]) & geometry.columnMasks[column];
}

function play(position, move) {
  return {
    current: position.current ^ position.mask,
    mask: position.mask | move,
    moves: position.moves + 1,
  };
}

export function hasClassicAlignment(geometry, bits) {
  for (const direction of geometry.directions) {
    const shift = BigInt(direction);
    let run = bits;
    for (let offset = 1; offset < geometry.connect && run !== 0n; offset += 1) {
      run &= bits >> (BigInt(offset) * shift);
    }
    if (run !== 0n) return true;
  }
  return false;
}

function mirrorBits(geometry, bits) {
  let mirrored = 0n;
  for (let column = 0; column < geometry.columns; column += 1) {
    const group = (bits >> BigInt(column * geometry.stride))
      & geometry.columnWithSentinel;
    mirrored |= group << BigInt((geometry.columns - 1 - column) * geometry.stride);
  }
  return mirrored;
}

export function canonicalClassicPosition(geometry, position) {
  const normal = position.current + position.mask;
  const mirroredCurrent = mirrorBits(geometry, position.current);
  const mirroredMask = mirrorBits(geometry, position.mask);
  const mirrored = mirroredCurrent + mirroredMask;
  return normal <= mirrored
    ? { key: normal, mirrored: false }
    : { key: mirrored, mirrored: true };
}

function immediateWinningMoves(geometry, position, pieces = position.current) {
  const possible = possibleMoves(geometry, position.mask);
  let winning = 0n;
  for (const column of geometry.columnOrder) {
    const move = moveForColumn(geometry, position.mask, column);
    if ((possible & move) !== 0n && hasClassicAlignment(geometry, pieces | move)) {
      winning |= move;
    }
  }
  return winning;
}

/**
 * Returns exactly the legal moves that do not concede an immediate reply.
 * A single opponent threat is forced; two distinct immediate threats prove a loss.
 */
export function possibleClassicNonLosingMoves(geometry, position) {
  let possible = possibleMoves(geometry, position.mask);
  const opponent = position.current ^ position.mask;
  const opponentWins = immediateWinningMoves(geometry, position, opponent);
  if (opponentWins !== 0n) {
    if ((opponentWins & (opponentWins - 1n)) !== 0n) return 0n;
    possible = opponentWins;
  }

  let safe = 0n;
  for (const column of geometry.columnOrder) {
    const move = moveForColumn(geometry, position.mask, column);
    if ((possible & move) === 0n) continue;
    if (hasClassicAlignment(geometry, position.current | move)) {
      safe |= move;
      continue;
    }
    const child = play(position, move);
    if (immediateWinningMoves(geometry, child) === 0n) safe |= move;
  }
  return safe;
}

export function boardToClassicBitboard(board, currentPlayer, connect = 4) {
  if ((currentPlayer !== RED && currentPlayer !== YELLOW)
      || !Array.isArray(board)
      || board.length === 0
      || !Array.isArray(board[0])
      || board[0].length === 0) return null;

  const rows = board.length;
  const columns = board[0].length;
  if (rows > 7 || columns > 7
      || board.some((row) => !Array.isArray(row) || row.length !== columns)) return null;

  let geometry;
  try {
    geometry = createClassicGeometry(rows, columns, connect);
  } catch {
    return null;
  }

  let current = 0n;
  let mask = 0n;
  let moves = 0;
  for (let column = 0; column < columns; column += 1) {
    let emptyBelow = false;
    for (let row = rows - 1; row >= 0; row -= 1) {
      const cell = board[row][column];
      if (cell === EMPTY) {
        emptyBelow = true;
        continue;
      }
      if (emptyBelow || (cell !== RED && cell !== YELLOW)) return null;
      const bit = 1n << BigInt(column * geometry.stride + rows - 1 - row);
      mask |= bit;
      if (cell === currentPlayer) current |= bit;
      moves += 1;
    }
  }
  return { geometry, current, mask, moves };
}

export function isExactClassicPosition(position) {
  return Boolean(
    position
      && position.chaosMode !== true
      && boardToClassicBitboard(position.board, position.currentPlayer, position.connect),
  );
}

function pieceCounts(position) {
  const current = popcount(position.current);
  return { current, opponent: position.moves - current };
}

function validateTurnCounts(position) {
  const counts = pieceCounts(position);
  if (Math.abs(counts.current - counts.opponent) > 1) {
    throw new RangeError('Classic position piece counts cannot differ by more than one.');
  }
  const valid = position.moves % 2 === 0
    ? counts.current === counts.opponent
    : counts.opponent === counts.current + 1;
  if (!valid) throw new RangeError('The side to move does not match the classic move counts.');
}

function terminalResult(geometry, position, currentPlayer, aiPlayer, start) {
  const currentWon = hasClassicAlignment(geometry, position.current);
  const opponentWon = hasClassicAlignment(geometry, position.current ^ position.mask);
  if (currentWon && opponentWon) {
    throw new RangeError('A classic position cannot contain winning lines for both players.');
  }
  if (!currentWon && !opponentWon && position.moves < geometry.cellCount) return null;

  const winner = currentWon
    ? currentPlayer
    : opponentWon ? (currentPlayer === RED ? YELLOW : RED) : EMPTY;
  const outcome = winner === EMPTY ? CLASSIC_DRAW : winner === aiPlayer ? CLASSIC_WIN : CLASSIC_LOSS;
  return {
    action: null,
    value: outcome,
    score: outcome === CLASSIC_DRAW ? 0 : outcome * (MATE_SCORE - position.moves),
    depth: 0,
    nodes: 0,
    elapsedMs: now() - start,
    tableHits: 0,
    cutoffs: 0,
    tableStores: 0,
    tableCollisions: 0,
    principalVariation: [],
    solved: true,
    solver: 'terminal',
  };
}

function defaultTableBits(cellCount) {
  if (cellCount <= 16) return 14;
  if (cellCount <= 25) return 18;
  if (cellCount <= 36) return 21;
  if (cellCount <= 42) return 22;
  return 23;
}

class ExactOutcomeTable {
  constructor(bits) {
    this.size = 2 ** bits;
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

class ExactClassicSearch {
  constructor(geometry, options = {}) {
    this.geometry = geometry;
    this.table = new ExactOutcomeTable(integerOption(
      options.tableBits,
      defaultTableBits(geometry.cellCount),
      8,
      24,
      'Exact classic transposition-table bits',
    ));
    this.maximumNodes = options.maximumNodes === Infinity
      ? Infinity
      : integerOption(
        options.maximumNodes,
        DEFAULT_MAXIMUM_NODES,
        1,
        Number.MAX_SAFE_INTEGER,
        'Exact classic node limit',
      );
    this.nodes = 0;
    this.tableHits = 0;
    this.cutoffs = 0;
    this.onProgress = null;
    this.nextProgressNode = 65_536;
    this.history = new Int32Array(2 * geometry.columns * geometry.stride);
  }

  visitNode() {
    this.nodes += 1;
    if (this.nodes > this.maximumNodes) {
      const error = new RangeError(
        `Exact classic search exceeded the ${this.maximumNodes.toLocaleString()}-node safety limit.`,
      );
      error.code = 'CLASSIC_EXACT_NODE_LIMIT';
      error.nodes = this.nodes;
      throw error;
    }
    if (this.onProgress && this.nodes >= this.nextProgressNode) {
      this.nextProgressNode += 65_536;
      this.onProgress(this);
    }
  }

  historyIndex(position, column) {
    const row = popcount(position.mask & this.geometry.columnMasks[column]);
    return (position.moves & 1) * this.geometry.columns * this.geometry.stride
      + column * this.geometry.stride + row;
  }

  orderedMoves(position, moves) {
    const candidates = [];
    for (const column of this.geometry.columnOrder) {
      const move = moveForColumn(this.geometry, position.mask, column);
      if ((moves & move) === 0n) continue;

      const nextMask = position.mask | move;
      let futureWins = 0;
      for (const futureColumn of this.geometry.columnOrder) {
        const future = moveForColumn(this.geometry, nextMask, futureColumn);
        if (future !== 0n
            && hasClassicAlignment(this.geometry, position.current | move | future)) {
          futureWins += 1;
        }
      }
      candidates.push({
        column,
        move,
        historyIndex: this.historyIndex(position, column),
        order: this.history[this.historyIndex(position, column)]
          + futureWins * 128
          + Math.round((this.geometry.columns
            - Math.abs(column - (this.geometry.columns - 1) / 2)) * 8),
      });
    }
    candidates.sort((first, second) => second.order - first.order);
    return candidates;
  }

  search(position, alpha, beta) {
    this.visitNode();
    const possible = possibleMoves(this.geometry, position.mask);
    if (possible === 0n) return CLASSIC_DRAW;
    if (immediateWinningMoves(this.geometry, position) !== 0n) return CLASSIC_WIN;

    const nonLosing = possibleClassicNonLosingMoves(this.geometry, position);
    if (nonLosing === 0n) return CLASSIC_LOSS;
    if (position.moves >= this.geometry.cellCount - 2) return CLASSIC_DRAW;

    const canonical = canonicalClassicPosition(this.geometry, position);
    const cached = this.table.probe(canonical.key);
    if (cached) {
      this.tableHits += 1;
      if (cached.lower >= beta) return cached.lower;
      if (cached.upper <= alpha) return cached.upper;
      alpha = Math.max(alpha, cached.lower);
      beta = Math.min(beta, cached.upper);
      if (alpha >= beta) return alpha;
    }

    const candidates = this.orderedMoves(position, nonLosing);
    const tried = [];
    for (const candidate of candidates) {
      const childScore = this.search(play(position, candidate.move), -beta, -alpha);
      const score = childScore === CLASSIC_DRAW ? CLASSIC_DRAW : -childScore;
      if (score >= beta) {
        this.cutoffs += 1;
        for (const previous of tried) this.history[previous.historyIndex] -= 1;
        this.history[candidate.historyIndex] += tried.length;
        this.table.storeLower(canonical.key, score);
        return score;
      }
      tried.push(candidate);
      if (score > alpha) alpha = score;
    }

    this.table.storeUpper(canonical.key, alpha);
    return alpha;
  }

  solve(position) {
    let minimum = CLASSIC_LOSS;
    let maximum = CLASSIC_WIN;
    while (minimum < maximum) {
      const middle = minimum + Math.floor((maximum - minimum) / 2);
      const score = this.search(position, middle, middle + 1);
      if (score <= middle) maximum = score;
      else minimum = score;
    }
    return minimum;
  }

  root(position) {
    const possible = possibleMoves(this.geometry, position.mask);
    if (possible === 0n) return { column: -1, outcome: CLASSIC_DRAW };

    const winning = immediateWinningMoves(this.geometry, position);
    if (winning !== 0n) {
      for (const column of this.geometry.columnOrder) {
        if ((winning & moveForColumn(this.geometry, position.mask, column)) !== 0n) {
          return { column, outcome: CLASSIC_WIN };
        }
      }
    }

    const nonLosing = possibleClassicNonLosingMoves(this.geometry, position);
    if (nonLosing === 0n) {
      for (const column of this.geometry.columnOrder) {
        if ((possible & moveForColumn(this.geometry, position.mask, column)) !== 0n) {
          return { column, outcome: CLASSIC_LOSS };
        }
      }
    }

    let bestColumn = -1;
    let bestOutcome = -2;
    for (const candidate of this.orderedMoves(position, nonLosing)) {
      const childOutcome = this.solve(play(position, candidate.move));
      const outcome = childOutcome === CLASSIC_DRAW ? CLASSIC_DRAW : -childOutcome;
      if (outcome > bestOutcome) {
        bestOutcome = outcome;
        bestColumn = candidate.column;
      }
      if (bestOutcome === CLASSIC_WIN) break;
    }
    return { column: bestColumn, outcome: bestOutcome };
  }
}

/**
 * Computes a game-theoretically exact move for any gravity-valid, non-Chaos
 * position with at most 7 rows and 7 columns. There is no depth or clock cutoff.
 * A caller may set maximumNodes for deterministic fail-closed batch work.
 */
export function solveClassicPosition(position, options = {}) {
  const encoded = boardToClassicBitboard(
    position?.board,
    position?.currentPlayer,
    position?.connect,
  );
  if (!encoded || position?.chaosMode === true) {
    throw new RangeError('Exact classic solving requires a gravity-valid non-Chaos board up to 7×7.');
  }

  const aiPlayer = options.aiPlayer ?? position.currentPlayer;
  if (aiPlayer !== RED && aiPlayer !== YELLOW) {
    throw new RangeError('AI player must be Red or Yellow.');
  }
  if (aiPlayer !== position.currentPlayer) {
    throw new RangeError('Exact classic solving can only choose a move for the side to move.');
  }
  if (options.maximumDepth !== undefined) {
    throw new RangeError('Exact classic solving does not accept a bounded-depth override.');
  }

  validateTurnCounts(encoded);
  const start = now();
  const terminal = terminalResult(
    encoded.geometry,
    encoded,
    position.currentPlayer,
    aiPlayer,
    start,
  );
  if (terminal) return terminal;

  const remaining = encoded.geometry.cellCount - encoded.moves;
  const initial = {
    action: null,
    value: null,
    score: 0,
    depth: remaining,
    nodes: 0,
    elapsedMs: 0,
    tableHits: 0,
    cutoffs: 0,
    tableStores: 0,
    tableCollisions: 0,
    principalVariation: [],
    solved: false,
    solver: 'classic-exact',
    rows: encoded.geometry.rows,
    columns: encoded.geometry.columns,
    connect: encoded.geometry.connect,
  };
  reportProgress(options.onIteration, initial);

  const search = new ExactClassicSearch(encoded.geometry, options);
  search.onProgress = () => reportProgress(options.onIteration, {
    ...initial,
    nodes: search.nodes,
    elapsedMs: now() - start,
    tableHits: search.tableHits,
    cutoffs: search.cutoffs,
    tableStores: search.table.stores,
    tableCollisions: search.table.collisions,
  });
  const selected = search.root(encoded);
  const action = selected.column < 0
    ? null
    : { type: ACTION_DROP, column: selected.column };
  const result = {
    ...initial,
    action,
    value: selected.outcome,
    score: selected.outcome === CLASSIC_DRAW
      ? 0
      : selected.outcome * (MATE_SCORE - encoded.moves),
    nodes: search.nodes,
    elapsedMs: now() - start,
    tableHits: search.tableHits,
    cutoffs: search.cutoffs,
    tableStores: search.table.stores,
    tableCollisions: search.table.collisions,
    principalVariation: action ? [{ ...action }] : [],
    solved: true,
  };
  reportProgress(options.onIteration, result);
  return result;
}
