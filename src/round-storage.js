import { EMPTY, RED, YELLOW, cloneBoard, positionKey } from './engine.js';

export const SETTINGS_KEY = 'connect4-chaos.settings.v1';
export const SCORES_KEY = 'connect4-chaos.scores.v1';
export const ROUND_KEY = 'connect4-chaos.round.v1';

/** Keep reload recovery in this tab; retain the latest round for a new tab. */
export function createRoundStore({
  sharedStorage = () => globalThis.localStorage,
  tabStorage = () => globalThis.sessionStorage,
} = {}) {
  const read = (getStorage) => {
    try { return getStorage().getItem(ROUND_KEY); } catch { return null; }
  };
  const parse = (value) => {
    try { return JSON.parse(value); } catch { return null; }
  };
  const write = (getStorage, round) => {
    try { getStorage().setItem(ROUND_KEY, JSON.stringify(round)); } catch { /* storage is optional */ }
  };
  return {
    read() {
      const saved = read(tabStorage);
      // An explicit null marks a completed round. It must not resume another
      // tab's game; only a tab with no recovery entry uses the shared save.
      return parse(saved === null ? read(sharedStorage) : saved);
    },
    save(round) {
      write(tabStorage, round);
      write(sharedStorage, round);
    },
    clear(roundId) {
      write(tabStorage, null);
      try {
        if (parse(read(sharedStorage))?.roundId === roundId) sharedStorage().removeItem(ROUND_KEY);
      } catch { /* another tab's recovery must remain intact */ }
    },
  };
}

export function storageHasValue(key) {
  try {
    return localStorage.getItem(key) !== null;
  } catch {
    return false;
  }
}

export function loadJson(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return (value ? JSON.parse(value) : null) ?? fallback;
  } catch {
    return fallback;
  }
}

export function saveJson(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // The game remains fully usable when storage is unavailable.
  }
}

export function normalizeScores(scores) {
  return {
    [RED]: Math.max(0, Number.parseInt(scores?.[RED] ?? scores?.red ?? 0, 10) || 0),
    [YELLOW]: Math.max(0, Number.parseInt(scores?.[YELLOW] ?? scores?.yellow ?? 0, 10) || 0),
    draw: Math.max(0, Number.parseInt(scores?.draw ?? 0, 10) || 0),
  };
}

/**
 * Applies this tab's score change to the latest shared tally. This prevents an
 * older tab from overwriting wins recorded elsewhere while still allowing Undo
 * to reverse the result contributed by the current round.
 */
export function mergeScoreDelta(sharedScores, beforeScores, afterScores) {
  const shared = normalizeScores(sharedScores);
  const before = normalizeScores(beforeScores);
  const after = normalizeScores(afterScores);
  return {
    [RED]: Math.max(0, shared[RED] + after[RED] - before[RED]),
    [YELLOW]: Math.max(0, shared[YELLOW] + after[YELLOW] - before[YELLOW]),
    draw: Math.max(0, shared.draw + after.draw - before.draw),
  };
}

export function makeSnapshot(state) {
  return {
    board: cloneBoard(state.board),
    currentPlayer: state.currentPlayer,
    status: state.status,
    winner: state.winner,
    winningCells: state.winningCells.map((cell) => [...cell]),
    simultaneousWin: state.simultaneousWin,
    drawReason: state.drawReason,
    lastMove: state.lastMove ? { ...state.lastMove } : null,
    lastMover: state.lastMover,
    moveCount: state.moveCount,
    selectedColumn: state.selectedColumn,
    repetitionCounts: [...state.repetitionCounts.entries()],
    // Retained only for legacy saved-round compatibility. The transactional
    // score ledger is authoritative; Undo uses result receipts, never totals.
    scores: { ...state.scores },
    lastSearch: state.lastSearch ? { ...state.lastSearch } : null,
  };
}

export function restoreSnapshot(state, snapshot, options = {}) {
  state.board = cloneBoard(snapshot.board);
  state.currentPlayer = snapshot.currentPlayer;
  state.status = snapshot.status;
  state.winner = snapshot.winner;
  state.winningCells = snapshot.winningCells.map((cell) => [...cell]);
  state.simultaneousWin = snapshot.simultaneousWin;
  state.drawReason = snapshot.drawReason;
  state.lastMove = snapshot.lastMove ? { ...snapshot.lastMove } : null;
  state.lastMover = snapshot.lastMover;
  state.moveCount = snapshot.moveCount;
  state.selectedColumn = snapshot.selectedColumn;
  state.repetitionCounts = new Map(snapshot.repetitionCounts);
  if (options.restoreScores !== false) state.scores = { ...snapshot.scores };
  const key = positionKey(state.board, state.currentPlayer, state.config.connect, state.config.chaosMode);
  state.lastSearch = snapshot.lastSearch?.positionKey === key ? { ...snapshot.lastSearch } : null;
  state.liveSearch = null;
  state.dropAnimation = null;
  state.aiError = null;
}

export function sameConfig(a, b) {
  return a.rows === b.rows
    && a.cols === b.cols
    && a.connect === b.connect
    && a.opponent === b.opponent
    && a.startingPlayer === b.startingPlayer
    && a.chaosMode === b.chaosMode;
}

export function validSnapshot(snapshot, config) {
  if (!snapshot || !Array.isArray(snapshot.board) || !Array.isArray(snapshot.board[0])) return false;
  // A Chaos rotation transposes the board, so either orientation is valid there.
  const rows = snapshot.board.length;
  const cols = snapshot.board[0].length;
  const upright = rows === config.rows && cols === config.cols;
  const transposed = config.chaosMode && rows === config.cols && cols === config.rows;
  if (!upright && !transposed) return false;
  const cells = snapshot.board.every((row) => Array.isArray(row)
    && row.length === cols
    && row.every((cell) => cell === EMPTY || cell === RED || cell === YELLOW));
  if (!cells) return false;
  if (snapshot.currentPlayer !== RED && snapshot.currentPlayer !== YELLOW) return false;
  const cellPosition = (cell) => Array.isArray(cell) && cell.length === 2
    && Number.isInteger(cell[0]) && cell[0] >= 0 && cell[0] < rows
    && Number.isInteger(cell[1]) && cell[1] >= 0 && cell[1] < cols;
  return ['playing', 'won', 'draw'].includes(snapshot.status)
    && [EMPTY, RED, YELLOW].includes(snapshot.winner)
    && Number.isSafeInteger(snapshot.moveCount) && snapshot.moveCount >= 0
    && Number.isInteger(snapshot.selectedColumn) && snapshot.selectedColumn >= 0 && snapshot.selectedColumn < cols
    && Array.isArray(snapshot.winningCells) && snapshot.winningCells.every(cellPosition)
    && Array.isArray(snapshot.repetitionCounts) && snapshot.repetitionCounts.every((entry) => (
      Array.isArray(entry) && entry.length === 2 && typeof entry[0] === 'string'
      && Number.isSafeInteger(entry[1]) && entry[1] > 0 && entry[1] <= 3
    ))
    && snapshot.scores && [RED, YELLOW, 'draw'].every((key) => (
      Number.isSafeInteger(snapshot.scores[key]) && snapshot.scores[key] >= 0
    ))
    && (!snapshot.lastMove || cellPosition([snapshot.lastMove.row, snapshot.lastMove.column]));
}
