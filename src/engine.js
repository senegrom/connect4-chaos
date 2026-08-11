export const EMPTY = 0;
export const RED = 1;
export const YELLOW = 2;

export const ACTION_DROP = 'drop';
export const ACTION_FLIP = 'flip';
export const ACTION_ROTATE_CW = 'rotateCW';
export const ACTION_ROTATE_CCW = 'rotateCCW';

const DIRECTIONS = Object.freeze([
  [0, 1],
  [1, 0],
  [1, 1],
  [1, -1],
]);

export function otherPlayer(player) {
  return player === RED ? YELLOW : RED;
}

export function clampInteger(value, minimum, maximum, fallback = minimum) {
  const parsed = Number.parseInt(String(value), 10);
  const safeValue = Number.isFinite(parsed) ? parsed : fallback;
  return Math.max(minimum, Math.min(maximum, safeValue));
}

export function normalizeConfig(config = {}) {
  const rows = clampInteger(config.rows, 4, 10, 6);
  const cols = clampInteger(config.cols, 4, 10, 7);
  const maximumConnect = Math.min(6, Math.max(rows, cols));
  const connect = clampInteger(config.connect, 3, maximumConnect, 4);
  const chaosMode = Boolean(config.chaosMode);
  let opponent = ['human', 'easy', 'medium', 'hard', 'brutal', 'perfect'].includes(config.opponent)
    ? config.opponent
    : 'medium';
  if (opponent === 'perfect'
      && (rows !== 6 || cols !== 7 || connect !== 4 || chaosMode)) {
    opponent = 'brutal';
  }

  return {
    rows,
    cols,
    connect,
    opponent,
    startingPlayer: Number(config.startingPlayer) === YELLOW ? YELLOW : RED,
    chaosMode,
  };
}

export function createBoard(rows, cols) {
  if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 1 || cols < 1) {
    throw new RangeError('Board dimensions must be positive integers.');
  }
  return Array.from({ length: rows }, () => Array(cols).fill(EMPTY));
}

export function cloneBoard(board) {
  return board.map((row) => [...row]);
}

export function boardDimensions(board) {
  return {
    rows: board.length,
    cols: board[0]?.length ?? 0,
  };
}

export function boardToString(board) {
  return board.map((row) => row.join('')).join('/');
}

export function positionKey(board, currentPlayer, connect, chaosMode = false) {
  const { rows, cols } = boardDimensions(board);
  return `${currentPlayer}:${rows}x${cols}:c${connect}:h${chaosMode ? 1 : 0}:${boardToString(board)}`;
}

export function getDropRow(board, column) {
  const { rows, cols } = boardDimensions(board);
  if (!Number.isInteger(column) || column < 0 || column >= cols) return -1;

  for (let row = rows - 1; row >= 0; row -= 1) {
    if (board[row][column] === EMPTY) return row;
  }
  return -1;
}

export function canDrop(board, column) {
  return getDropRow(board, column) >= 0;
}

export function legalDropColumns(board) {
  const { cols } = boardDimensions(board);
  const columns = [];
  for (let column = 0; column < cols; column += 1) {
    if (canDrop(board, column)) columns.push(column);
  }
  return columns;
}

export function isBoardFull(board) {
  return board.every((row) => row.every((cell) => cell !== EMPTY));
}

export function applyGravity(board) {
  const nextBoard = cloneBoard(board);
  const { rows, cols } = boardDimensions(nextBoard);

  for (let column = 0; column < cols; column += 1) {
    const pieces = [];
    for (let row = 0; row < rows; row += 1) {
      if (nextBoard[row][column] !== EMPTY) pieces.push(nextBoard[row][column]);
      nextBoard[row][column] = EMPTY;
    }

    let writeRow = rows - 1;
    for (let index = pieces.length - 1; index >= 0; index -= 1) {
      nextBoard[writeRow][column] = pieces[index];
      writeRow -= 1;
    }
  }

  return nextBoard;
}

export function flipBoard(board) {
  return applyGravity([...cloneBoard(board)].reverse());
}

export function rotateBoard(board, direction) {
  if (direction !== 1 && direction !== -1) {
    throw new RangeError('Rotation direction must be 1 (clockwise) or -1 (counter-clockwise).');
  }

  const { rows, cols } = boardDimensions(board);
  const rotated = createBoard(cols, rows);

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < cols; column += 1) {
      if (direction === 1) {
        rotated[column][rows - 1 - row] = board[row][column];
      } else {
        rotated[cols - 1 - column][row] = board[row][column];
      }
    }
  }

  return applyGravity(rotated);
}

export function applyAction(board, action, player) {
  if (!action || typeof action.type !== 'string') {
    throw new TypeError('An action with a type is required.');
  }

  if (action.type === ACTION_DROP) {
    const row = getDropRow(board, action.column);
    if (row < 0) return null;
    const nextBoard = cloneBoard(board);
    nextBoard[row][action.column] = player;
    return { board: nextBoard, row, column: action.column };
  }

  if (action.type === ACTION_FLIP) {
    return { board: flipBoard(board), row: null, column: null };
  }

  if (action.type === ACTION_ROTATE_CW) {
    return { board: rotateBoard(board, 1), row: null, column: null };
  }

  if (action.type === ACTION_ROTATE_CCW) {
    return { board: rotateBoard(board, -1), row: null, column: null };
  }

  throw new RangeError(`Unknown action type: ${action.type}`);
}

export function legalActions(board, chaosMode = false) {
  const actions = legalDropColumns(board).map((column) => ({ type: ACTION_DROP, column }));
  if (chaosMode) {
    actions.push(
      { type: ACTION_FLIP },
      { type: ACTION_ROTATE_CW },
      { type: ACTION_ROTATE_CCW },
    );
  }
  return actions;
}

function isInside(board, row, column) {
  const { rows, cols } = boardDimensions(board);
  return row >= 0 && row < rows && column >= 0 && column < cols;
}

export function hasWinFrom(board, row, column, player, connect) {
  if (!isInside(board, row, column) || board[row][column] !== player) return false;

  for (const [deltaRow, deltaColumn] of DIRECTIONS) {
    let count = 1;

    let nextRow = row + deltaRow;
    let nextColumn = column + deltaColumn;
    while (isInside(board, nextRow, nextColumn) && board[nextRow][nextColumn] === player) {
      count += 1;
      nextRow += deltaRow;
      nextColumn += deltaColumn;
    }

    nextRow = row - deltaRow;
    nextColumn = column - deltaColumn;
    while (isInside(board, nextRow, nextColumn) && board[nextRow][nextColumn] === player) {
      count += 1;
      nextRow -= deltaRow;
      nextColumn -= deltaColumn;
    }

    if (count >= connect) return true;
  }
  return false;
}

export function winningCells(board, player, connect) {
  const { rows, cols } = boardDimensions(board);
  const cells = new Set();

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < cols; column += 1) {
      if (board[row][column] !== player) continue;

      for (const [deltaRow, deltaColumn] of DIRECTIONS) {
        const endRow = row + (connect - 1) * deltaRow;
        const endColumn = column + (connect - 1) * deltaColumn;
        if (!isInside(board, endRow, endColumn)) continue;

        let matches = true;
        for (let offset = 1; offset < connect; offset += 1) {
          if (board[row + offset * deltaRow][column + offset * deltaColumn] !== player) {
            matches = false;
            break;
          }
        }

        if (matches) {
          for (let offset = 0; offset < connect; offset += 1) {
            cells.add(`${row + offset * deltaRow},${column + offset * deltaColumn}`);
          }
        }
      }
    }
  }

  return [...cells].map((cell) => cell.split(',').map(Number));
}

export function resolveActionOutcome(board, connect, mover, actionType, lastDrop = null) {
  let redCells = [];
  let yellowCells = [];

  if (actionType === ACTION_DROP && lastDrop) {
    if (hasWinFrom(board, lastDrop.row, lastDrop.column, mover, connect)) {
      const cells = winningCells(board, mover, connect);
      return {
        status: 'won',
        winner: mover,
        winningCells: cells,
        simultaneousWin: false,
      };
    }
  } else {
    redCells = winningCells(board, RED, connect);
    yellowCells = winningCells(board, YELLOW, connect);

    if (redCells.length > 0 && yellowCells.length > 0) {
      const winner = otherPlayer(mover);
      return {
        status: 'won',
        winner,
        winningCells: winner === RED ? redCells : yellowCells,
        simultaneousWin: true,
      };
    }

    if (redCells.length > 0 || yellowCells.length > 0) {
      const winner = redCells.length > 0 ? RED : YELLOW;
      return {
        status: 'won',
        winner,
        winningCells: winner === RED ? redCells : yellowCells,
        simultaneousWin: false,
      };
    }
  }

  if (isBoardFull(board)) {
    return { status: 'draw', winner: EMPTY, winningCells: [], simultaneousWin: false };
  }

  return { status: 'playing', winner: EMPTY, winningCells: [], simultaneousWin: false };
}

export function actionLabel(action) {
  switch (action?.type) {
    case ACTION_DROP:
      return `dropped in column ${action.column + 1}`;
    case ACTION_FLIP:
      return 'flipped the board';
    case ACTION_ROTATE_CW:
      return 'rotated clockwise';
    case ACTION_ROTATE_CCW:
      return 'rotated counter-clockwise';
    default:
      return 'moved';
  }
}

export function sameAction(first, second) {
  return first?.type === second?.type
    && (first?.type !== ACTION_DROP || first.column === second.column);
}
