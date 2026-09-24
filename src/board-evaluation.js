// The static board evaluation: piece placement, open windows and immediate
// threats. It lives apart from the search in ai.js so that the page, which
// shows it beside the board, loads no search code until a move needs one.
import {
  EMPTY,
  YELLOW,
  boardDimensions,
  getDropRow,
  immediateWinningDropActions,
  otherPlayer,
} from './engine.js';

function shapeWeight(pieceCount) {
  const weights = [0, 4, 24, 180, 1_200, 8_000, 50_000];
  return weights[Math.min(pieceCount, weights.length - 1)];
}

function windowScore(board, coordinates, connect, aiPlayer, dropRows) {
  let aiCount = 0;
  let opponentCount = 0;
  const emptyCoordinates = [];

  for (const [row, column] of coordinates) {
    const cell = board[row][column];
    if (cell === aiPlayer) aiCount += 1;
    else if (cell === EMPTY) emptyCoordinates.push([row, column]);
    else opponentCount += 1;
  }

  if ((aiCount > 0 && opponentCount > 0) || emptyCoordinates.length === connect) return 0;

  const pieceCount = Math.max(aiCount, opponentCount);
  let score = shapeWeight(pieceCount);
  const playableEmptyCount = emptyCoordinates.reduce((count, [row, column]) => (
    count + (dropRows[column] === row ? 1 : 0)
  ), 0);

  if (pieceCount === connect - 1 && emptyCoordinates.length === 1) {
    score *= playableEmptyCount === 1 ? 8 : 2;
  } else if (pieceCount === connect - 2 && emptyCoordinates.length === 2) {
    score = Math.round(score * (1 + playableEmptyCount * 0.8));
  } else if (playableEmptyCount > 0) {
    score = Math.round(score * (1 + playableEmptyCount * 0.18));
  }

  if (aiCount > 0) return score;
  return -Math.round(score * 1.14);
}

export function evaluateShape(board, connect, aiPlayer) {
  const { rows, cols } = boardDimensions(board);
  let score = 0;

  const centre = (cols - 1) / 2;
  const dropRows = Array.from({ length: cols }, (_, column) => getDropRow(board, column));
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < cols; column += 1) {
      const cell = board[row][column];
      if (cell === EMPTY) continue;
      const value = Math.max(1, Math.round(cols / 2 - Math.abs(column - centre) + 1));
      score += cell === aiPlayer ? value * 3 : -value * 3;
    }
  }

  const directions = [
    [0, 1],
    [1, 0],
    [1, 1],
    [1, -1],
  ];

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < cols; column += 1) {
      for (const [deltaRow, deltaColumn] of directions) {
        const endRow = row + (connect - 1) * deltaRow;
        const endColumn = column + (connect - 1) * deltaColumn;
        if (endRow < 0 || endRow >= rows || endColumn < 0 || endColumn >= cols) continue;

        const coordinates = [];
        for (let offset = 0; offset < connect; offset += 1) {
          coordinates.push([row + offset * deltaRow, column + offset * deltaColumn]);
        }
        score += windowScore(board, coordinates, connect, aiPlayer, dropRows);
      }
    }
  }

  return score;
}

export function evaluateWithThreatCounts(board, connect, aiPlayer, aiWinningDrops, opponentWinningDrops) {
  let score = evaluateShape(board, connect, aiPlayer);
  score += aiWinningDrops * 5_500;
  score -= opponentWinningDrops * 6_200;
  if (aiWinningDrops >= 2) score += 22_000 + (aiWinningDrops - 2) * 4_000;
  if (opponentWinningDrops >= 2) score -= 25_000 + (opponentWinningDrops - 2) * 4_500;
  return score;
}

export function evaluateBoard(board, connect, aiPlayer = YELLOW) {
  const opponent = otherPlayer(aiPlayer);
  const aiWinningDrops = immediateWinningDropActions(board, aiPlayer, connect).length;
  const opponentWinningDrops = immediateWinningDropActions(board, opponent, connect).length;
  return evaluateWithThreatCounts(
    board,
    connect,
    aiPlayer,
    aiWinningDrops,
    opponentWinningDrops,
  );
}
