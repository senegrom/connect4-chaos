import { chooseBitboardMove, isBitboardPosition } from './bitboard.js';

import {
  ACTION_DROP,
  ACTION_FLIP,
  ACTION_ROTATE_CCW,
  ACTION_ROTATE_CW,
  EMPTY,
  RED,
  YELLOW,
  applyAction,
  boardDimensions,
  boardToString,
  getDropRow,
  hasWinFrom,
  legalActions,
  otherPlayer,
  positionKey,
  resolveActionOutcome,
  sameAction,
} from './engine.js';

const INF = 1_000_000_000;
const MATE_SCORE = 10_000_000;
const MAX_TABLE_ENTRIES = 350_000;
const PREFERRED_MOVE_BONUS = 4_000_000;
const FIRST_KILLER_BONUS = 240_000;
const SECOND_KILLER_BONUS = 120_000;

const DIFFICULTY = Object.freeze({
  medium: { maximumDepth: 6, chaosMaximumDepth: 4, quiescenceDepth: 2 },
  hard: { maximumDepth: 9, chaosMaximumDepth: 5, quiescenceDepth: 3 },
  brutal: { maximumDepth: 12, chaosMaximumDepth: 6, quiescenceDepth: 4 },
});

const VALID_DIFFICULTIES = new Set(['easy', 'medium', 'hard', 'brutal', 'perfect']);

function now() {
  return globalThis.performance?.now?.() ?? Date.now();
}

function visitNode(context) {
  context.nodes += 1;
}

function copyRepetitionCounts(entries) {
  if (entries === undefined || entries === null) return new Map();
  let pairs;
  if (entries instanceof Map) pairs = entries.entries();
  else if (Array.isArray(entries)) pairs = entries;
  else if (typeof entries === 'object') pairs = Object.entries(entries);
  else throw new TypeError('Repetition counts must be a map, entry array, or object.');

  const repetitions = new Map();
  for (const pair of pairs) {
    if (!Array.isArray(pair) || pair.length < 2 || typeof pair[0] !== 'string') {
      throw new TypeError('Each repetition entry must contain a string key and count.');
    }
    const count = pair[1];
    if (!Number.isInteger(count) || count < 0) {
      throw new RangeError('Repetition counts must be non-negative integers.');
    }
    if (count > 0) repetitions.set(pair[0], count);
  }
  return repetitions;
}

function integerSearchOption(value, fallback, minimum, maximum, label) {
  const selected = value ?? fallback;
  if (!Number.isInteger(selected) || selected < minimum || selected > maximum) {
    throw new RangeError(`${label} must be an integer from ${minimum} through ${maximum}.`);
  }
  return selected;
}

function validateSearchPosition(position) {
  if (!position || !Array.isArray(position.board) || position.board.length === 0) {
    throw new TypeError('A non-empty board is required.');
  }
  if (position.currentPlayer !== RED && position.currentPlayer !== YELLOW) {
    throw new RangeError('Current player must be Red or Yellow.');
  }

  const rows = position.board.length;
  const cols = position.board[0]?.length ?? 0;
  if (cols === 0 || position.board.some((row) => !Array.isArray(row) || row.length !== cols)) {
    throw new TypeError('The board must be a non-empty rectangle.');
  }
  if (position.board.some((row) => row.some((cell) => (
    cell !== EMPTY && cell !== RED && cell !== YELLOW
  )))) {
    throw new RangeError('Board cells must be empty, Red, or Yellow.');
  }
  if (!Number.isInteger(position.connect)
      || position.connect < 1
      || position.connect > Math.max(rows, cols)) {
    throw new RangeError('Connect length must fit the board.');
  }

  let redCount = 0;
  let yellowCount = 0;
  for (let column = 0; column < cols; column += 1) {
    let foundPiece = false;
    for (let row = 0; row < rows; row += 1) {
      const cell = position.board[row][column];
      if (cell !== EMPTY) {
        foundPiece = true;
        if (cell === RED) redCount += 1;
        else yellowCount += 1;
      } else if (foundPiece) {
        throw new RangeError('Board pieces must obey gravity.');
      }
    }
  }

  if (!position.chaosMode && Math.abs(redCount - yellowCount) > 1) {
    throw new RangeError('Classic positions cannot differ by more than one piece.');
  }
  return { redCount, yellowCount };
}

function validateClassicTurn(position, counts) {
  if (position.chaosMode) return;
  if (counts.redCount > counts.yellowCount && position.currentPlayer !== YELLOW) {
    throw new RangeError('The side to move does not match the classic move counts.');
  }
  if (counts.yellowCount > counts.redCount && position.currentPlayer !== RED) {
    throw new RangeError('The side to move does not match the classic move counts.');
  }
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

function terminalResult(position, aiPlayer, start) {
  const redWon = boardHasWin(position.board, RED, position.connect);
  const yellowWon = boardHasWin(position.board, YELLOW, position.connect);
  const boardFull = position.board.every((row) => row.every((cell) => cell !== EMPTY));
  if (!redWon && !yellowWon && !boardFull) return null;

  if (redWon && yellowWon) {
    throw new RangeError('A searchable position cannot contain winning lines for both players.');
  }
  const winner = redWon ? RED : yellowWon ? YELLOW : EMPTY;
  const score = winner === EMPTY ? 0 : mateScoreForWinner(winner, aiPlayer, 0);
  return {
    action: null,
    score,
    depth: 0,
    nodes: 0,
    elapsedMs: now() - start,
    tableHits: 0,
    cutoffs: 0,
    tableResets: 0,
    principalVariation: [],
    solved: true,
    solver: 'terminal',
  };
}

function aspirationSearch(depth, previousScore, search) {
  const useAspiration = depth >= 4 && Math.abs(previousScore) < MATE_SCORE / 2;
  if (!useAspiration) return search(-INF, INF);

  let window = 180 + depth * 22;
  let alpha = previousScore - window;
  let beta = previousScore + window;
  let result = search(alpha, beta);
  if (result.score > alpha && result.score < beta) return result;

  window *= 4;
  alpha = previousScore - window;
  beta = previousScore + window;
  result = search(alpha, beta);
  return result.score > alpha && result.score < beta
    ? result
    : search(-INF, INF);
}

function randomChoice(values, random) {
  if (values.length === 0) return null;
  const sample = Number(random());
  if (!Number.isFinite(sample)) throw new RangeError('Random source must return a finite number.');
  const index = Math.max(0, Math.min(values.length - 1, Math.floor(sample * values.length)));
  return values[index];
}

function incrementRepetition(repetitions, key) {
  const previous = repetitions.get(key) ?? 0;
  repetitions.set(key, previous + 1);
  return previous + 1;
}

function decrementRepetition(repetitions, key) {
  const current = repetitions.get(key) ?? 0;
  if (current <= 1) repetitions.delete(key);
  else repetitions.set(key, current - 1);
}

function actionOutcome(result, action, player, connect) {
  return resolveActionOutcome(
    result.board,
    connect,
    player,
    action.type,
    action.type === ACTION_DROP ? { row: result.row, column: result.column } : null,
  );
}

function terminalScore(outcome, aiPlayer, ply) {
  if (outcome.status === 'draw') return 0;
  if (outcome.status !== 'won') return null;
  return outcome.winner === aiPlayer ? MATE_SCORE - ply : -MATE_SCORE + ply;
}

function mateScoreForWinner(winner, aiPlayer, ply) {
  return winner === aiPlayer ? MATE_SCORE - ply : -MATE_SCORE + ply;
}

function centralityScore(column, cols) {
  const centre = (cols - 1) / 2;
  return Math.round((cols - Math.abs(column - centre)) * 12);
}

function actionKey(action, player) {
  if (!action) return '';
  return action.type === ACTION_DROP
    ? `${player}:drop:${action.column}`
    : `${player}:${action.type}`;
}

function mirrorAction(action, cols) {
  if (!action) return null;
  return { type: ACTION_DROP, column: cols - 1 - action.column };
}

function mirroredBoardString(board) {
  return board.map((row) => [...row].reverse().join('')).join('/');
}

function canonicalTablePosition(board, player) {
  const { rows, cols } = boardDimensions(board);
  const normal = boardToString(board);
  const mirrored = mirroredBoardString(board);
  const useMirror = mirrored < normal;
  return {
    key: `${player}|${rows}x${cols}|${useMirror ? mirrored : normal}`,
    mirrored: useMirror,
    cols,
  };
}

function tableActionToBoard(action, tablePosition) {
  return tablePosition.mirrored ? mirrorAction(action, tablePosition.cols) : action;
}

function boardActionToTable(action, tablePosition) {
  return tablePosition.mirrored ? mirrorAction(action, tablePosition.cols) : action;
}

function immediateWinningDropActions(board, player, connect) {
  const { cols } = boardDimensions(board);
  const wins = [];

  for (let column = 0; column < cols; column += 1) {
    const row = getDropRow(board, column);
    if (row < 0) continue;
    board[row][column] = player;
    try {
      if (hasWinFrom(board, row, column, player, connect)) {
        wins.push({ type: ACTION_DROP, column });
      }
    } finally {
      board[row][column] = EMPTY;
    }
  }

  return wins;
}

function immediateWinningActions(board, player, connect, chaosMode) {
  const wins = immediateWinningDropActions(board, player, connect);
  if (!chaosMode) return wins;

  const transformations = [
    { type: ACTION_FLIP },
    { type: ACTION_ROTATE_CW },
    { type: ACTION_ROTATE_CCW },
  ];

  for (const action of transformations) {
    const result = applyAction(board, action, player);
    if (!result) continue;
    const outcome = actionOutcome(result, action, player, connect);
    if (outcome.status === 'won' && outcome.winner === player) wins.push(action);
  }
  return wins;
}

function tacticallySafeActions(position) {
  const { board, connect, currentPlayer, chaosMode } = position;
  const opponent = otherPlayer(currentPlayer);
  const repetitions = copyRepetitionCounts(position.repetitionCounts);
  const safe = [];

  for (const action of legalActions(board, chaosMode)) {
    const result = applyAction(board, action, currentPlayer);
    if (!result) continue;
    const outcome = actionOutcome(result, action, currentPlayer, connect);
    if (outcome.status === 'won') {
      if (outcome.winner === currentPlayer) safe.push(action);
      continue;
    }
    if (outcome.status === 'draw') {
      safe.push(action);
      continue;
    }

    const repetitionKey = positionKey(result.board, opponent, connect, chaosMode);
    if ((repetitions.get(repetitionKey) ?? 0) + 1 >= 3) {
      safe.push(action);
      continue;
    }

    if (immediateWinningActions(result.board, opponent, connect, chaosMode).length === 0) {
      safe.push(action);
    }
  }
  return safe;
}

function enforceTacticalSafety(position, result) {
  if (!result?.action) return result;
  const safe = tacticallySafeActions(position);
  if (safe.length === 0 || safe.some((action) => sameAction(action, result.action))) return result;

  const { cols } = boardDimensions(position.board);
  safe.sort((first, second) => {
    const firstScore = first.type === ACTION_DROP ? centralityScore(first.column, cols) : -1;
    const secondScore = second.type === ACTION_DROP ? centralityScore(second.column, cols) : -1;
    return secondScore - firstScore;
  });
  const action = safe[0];
  return { ...result, action, principalVariation: [{ ...action }], tacticalOverride: true };
}

function chooseEasy(position, random = Math.random) {
  if (typeof random !== 'function') throw new TypeError('Random source must be a function.');
  const {
    board,
    connect,
    currentPlayer,
    chaosMode,
  } = position;
  const ownWins = immediateWinningActions(board, currentPlayer, connect, chaosMode);
  if (ownWins.length > 0) return randomChoice(ownWins, random);

  const actions = legalActions(board, chaosMode);
  if (actions.length === 0) return null;
  const safeActions = tacticallySafeActions(position);
  const candidates = safeActions.length > 0 ? safeActions : actions;

  const { cols } = boardDimensions(board);
  const weighted = [];
  for (const action of candidates) {
    let weight = 2;
    if (action.type === ACTION_DROP) {
      weight = Math.max(2, Math.round(cols - Math.abs(action.column - (cols - 1) / 2)));
    }
    for (let index = 0; index < weight; index += 1) weighted.push(action);
  }

  return randomChoice(weighted, random);
}

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

function evaluateShape(board, connect, aiPlayer) {
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

function evaluateWithThreatCounts(board, connect, aiPlayer, aiWinningDrops, opponentWinningDrops) {
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

function childOrderScore(child, player, context, preferredAction, ply) {
  if (child.outcome.status === 'won') {
    return child.outcome.winner === player ? MATE_SCORE : -MATE_SCORE;
  }
  if (child.outcome.status === 'draw') return 0;

  let score = 0;
  if (ply <= 1) {
    score = evaluateShape(child.result.board, context.connect, context.aiPlayer);
    if (player !== context.aiPlayer) score = -score;
  }

  const { cols } = boardDimensions(child.result.board);
  if (child.action.type === ACTION_DROP) {
    score += centralityScore(child.action.column, cols);
  } else {
    score -= 18;
  }

  if (preferredAction && sameAction(child.action, preferredAction)) score += PREFERRED_MOVE_BONUS;

  const key = actionKey(child.action, player);
  const killers = context.killers[ply] ?? [];
  if (killers[0] && sameAction(child.action, killers[0])) score += FIRST_KILLER_BONUS;
  else if (killers[1] && sameAction(child.action, killers[1])) score += SECOND_KILLER_BONUS;
  score += Math.min(100_000, context.historyHeuristic.get(key) ?? 0);
  return score;
}

function makeChildren(board, player, context, preferredAction = null, ply = 0) {
  const actions = legalActions(board, context.chaosMode);
  const children = [];
  const seenPositions = new Set();

  for (const action of actions) {
    const result = applyAction(board, action, player);
    if (!result) continue;

    const { rows, cols } = boardDimensions(result.board);
    const nextPosition = `${otherPlayer(player)}|${rows}x${cols}|${boardToString(result.board)}`;
    if (seenPositions.has(nextPosition)) continue;
    seenPositions.add(nextPosition);

    const outcome = actionOutcome(result, action, player, context.connect);
    const child = { action, result, outcome };
    child.orderScore = childOrderScore(child, player, context, preferredAction, ply);
    children.push(child);
  }

  children.sort((first, second) => second.orderScore - first.orderScore);
  return children;
}

function recordCutoff(context, action, player, depth, ply) {
  const current = context.killers[ply] ?? [];
  if (!current[0] || !sameAction(current[0], action)) {
    context.killers[ply] = [action, current[0]].filter(Boolean).slice(0, 2);
  }

  const key = actionKey(action, player);
  const previous = context.historyHeuristic.get(key) ?? 0;
  context.historyHeuristic.set(key, previous + depth * depth * 16);
  context.cutoffs += 1;
}

function withRepetition(child, player, repetitions, context, callback) {
  const nextPlayer = otherPlayer(player);
  const repetitionKey = positionKey(
    child.result.board,
    nextPlayer,
    context.connect,
    context.chaosMode,
  );
  const repetitionCount = incrementRepetition(repetitions, repetitionKey);
  try {
    return repetitionCount >= 3 ? 0 : callback(nextPlayer);
  } finally {
    decrementRepetition(repetitions, repetitionKey);
  }
}

function quiescence(board, player, alpha, beta, ply, repetitions, context, remainingDepth) {
  visitNode(context);

  const ownWins = immediateWinningActions(
    board,
    player,
    context.connect,
    context.chaosMode,
  );
  if (ownWins.length > 0) return mateScoreForWinner(player, context.aiPlayer, ply);

  const opponent = otherPlayer(player);
  const opponentWins = immediateWinningActions(
    board,
    opponent,
    context.connect,
    context.chaosMode,
  );

  if (opponentWins.length === 0 || remainingDepth <= 0) {
    const aiWinningDrops = player === context.aiPlayer
      ? ownWins.filter((action) => action.type === ACTION_DROP).length
      : opponentWins.filter((action) => action.type === ACTION_DROP).length;
    const opponentWinningDrops = player === context.aiPlayer
      ? opponentWins.filter((action) => action.type === ACTION_DROP).length
      : ownWins.filter((action) => action.type === ACTION_DROP).length;
    return evaluateWithThreatCounts(
      board,
      context.connect,
      context.aiPlayer,
      aiWinningDrops,
      opponentWinningDrops,
    );
  }

  const children = makeChildren(board, player, context, null, ply);
  const defensiveChildren = [];

  for (const child of children) {
    const immediateScore = terminalScore(child.outcome, context.aiPlayer, ply);
    if (immediateScore !== null) {
      defensiveChildren.push({ child, immediateScore });
      continue;
    }

    const nextOpponentWins = immediateWinningActions(
      child.result.board,
      opponent,
      context.connect,
      context.chaosMode,
    );
    if (nextOpponentWins.length === 0) defensiveChildren.push({ child, immediateScore: null });
  }

  if (defensiveChildren.length === 0) {
    return mateScoreForWinner(opponent, context.aiPlayer, ply + 1);
  }

  const maximizing = player === context.aiPlayer;
  let bestScore = maximizing ? -INF : INF;

  for (const { child, immediateScore } of defensiveChildren) {
    visitNode(context);
    const score = immediateScore ?? withRepetition(
      child,
      player,
      repetitions,
      context,
      (nextPlayer) => quiescence(
        child.result.board,
        nextPlayer,
        alpha,
        beta,
        ply + 1,
        repetitions,
        context,
        remainingDepth - 1,
      ),
    );

    if (maximizing) {
      bestScore = Math.max(bestScore, score);
      alpha = Math.max(alpha, bestScore);
    } else {
      bestScore = Math.min(bestScore, score);
      beta = Math.min(beta, bestScore);
    }
    if (alpha >= beta) break;
  }

  return bestScore;
}

function minimax(board, player, depth, alpha, beta, ply, repetitions, context) {
  if (depth <= 0) {
    return quiescence(
      board,
      player,
      alpha,
      beta,
      ply,
      repetitions,
      context,
      context.quiescenceDepth,
    );
  }
  visitNode(context);

  const children = makeChildren(board, player, context, null, ply);
  if (children.length === 0) return 0;

  const maximizing = player === context.aiPlayer;
  let bestScore = maximizing ? -INF : INF;

  for (const child of children) {
    visitNode(context);
    const immediateScore = terminalScore(child.outcome, context.aiPlayer, ply);
    const score = immediateScore ?? withRepetition(
      child,
      player,
      repetitions,
      context,
      (nextPlayer) => minimax(
        child.result.board,
        nextPlayer,
        depth - 1,
        alpha,
        beta,
        ply + 1,
        repetitions,
        context,
      ),
    );

    if (maximizing) {
      if (score > bestScore) {
        bestScore = score;
      }
      alpha = Math.max(alpha, bestScore);
    } else {
      if (score < bestScore) {
        bestScore = score;
      }
      beta = Math.min(beta, bestScore);
    }

    if (alpha >= beta) {
      recordCutoff(context, child.action, player, depth, ply);
      break;
    }
  }

  return bestScore;
}

function searchRoot(position, depth, repetitions, context, preferredAction, alpha = -INF, beta = INF) {
  const children = makeChildren(position.board, position.currentPlayer, context, preferredAction, 0);
  if (children.length === 0) return { action: null, score: 0 };

  const maximizing = position.currentPlayer === context.aiPlayer;
  let bestAction = children[0].action;
  let bestScore = maximizing ? -INF : INF;

  for (const child of children) {
    visitNode(context);
    const immediateScore = terminalScore(child.outcome, context.aiPlayer, 0);
    const score = immediateScore ?? withRepetition(
      child,
      position.currentPlayer,
      repetitions,
      context,
      (nextPlayer) => minimax(
        child.result.board,
        nextPlayer,
        depth - 1,
        alpha,
        beta,
        1,
        repetitions,
        context,
      ),
    );

    if ((maximizing && score > bestScore) || (!maximizing && score < bestScore)) {
      bestScore = score;
      bestAction = child.action;
    }

    if (maximizing) alpha = Math.max(alpha, bestScore);
    else beta = Math.min(beta, bestScore);
    if (alpha >= beta) break;
  }

  return { action: bestAction, score: bestScore };
}

function principalVariation(firstAction) {
  return firstAction ? [{ ...firstAction }] : [];
}

// Fallback mutable search for configurable classic boards. The board passed by the UI
// is cloned once; every search move is then made and unmade in a finally block.
function classicHeights(board) {
  const { cols } = boardDimensions(board);
  return Array.from({ length: cols }, (_, column) => getDropRow(board, column));
}

function playClassicDrop(board, heights, column, player) {
  const row = heights[column];
  if (row < 0) return -1;
  board[row][column] = player;
  heights[column] = row - 1;
  return row;
}

function undoClassicDrop(board, heights, column, row) {
  board[row][column] = EMPTY;
  heights[column] = row;
}

function classicWinningColumns(board, heights, player, connect) {
  const wins = [];
  for (let column = 0; column < heights.length; column += 1) {
    const row = heights[column];
    if (row < 0) continue;
    board[row][column] = player;
    const winsNow = hasWinFrom(board, row, column, player, connect);
    board[row][column] = EMPTY;
    if (winsNow) wins.push(column);
  }
  return wins;
}

function evaluateClassic(board, heights, connect, aiPlayer) {
  const opponent = otherPlayer(aiPlayer);
  const aiWinningDrops = classicWinningColumns(board, heights, aiPlayer, connect).length;
  const opponentWinningDrops = classicWinningColumns(board, heights, opponent, connect).length;
  return evaluateWithThreatCounts(
    board,
    connect,
    aiPlayer,
    aiWinningDrops,
    opponentWinningDrops,
  );
}

function classicColumnOrderScore(
  board,
  heights,
  column,
  player,
  context,
  preferredAction,
  opponentWinningColumns,
  ply,
) {
  const action = { type: ACTION_DROP, column };
  let score = centralityScore(column, heights.length);

  if (preferredAction && sameAction(action, preferredAction)) score += PREFERRED_MOVE_BONUS;
  if (opponentWinningColumns.has(column)) score += PREFERRED_MOVE_BONUS / 2;

  const killers = context.killers[ply] ?? [];
  if (killers[0] && sameAction(action, killers[0])) score += FIRST_KILLER_BONUS;
  else if (killers[1] && sameAction(action, killers[1])) score += SECOND_KILLER_BONUS;
  score += Math.min(100_000, context.historyHeuristic.get(actionKey(action, player)) ?? 0);

  const row = playClassicDrop(board, heights, column, player);
  if (row < 0) return -INF;
  try {
    if (hasWinFrom(board, row, column, player, context.connect)) return MATE_SCORE;
    if (ply <= 1) {
      let shape = evaluateShape(board, context.connect, context.aiPlayer);
      if (player !== context.aiPlayer) shape = -shape;
      score += shape;
    }
    return score;
  } finally {
    undoClassicDrop(board, heights, column, row);
  }
}

function orderedClassicColumns(board, heights, player, context, preferredAction = null, ply = 0) {
  const opponentWinningColumns = new Set(
    classicWinningColumns(board, heights, otherPlayer(player), context.connect),
  );
  const ordered = [];

  for (let column = 0; column < heights.length; column += 1) {
    if (heights[column] < 0) continue;
    ordered.push({
      column,
      score: classicColumnOrderScore(
        board,
        heights,
        column,
        player,
        context,
        preferredAction,
        opponentWinningColumns,
        ply,
      ),
    });
  }

  ordered.sort((first, second) => second.score - first.score);
  return ordered.map(({ column }) => column);
}

function classicQuiescence(
  board,
  heights,
  player,
  alpha,
  beta,
  ply,
  movesLeft,
  context,
  remainingDepth,
) {
  visitNode(context);

  const ownWins = classicWinningColumns(board, heights, player, context.connect);
  if (ownWins.length > 0) return mateScoreForWinner(player, context.aiPlayer, ply);

  const opponent = otherPlayer(player);
  const opponentWins = classicWinningColumns(board, heights, opponent, context.connect);
  if (opponentWins.length === 0 || remainingDepth <= 0) {
    return evaluateWithThreatCounts(
      board,
      context.connect,
      context.aiPlayer,
      player === context.aiPlayer ? ownWins.length : opponentWins.length,
      player === context.aiPlayer ? opponentWins.length : ownWins.length,
    );
  }

  const preferredAction = opponentWins.length === 1
    ? { type: ACTION_DROP, column: opponentWins[0] }
    : null;
  const columns = orderedClassicColumns(board, heights, player, context, preferredAction, ply);
  const maximizing = player === context.aiPlayer;
  let bestScore = maximizing ? -INF : INF;
  let foundDefence = false;

  for (const column of columns) {
    visitNode(context);
    const row = playClassicDrop(board, heights, column, player);
    if (row < 0) continue;

    let score = null;
    let isDefence = false;
    try {
      if (hasWinFrom(board, row, column, player, context.connect)) {
        score = mateScoreForWinner(player, context.aiPlayer, ply);
        isDefence = true;
      } else if (movesLeft <= 1) {
        score = 0;
        isDefence = true;
      } else {
        const remainingThreats = classicWinningColumns(
          board,
          heights,
          opponent,
          context.connect,
        );
        if (remainingThreats.length === 0) {
          isDefence = true;
          score = classicQuiescence(
            board,
            heights,
            opponent,
            alpha,
            beta,
            ply + 1,
            movesLeft - 1,
            context,
            remainingDepth - 1,
          );
        }
      }
    } finally {
      undoClassicDrop(board, heights, column, row);
    }

    if (!isDefence) continue;
    foundDefence = true;
    if (maximizing) {
      bestScore = Math.max(bestScore, score);
      alpha = Math.max(alpha, bestScore);
    } else {
      bestScore = Math.min(bestScore, score);
      beta = Math.min(beta, bestScore);
    }
    if (alpha >= beta) break;
  }

  if (!foundDefence) return mateScoreForWinner(opponent, context.aiPlayer, ply + 1);
  return bestScore;
}

function classicMinimax(
  board,
  heights,
  player,
  depth,
  alpha,
  beta,
  ply,
  movesLeft,
  context,
) {
  if (depth <= 0) {
    return classicQuiescence(
      board,
      heights,
      player,
      alpha,
      beta,
      ply,
      movesLeft,
      context,
      context.quiescenceDepth,
    );
  }
  visitNode(context);

  const alphaOriginal = alpha;
  const betaOriginal = beta;
  const tablePosition = canonicalTablePosition(board, player);
  const cached = context.transpositionTable.get(tablePosition.key);
  let preferredAction = null;

  if (cached) {
    context.tableHits += 1;
    preferredAction = tableActionToBoard(cached.bestAction, tablePosition);
    if (cached.depth >= depth) {
      if (cached.flag === 'exact') return cached.score;
      if (cached.flag === 'lower') alpha = Math.max(alpha, cached.score);
      if (cached.flag === 'upper') beta = Math.min(beta, cached.score);
      if (alpha >= beta) return cached.score;
    }
  }

  const columns = orderedClassicColumns(board, heights, player, context, preferredAction, ply);
  if (columns.length === 0) return 0;

  const maximizing = player === context.aiPlayer;
  let bestScore = maximizing ? -INF : INF;
  let bestColumn = columns[0];

  for (const column of columns) {
    visitNode(context);
    const row = playClassicDrop(board, heights, column, player);
    if (row < 0) continue;

    let score;
    try {
      if (hasWinFrom(board, row, column, player, context.connect)) {
        score = mateScoreForWinner(player, context.aiPlayer, ply);
      } else if (movesLeft <= 1) {
        score = 0;
      } else {
        score = classicMinimax(
          board,
          heights,
          otherPlayer(player),
          depth - 1,
          alpha,
          beta,
          ply + 1,
          movesLeft - 1,
          context,
        );
      }
    } finally {
      undoClassicDrop(board, heights, column, row);
    }

    if (maximizing) {
      if (score > bestScore) {
        bestScore = score;
        bestColumn = column;
      }
      alpha = Math.max(alpha, bestScore);
    } else {
      if (score < bestScore) {
        bestScore = score;
        bestColumn = column;
      }
      beta = Math.min(beta, bestScore);
    }

    if (alpha >= beta) {
      recordCutoff(context, { type: ACTION_DROP, column }, player, depth, ply);
      break;
    }
  }

  let flag = 'exact';
  if (bestScore <= alphaOriginal) flag = 'upper';
  else if (bestScore >= betaOriginal) flag = 'lower';

  const existing = context.transpositionTable.get(tablePosition.key);
  if (!existing || depth >= existing.depth) {
    if (context.transpositionTable.size >= MAX_TABLE_ENTRIES) {
      context.transpositionTable.clear();
      context.tableResets += 1;
    }
    context.transpositionTable.set(tablePosition.key, {
      depth,
      score: bestScore,
      flag,
      bestAction: boardActionToTable(
        { type: ACTION_DROP, column: bestColumn },
        tablePosition,
      ),
    });
  }

  return bestScore;
}

function classicSearchRoot(
  board,
  heights,
  player,
  depth,
  movesLeft,
  context,
  preferredAction,
  alpha = -INF,
  beta = INF,
) {
  const columns = orderedClassicColumns(board, heights, player, context, preferredAction, 0);
  if (columns.length === 0) return { action: null, score: 0 };

  const maximizing = player === context.aiPlayer;
  let bestAction = { type: ACTION_DROP, column: columns[0] };
  let bestScore = maximizing ? -INF : INF;

  for (const column of columns) {
    visitNode(context);
    const row = playClassicDrop(board, heights, column, player);
    if (row < 0) continue;

    let score;
    try {
      if (hasWinFrom(board, row, column, player, context.connect)) {
        score = mateScoreForWinner(player, context.aiPlayer, 0);
      } else if (movesLeft <= 1) {
        score = 0;
      } else {
        score = classicMinimax(
          board,
          heights,
          otherPlayer(player),
          depth - 1,
          alpha,
          beta,
          1,
          movesLeft - 1,
          context,
        );
      }
    } finally {
      undoClassicDrop(board, heights, column, row);
    }

    if ((maximizing && score > bestScore) || (!maximizing && score < bestScore)) {
      bestScore = score;
      bestAction = { type: ACTION_DROP, column };
    }

    if (maximizing) alpha = Math.max(alpha, bestScore);
    else beta = Math.min(beta, bestScore);
    if (alpha >= beta) break;
  }

  return { action: bestAction, score: bestScore };
}

function classicPrincipalVariation(position, firstAction, depth, context) {
  if (!firstAction) return [];
  const board = position.board.map((row) => [...row]);
  const heights = classicHeights(board);
  const variation = [];
  let player = position.currentPlayer;
  let action = { ...firstAction };

  for (let ply = 0; ply < depth && action; ply += 1) {
    const row = playClassicDrop(board, heights, action.column, player);
    if (row < 0) break;
    variation.push({ ...action });
    if (hasWinFrom(board, row, action.column, player, context.connect)) break;

    player = otherPlayer(player);
    const tablePosition = canonicalTablePosition(board, player);
    const cached = context.transpositionTable.get(tablePosition.key);
    action = cached?.bestAction
      ? tableActionToBoard(cached.bestAction, tablePosition)
      : null;
  }

  return variation;
}

function chooseClassicMove(position, options, defaults, aiPlayer, start) {
  const board = position.board.map((row) => [...row]);
  const heights = classicHeights(board);
  const movesLeft = emptyCellCount(board);
  const maximumDepth = Math.min(
    movesLeft,
    integerSearchOption(
      options.maximumDepth,
      defaults.maximumDepth,
      1,
      board.length * board[0].length,
      'Maximum search depth',
    ),
  );
  const context = {
    aiPlayer,
    connect: position.connect,
    chaosMode: false,
    quiescenceDepth: integerSearchOption(
      options.quiescenceDepth,
      defaults.quiescenceDepth,
      0,
      board.length * board[0].length,
      'Quiescence depth',
    ),
    nodes: 0,
    tableHits: 0,
    cutoffs: 0,
    tableResets: 0,
    transpositionTable: new Map(),
    historyHeuristic: new Map(),
    killers: [],
  };

  let best = {
    action: chooseEasy(position, () => 0.5),
    score: evaluateClassic(board, heights, position.connect, aiPlayer),
    depth: 0,
    principalVariation: [],
  };
  let preferredAction = best.action;

  for (let depth = 1; depth <= maximumDepth; depth += 1) {
    const result = aspirationSearch(depth, best.score, (alpha, beta) => classicSearchRoot(
      board,
      heights,
      position.currentPlayer,
      depth,
      movesLeft,
      context,
      preferredAction,
      alpha,
      beta,
    ));

    if (result.action) {
      best = {
        ...result,
        depth,
        principalVariation: classicPrincipalVariation(position, result.action, depth, context),
      };
      preferredAction = result.action;
      safeIterationCallback(options.onIteration, {
        ...best,
        nodes: context.nodes,
        elapsedMs: now() - start,
        tableHits: context.tableHits,
        cutoffs: context.cutoffs,
      });
    }
    if (Math.abs(result.score) >= MATE_SCORE - depth - 1) break;
  }

  return enforceTacticalSafety(position, {
    ...best,
    nodes: context.nodes,
    elapsedMs: now() - start,
    tableHits: context.tableHits,
    cutoffs: context.cutoffs,
    tableResets: context.tableResets,
  });
}

function emptyCellCount(board) {
  let count = 0;
  for (const row of board) {
    for (const cell of row) if (cell === EMPTY) count += 1;
  }
  return count;
}

function safeIterationCallback(callback, progress) {
  if (typeof callback !== 'function') return;
  try {
    callback(progress);
  } catch {
    // Progress reporting must never affect the search result.
  }
}

export function chooseMove(position, options = {}) {
  const counts = validateSearchPosition(position);
  const difficulty = options.difficulty ?? position.difficulty ?? 'medium';
  if (!VALID_DIFFICULTIES.has(difficulty)) {
    throw new RangeError(`Unknown AI difficulty: ${difficulty}`);
  }
  const aiPlayer = options.aiPlayer ?? position.currentPlayer;
  if (aiPlayer !== RED && aiPlayer !== YELLOW) {
    throw new RangeError('AI player must be Red or Yellow.');
  }
  const start = now();
  const finished = terminalResult(position, aiPlayer, start);
  if (finished) return finished;
  validateClassicTurn(position, counts);

  if (difficulty === 'easy') {
    const action = chooseEasy(position, options.random ?? Math.random);
    if (!action) {
      return {
        action: null,
        score: 0,
        depth: 0,
        nodes: 0,
        elapsedMs: now() - start,
        tableHits: 0,
        cutoffs: 0,
        tableResets: 0,
        principalVariation: [],
      };
    }
    const result = applyAction(position.board, action, position.currentPlayer);
    const outcome = actionOutcome(result, action, position.currentPlayer, position.connect);
    const score = terminalScore(outcome, aiPlayer, 0)
      ?? evaluateBoard(result.board, position.connect, aiPlayer);
    return {
      action,
      score,
      depth: 1,
      nodes: 0,
      elapsedMs: now() - start,
      tableHits: 0,
      cutoffs: 0,
      tableResets: 0,
      principalVariation: [{ ...action }],
    };
  }

  if (isBitboardPosition(position)) {
    return chooseBitboardMove(position, {
      ...options,
      difficulty,
      aiPlayer,
    });
  }
  if (difficulty === 'perfect') {
    throw new RangeError('Perfect AI requires classic 7×6 Connect Four.');
  }

  const defaults = DIFFICULTY[difficulty] ?? DIFFICULTY.medium;
  if (!position.chaosMode) {
    return chooseClassicMove(position, options, defaults, aiPlayer, start);
  }

  const boardCells = position.board.length * position.board[0].length;
  const maximumDepth = integerSearchOption(
    options.maximumDepth,
    defaults.chaosMaximumDepth,
    1,
    boardCells,
    'Maximum search depth',
  );
  const repetitions = copyRepetitionCounts(position.repetitionCounts);
  const context = {
    aiPlayer,
    connect: position.connect,
    chaosMode: true,
    quiescenceDepth: integerSearchOption(
      options.quiescenceDepth,
      defaults.quiescenceDepth,
      0,
      boardCells,
      'Quiescence depth',
    ),
    nodes: 0,
    tableHits: 0,
    cutoffs: 0,
    tableResets: 0,
    historyHeuristic: new Map(),
    killers: [],
  };

  let best = {
    action: chooseEasy(position, () => 0.5),
    score: evaluateBoard(position.board, position.connect, aiPlayer),
    depth: 0,
    principalVariation: [],
  };
  let preferredAction = best.action;

  for (let depth = 1; depth <= maximumDepth; depth += 1) {
    const result = aspirationSearch(depth, best.score, (alpha, beta) => searchRoot(
      position,
      depth,
      repetitions,
      context,
      preferredAction,
      alpha,
      beta,
    ));

    if (result.action) {
      best = { ...result, depth, principalVariation: principalVariation(result.action) };
      preferredAction = result.action;
      safeIterationCallback(options.onIteration, {
        ...best,
        nodes: context.nodes,
        elapsedMs: now() - start,
        tableHits: context.tableHits,
        cutoffs: context.cutoffs,
      });
    }
    if (Math.abs(result.score) >= MATE_SCORE - depth - 1) break;
  }

  return enforceTacticalSafety(position, {
    ...best,
    nodes: context.nodes,
    elapsedMs: now() - start,
    tableHits: context.tableHits,
    cutoffs: context.cutoffs,
    tableResets: context.tableResets,
  });
}
