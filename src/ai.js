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
const TIME_CHECK_MASK = 255;

const DIFFICULTY = Object.freeze({
  medium: { timeBudgetMs: 250, maximumDepth: 5 },
  hard: { timeBudgetMs: 900, maximumDepth: 8 },
  brutal: { timeBudgetMs: 2_500, maximumDepth: 12 },
});

class SearchTimeout extends Error {
  constructor() {
    super('Search time limit reached.');
    this.name = 'SearchTimeout';
  }
}

function now() {
  return globalThis.performance?.now?.() ?? Date.now();
}

function checkTime(context) {
  context.nodes += 1;
  if ((context.nodes & TIME_CHECK_MASK) === 0 && now() >= context.deadline) {
    throw new SearchTimeout();
  }
}

function copyRepetitionCounts(entries) {
  if (entries instanceof Map) return new Map(entries);
  if (Array.isArray(entries)) return new Map(entries);
  if (entries && typeof entries === 'object') return new Map(Object.entries(entries));
  return new Map();
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

function centralityScore(column, cols) {
  const centre = (cols - 1) / 2;
  return Math.round((cols - Math.abs(column - centre)) * 10);
}

function immediateWinningActions(board, player, connect, chaosMode) {
  const wins = [];
  for (const action of legalActions(board, chaosMode)) {
    const result = applyAction(board, action, player);
    if (!result) continue;
    const outcome = actionOutcome(result, action, player, connect);
    if (outcome.status === 'won' && outcome.winner === player) wins.push(action);
  }
  return wins;
}

function chooseEasy(position, random = Math.random) {
  const {
    board,
    connect,
    currentPlayer,
    chaosMode,
  } = position;
  const opponent = otherPlayer(currentPlayer);

  const ownWins = immediateWinningActions(board, currentPlayer, connect, chaosMode);
  if (ownWins.length > 0) {
    return ownWins[Math.floor(random() * ownWins.length)];
  }

  const opponentWinningDrops = new Set(
    immediateWinningActions(board, opponent, connect, false)
      .filter((action) => action.type === ACTION_DROP)
      .map((action) => action.column),
  );

  if (opponentWinningDrops.size > 0) {
    const blocks = legalActions(board, false)
      .filter((action) => opponentWinningDrops.has(action.column));
    if (blocks.length > 0) return blocks[Math.floor(random() * blocks.length)];
  }

  const actions = legalActions(board, chaosMode);
  if (actions.length === 0) return null;

  const { cols } = boardDimensions(board);
  const weighted = [];
  for (const action of actions) {
    let weight = 2;
    if (action.type === ACTION_DROP) {
      weight = Math.max(2, Math.round(cols - Math.abs(action.column - (cols - 1) / 2)));
    }
    for (let index = 0; index < weight; index += 1) weighted.push(action);
  }

  return weighted[Math.floor(random() * weighted.length)];
}

function windowScore(cells, connect, aiPlayer) {
  let aiCount = 0;
  let opponentCount = 0;
  let emptyCount = 0;

  for (const cell of cells) {
    if (cell === aiPlayer) aiCount += 1;
    else if (cell === EMPTY) emptyCount += 1;
    else opponentCount += 1;
  }

  if (aiCount > 0 && opponentCount > 0) return 0;
  if (emptyCount === connect) return 0;

  const weights = [0, 2, 12, 70, 420, 2_500, 15_000];
  if (aiCount > 0) return weights[Math.min(aiCount, weights.length - 1)];
  if (opponentCount > 0) {
    return -Math.round(weights[Math.min(opponentCount, weights.length - 1)] * 1.12);
  }
  return 0;
}

export function evaluateBoard(board, connect, aiPlayer = YELLOW) {
  const { rows, cols } = boardDimensions(board);
  let score = 0;

  const centre = (cols - 1) / 2;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < cols; column += 1) {
      const cell = board[row][column];
      if (cell === EMPTY) continue;
      const value = Math.max(1, Math.round(cols / 2 - Math.abs(column - centre) + 1));
      score += cell === aiPlayer ? value : -value;
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

        const cells = [];
        for (let offset = 0; offset < connect; offset += 1) {
          cells.push(board[row + offset * deltaRow][column + offset * deltaColumn]);
        }
        score += windowScore(cells, connect, aiPlayer);
      }
    }
  }

  return score;
}

function childOrderScore(child, player, aiPlayer, connect) {
  const terminal = terminalScore(child.outcome, aiPlayer, 1);
  if (terminal !== null) return terminal;

  let score = evaluateBoard(child.result.board, connect, aiPlayer);
  const { cols } = boardDimensions(child.result.board);
  if (child.action.type === ACTION_DROP) {
    const centreBonus = centralityScore(child.action.column, cols);
    score += player === aiPlayer ? centreBonus : -centreBonus;
  } else {
    score += player === aiPlayer ? -5 : 5;
  }
  return score;
}

function makeChildren(board, player, context, preferredAction = null) {
  const actions = legalActions(board, context.chaosMode);
  const children = [];
  const seenPositions = new Set();

  for (const action of actions) {
    const result = applyAction(board, action, player);
    if (!result) continue;

    const nextPosition = `${otherPlayer(player)}|${boardToString(result.board)}`;
    if (seenPositions.has(nextPosition)) continue;
    seenPositions.add(nextPosition);

    const outcome = actionOutcome(result, action, player, context.connect);
    const child = { action, result, outcome };
    child.orderScore = childOrderScore(child, player, context.aiPlayer, context.connect);
    if (preferredAction && sameAction(action, preferredAction)) {
      child.orderScore += player === context.aiPlayer ? MATE_SCORE / 2 : -MATE_SCORE / 2;
    }
    children.push(child);
  }

  children.sort((first, second) => (
    player === context.aiPlayer
      ? second.orderScore - first.orderScore
      : first.orderScore - second.orderScore
  ));
  return children;
}

function transpositionKey(board, player, depth) {
  const { rows, cols } = boardDimensions(board);
  return `${player}|${depth}|${rows}x${cols}|${boardToString(board)}`;
}

function minimax(board, player, depth, alpha, beta, ply, repetitions, context) {
  checkTime(context);
  if (depth <= 0) return evaluateBoard(board, context.connect, context.aiPlayer);

  const alphaOriginal = alpha;
  const betaOriginal = beta;
  const useTable = !context.chaosMode;
  const key = useTable ? transpositionKey(board, player, depth) : null;

  if (key) {
    const cached = context.transpositionTable.get(key);
    if (cached) {
      if (cached.flag === 'exact') return cached.score;
      if (cached.flag === 'lower') alpha = Math.max(alpha, cached.score);
      if (cached.flag === 'upper') beta = Math.min(beta, cached.score);
      if (alpha >= beta) return cached.score;
    }
  }

  const children = makeChildren(board, player, context);
  if (children.length === 0) return 0;

  const maximizing = player === context.aiPlayer;
  let bestScore = maximizing ? -INF : INF;

  for (const child of children) {
    checkTime(context);
    const immediateScore = terminalScore(child.outcome, context.aiPlayer, ply);
    let score;

    if (immediateScore !== null) {
      score = immediateScore;
    } else {
      const nextPlayer = otherPlayer(player);
      const repetitionKey = positionKey(
        child.result.board,
        nextPlayer,
        context.connect,
        context.chaosMode,
      );
      const repetitionCount = incrementRepetition(repetitions, repetitionKey);

      if (repetitionCount >= 3) {
        score = 0;
      } else {
        score = minimax(
          child.result.board,
          nextPlayer,
          depth - 1,
          alpha,
          beta,
          ply + 1,
          repetitions,
          context,
        );
      }
      decrementRepetition(repetitions, repetitionKey);
    }

    if (maximizing) {
      bestScore = Math.max(bestScore, score);
      alpha = Math.max(alpha, bestScore);
    } else {
      bestScore = Math.min(bestScore, score);
      beta = Math.min(beta, bestScore);
    }

    if (alpha >= beta) break;
  }

  if (key) {
    let flag = 'exact';
    if (bestScore <= alphaOriginal) flag = 'upper';
    else if (bestScore >= betaOriginal) flag = 'lower';
    context.transpositionTable.set(key, { score: bestScore, flag });
  }

  return bestScore;
}

function searchRoot(position, depth, repetitions, context, preferredAction) {
  const children = makeChildren(position.board, position.currentPlayer, context, preferredAction);
  if (children.length === 0) return { action: null, score: 0 };

  let alpha = -INF;
  let beta = INF;
  let bestAction = children[0].action;
  let bestScore = position.currentPlayer === context.aiPlayer ? -INF : INF;

  for (const child of children) {
    checkTime(context);
    const immediateScore = terminalScore(child.outcome, context.aiPlayer, 0);
    let score;

    if (immediateScore !== null) {
      score = immediateScore;
    } else {
      const nextPlayer = otherPlayer(position.currentPlayer);
      const repetitionKey = positionKey(
        child.result.board,
        nextPlayer,
        context.connect,
        context.chaosMode,
      );
      const repetitionCount = incrementRepetition(repetitions, repetitionKey);

      if (repetitionCount >= 3) {
        score = 0;
      } else {
        score = minimax(
          child.result.board,
          nextPlayer,
          depth - 1,
          alpha,
          beta,
          1,
          repetitions,
          context,
        );
      }
      decrementRepetition(repetitions, repetitionKey);
    }

    const maximizing = position.currentPlayer === context.aiPlayer;
    if ((maximizing && score > bestScore) || (!maximizing && score < bestScore)) {
      bestScore = score;
      bestAction = child.action;
    }

    if (maximizing) alpha = Math.max(alpha, bestScore);
    else beta = Math.min(beta, bestScore);
  }

  return { action: bestAction, score: bestScore };
}

export function chooseMove(position, options = {}) {
  if (!position?.board || !position.currentPlayer) {
    throw new TypeError('A position with a board and currentPlayer is required.');
  }

  const difficulty = options.difficulty ?? position.difficulty ?? 'medium';
  const aiPlayer = options.aiPlayer ?? position.currentPlayer;
  const start = now();

  if (difficulty === 'easy') {
    const action = chooseEasy(position, options.random ?? Math.random);
    if (!action) return { action: null, score: 0, depth: 0, nodes: 0, elapsedMs: now() - start };
    const result = applyAction(position.board, action, position.currentPlayer);
    const outcome = actionOutcome(result, action, position.currentPlayer, position.connect);
    const score = terminalScore(outcome, aiPlayer, 0)
      ?? evaluateBoard(result.board, position.connect, aiPlayer);
    return { action, score, depth: 1, nodes: 0, elapsedMs: now() - start };
  }

  const defaults = DIFFICULTY[difficulty] ?? DIFFICULTY.medium;
  const timeBudgetMs = Math.max(10, options.timeBudgetMs ?? defaults.timeBudgetMs);
  const maximumDepth = Math.max(1, options.maximumDepth ?? defaults.maximumDepth);
  const repetitions = copyRepetitionCounts(position.repetitionCounts);
  const context = {
    aiPlayer,
    connect: position.connect,
    chaosMode: Boolean(position.chaosMode),
    deadline: start + timeBudgetMs,
    nodes: 0,
    transpositionTable: new Map(),
  };

  let best = {
    action: chooseEasy(position, () => 0.5),
    score: evaluateBoard(position.board, position.connect, aiPlayer),
    depth: 0,
  };
  let preferredAction = best.action;

  for (let depth = 1; depth <= maximumDepth; depth += 1) {
    try {
      const result = searchRoot(position, depth, repetitions, context, preferredAction);
      if (result.action) {
        best = { ...result, depth };
        preferredAction = result.action;
      }
      if (Math.abs(result.score) >= MATE_SCORE - maximumDepth) break;
    } catch (error) {
      if (error instanceof SearchTimeout) break;
      throw error;
    }
  }

  return {
    ...best,
    nodes: context.nodes,
    elapsedMs: now() - start,
  };
}

export const AI_ACTION_TYPES = Object.freeze([
  ACTION_DROP,
  ACTION_FLIP,
  ACTION_ROTATE_CW,
  ACTION_ROTATE_CCW,
]);
