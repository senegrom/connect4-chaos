import test from 'node:test';
import assert from 'node:assert/strict';

import { chooseMove, evaluateBoard } from '../src/ai.js';
import {
  ACTION_DROP,
  ACTION_FLIP,
  ACTION_ROTATE_CW,
  RED,
  YELLOW,
  applyAction,
  getDropRow,
  hasWinFrom,
  positionKey,
} from '../src/engine.js';

function emptyBoard(rows = 6, cols = 7) {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

function winningDropColumns(board, player, connect = 4) {
  const columns = [];
  for (let column = 0; column < board[0].length; column += 1) {
    const row = getDropRow(board, column);
    if (row < 0) continue;
    board[row][column] = player;
    const wins = hasWinFrom(board, row, column, player, connect);
    board[row][column] = 0;
    if (wins) columns.push(column);
  }
  return columns;
}

function position(board, overrides = {}) {
  const currentPlayer = overrides.currentPlayer ?? YELLOW;
  const connect = overrides.connect ?? 4;
  const chaosMode = overrides.chaosMode ?? false;
  return {
    board,
    currentPlayer,
    connect,
    chaosMode,
    repetitionCounts: [[positionKey(board, currentPlayer, connect, chaosMode), 1]],
  };
}

test('easy AI takes an immediate win', () => {
  const board = emptyBoard();
  board[5] = [YELLOW, YELLOW, YELLOW, 0, RED, RED, RED];

  const result = chooseMove(position(board), { difficulty: 'easy', random: () => 0 });
  assert.deepEqual(result.action, { type: ACTION_DROP, column: 3 });
});

test('easy AI blocks an immediate human win', () => {
  const board = emptyBoard();
  board[5] = [RED, RED, RED, 0, YELLOW, YELLOW, 0];

  const result = chooseMove(position(board), { difficulty: 'easy', random: () => 0 });
  assert.deepEqual(result.action, { type: ACTION_DROP, column: 3 });
});

test('easy AI recognises a winning chaos transform', () => {
  const board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, RED, 0, 0],
    [0, 0, YELLOW, YELLOW, RED, 0, 0],
    [YELLOW, YELLOW, RED, RED, RED, 0, 0],
  ];

  const result = chooseMove(position(board, { chaosMode: true }), {
    difficulty: 'easy',
    random: () => 0,
  });
  assert.deepEqual(result.action, { type: ACTION_FLIP });
});

test('depth-limited AI completes the requested search depth', () => {
  const board = emptyBoard();
  const result = chooseMove(position(board), {
    difficulty: 'hard',
    maximumDepth: 5,
  });

  assert.ok(result.action);
  assert.equal(result.action.type, ACTION_DROP);
  assert.ok(result.action.column >= 0 && result.action.column < 7);
  assert.ok(result.depth >= 1);
  assert.ok(result.nodes > 0);
});

test('search depth is not curtailed by a legacy time budget option', () => {
  const board = emptyBoard(4, 4);
  const result = chooseMove(position(board, { connect: 3 }), { difficulty: 'medium', timeBudgetMs: 0, maximumDepth: 4 });
  assert.equal(result.depth, 4);
  assert.ok(result.action);
});

test('standard 7x6 classic positions route through the bitboard solver', () => {
  const result = chooseMove(position(emptyBoard()), {
    difficulty: 'medium',
    maximumDepth: 4,
  });
  assert.equal(result.solver, 'bitboard');
  assert.equal(result.depth, 4);
  assert.deepEqual(result.action, { type: ACTION_DROP, column: 3 });
});

test('the evaluator rewards central control for the AI', () => {
  const centreBoard = emptyBoard();
  const edgeBoard = emptyBoard();
  centreBoard[5][3] = YELLOW;
  edgeBoard[5][0] = YELLOW;

  assert.ok(evaluateBoard(centreBoard, 4, YELLOW) > evaluateBoard(edgeBoard, 4, YELLOW));
});

test('the selected move can be applied to the searched position', () => {
  const board = emptyBoard();
  board[5][3] = RED;
  const result = chooseMove(position(board), {
    difficulty: 'medium',
    maximumDepth: 4,
  });

  const applied = applyAction(board, result.action, YELLOW);
  assert.ok(applied);
  assert.equal(applied.board.flat().filter((cell) => cell === YELLOW).length, 1);
});

test('tactical extension rejects a horizon move that concedes an immediate win', () => {
  const board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, YELLOW, 0, RED, RED, 0, 0],
    [YELLOW, RED, YELLOW, RED, YELLOW, 0, RED],
    [RED, RED, YELLOW, YELLOW, YELLOW, 0, RED],
    [YELLOW, YELLOW, RED, RED, RED, 0, YELLOW],
  ];

  const result = chooseMove(position(board), {
    difficulty: 'medium',
    maximumDepth: 1,
  });
  const next = applyAction(board, result.action, YELLOW);

  assert.deepEqual(result.action, { type: ACTION_DROP, column: 5 });
  assert.deepEqual(winningDropColumns(next.board, RED), []);
});

test('searched AI never concedes an immediate win when a safe move exists', () => {
  const board = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,YELLOW,0,RED,RED,0,0],[YELLOW,RED,YELLOW,RED,YELLOW,0,RED],[RED,RED,YELLOW,YELLOW,YELLOW,0,RED],[YELLOW,YELLOW,RED,RED,RED,0,YELLOW]];
  for (const difficulty of ['medium', 'hard']) {
    const result = chooseMove(position(board), { difficulty, maximumDepth: 2 });
    const next = applyAction(board, result.action, YELLOW);
    assert.deepEqual(winningDropColumns(next.board, RED), [], difficulty);
  }
});

test('the evaluator values a playable threat above the same floating shape', () => {
  const playable = emptyBoard();
  const floating = emptyBoard();
  playable[4] = [YELLOW, YELLOW, YELLOW, 0, 0, 0, 0];
  playable[5] = [RED, YELLOW, RED, RED, 0, 0, 0];
  floating[4] = [YELLOW, YELLOW, YELLOW, 0, 0, 0, 0];
  floating[5] = [RED, YELLOW, RED, 0, 0, 0, RED];

  assert.ok(evaluateBoard(playable, 4, YELLOW) > evaluateBoard(floating, 4, YELLOW));
});

test('iterative deepening reports completed depths and a principal variation', () => {
  const board = emptyBoard();
  board[5][3] = RED;
  const progress = [];
  const result = chooseMove(position(board), {
    difficulty: 'medium',
    maximumDepth: 6,
    onIteration(update) {
      progress.push(update);
    },
  });

  assert.ok(progress.length >= 1);
  assert.deepEqual(progress.map((update) => update.depth),
    [...progress.map((update) => update.depth)].sort((a, b) => a - b));
  assert.equal(progress.at(-1).depth, result.depth);
  assert.deepEqual(progress.at(-1).action, result.action);
  assert.ok(result.principalVariation.length >= 1);
  assert.deepEqual(result.principalVariation[0], result.action);
});

test('classic search leaves the caller board unchanged', () => {
  const board = emptyBoard();
  board[5] = [RED, YELLOW, RED, YELLOW, 0, 0, 0];
  const before = board.map((row) => [...row]);

  const result = chooseMove(position(board), {
    difficulty: 'brutal',
    maximumDepth: 6,
  });

  assert.ok(result.action);
  assert.deepEqual(board, before);
});

test('searched chaos positions still return a legal action without mutation', () => {
  const board = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, RED, 0, 0],
    [YELLOW, RED, YELLOW, 0],
  ];
  const before = board.map((row) => [...row]);

  const result = chooseMove(position(board, { chaosMode: true }), {
    difficulty: 'medium',
    maximumDepth: 2,
  });
  const applied = applyAction(board, result.action, YELLOW);

  assert.ok(applied);
  assert.deepEqual(board, before);
});

test('Brutal Chaos search does not trade opening drop depth for a horizon rotation', () => {
  const board = emptyBoard();
  board[5] = [YELLOW, RED, 0, RED, RED, 0, YELLOW];

  const result = chooseMove(position(board, {
    currentPlayer: YELLOW,
    chaosMode: true,
  }), {
    difficulty: 'brutal',
    maximumDepth: 3,
    quiescenceDepth: 2,
    chaosExactEmptyThreshold: 0,
  });

  assert.deepEqual(result.action, { type: ACTION_DROP, column: 2 });
  assert.equal(result.depth, 3);
  assert.equal(result.transformVerification, undefined);
});

test('a quiet Chaos transform at the nominal horizon is verified one drop deeper', () => {
  const board = [
    [0, 0, RED, 0],
    [0, 0, YELLOW, 0],
    [0, YELLOW, RED, 0],
    [0, RED, YELLOW, 0],
  ];

  const result = chooseMove(position(board, {
    currentPlayer: RED,
    connect: 3,
    chaosMode: true,
  }), {
    difficulty: 'brutal',
    maximumDepth: 4,
    quiescenceDepth: 2,
    chaosExactEmptyThreshold: 0,
  });

  assert.deepEqual(result.action, { type: ACTION_DROP, column: 0 });
  assert.equal(result.depth, 5);
  assert.equal(result.transformVerification, true);
});

test('Chaos transposition caching preserves fixed-depth results and includes repetition history', () => {
  const board = emptyBoard();
  const searchPosition = position(board, {
    currentPlayer: RED,
    chaosMode: true,
  });
  searchPosition.repetitionCounts.push([
    positionKey(emptyBoard(7, 6), YELLOW, 4, true),
    2,
  ]);
  const common = {
    difficulty: 'brutal',
    aiPlayer: RED,
    maximumDepth: 6,
    quiescenceDepth: 2,
    chaosExactEmptyThreshold: 0,
  };

  const cached = chooseMove(searchPosition, {
    ...common,
    useChaosTranspositionTable: true,
  });
  const uncached = chooseMove(searchPosition, {
    ...common,
    useChaosTranspositionTable: false,
  });

  assert.deepEqual(cached.action, uncached.action);
  assert.equal(cached.score, uncached.score);
  assert.equal(cached.depth, uncached.depth);
  assert.ok(cached.tableHits > 0);
  assert.equal(uncached.tableHits, 0);
  assert.ok(cached.nodes <= uncached.nodes);
  assert.throws(
    () => chooseMove(searchPosition, {
      ...common,
      useChaosTranspositionTable: 'yes',
    }),
    /must be a boolean/,
  );
});

test('Perfect AI rejects configurable and Chaos positions rather than using a heuristic fallback', () => {
  assert.throws(
    () => chooseMove(position(Array.from({ length: 4 }, () => Array(4).fill(0)), {
      connect: 4,
      chaosMode: false,
    }), { difficulty: 'perfect' }),
    /requires classic 7×6/,
  );
  assert.throws(
    () => chooseMove(position(emptyBoard(), { chaosMode: true }), { difficulty: 'perfect' }),
    /Perfect Chaos play is currently exact only/,
  );
});

function exactChaosFixture() {
  return [
    [RED, RED, RED, YELLOW, RED, 0, 0],
    [YELLOW, YELLOW, YELLOW, RED, YELLOW, 0, 0],
    [YELLOW, RED, YELLOW, RED, YELLOW, RED, 0],
    [YELLOW, RED, RED, RED, YELLOW, YELLOW, 0],
    [RED, YELLOW, YELLOW, YELLOW, RED, YELLOW, YELLOW],
    [RED, RED, YELLOW, YELLOW, RED, RED, RED],
  ];
}

test('Perfect Chaos uses retrograde exact play inside the verified endgame frontier', () => {
  const board = exactChaosFixture();
  const result = chooseMove(position(board, {
    currentPlayer: RED,
    chaosMode: true,
  }), { difficulty: 'perfect' });

  assert.equal(result.solver, 'chaos-exact-graph');
  assert.equal(result.solved, true);
  assert.equal(result.score, 1);
  assert.deepEqual(result.action, { type: ACTION_ROTATE_CW });
  assert.ok(result.nodes > 100);
});

test('searched Chaos levels automatically use the verified exact frontier', () => {
  for (const difficulty of ['medium', 'hard', 'brutal']) {
    const board = exactChaosFixture();
    const result = chooseMove(position(board, {
      currentPlayer: RED,
      chaosMode: true,
    }), { difficulty });
    assert.equal(result.solver, 'chaos-exact-graph', difficulty);
    assert.deepEqual(result.action, { type: ACTION_ROTATE_CW }, difficulty);
  }
});

test('Perfect Chaos fails closed when repetition history or graph limits invalidate the proof', () => {
  const board = exactChaosFixture();
  const repeated = position(board, { currentPlayer: RED, chaosMode: true });
  repeated.repetitionCounts = [[positionKey(board, RED, 4, true), 2]];
  assert.throws(
    () => chooseMove(repeated, { difficulty: 'perfect' }),
    /verified endgame frontier/,
  );
  assert.throws(
    () => chooseMove(position(board, { currentPlayer: RED, chaosMode: true }), {
      difficulty: 'perfect',
      chaosMaximumStates: 10,
    }),
    (error) => error?.code === 'CHAOS_GRAPH_LIMIT',
  );

  const fallback = chooseMove(repeated, {
    difficulty: 'brutal',
    maximumDepth: 1,
  });
  assert.notEqual(fallback.solver, 'chaos-exact-graph');
});

test('terminal positions return no move instead of entering search', () => {
  const board = emptyBoard();
  board[5] = [RED, RED, RED, RED, YELLOW, YELLOW, YELLOW];
  const result = chooseMove(position(board), { difficulty: 'brutal' });
  assert.equal(result.action, null);
  assert.equal(result.solver, 'terminal');
  assert.equal(result.solved, true);
  assert.ok(result.score < 0);
});

test('search boundaries reject malformed boards and unsafe options', () => {
  assert.throws(
    () => chooseMove({ board: [[0], [0, 0]], currentPlayer: RED, connect: 1 }),
    /rectangle/,
  );
  assert.throws(
    () => chooseMove({ board: [[0, 3]], currentPlayer: RED, connect: 1 }),
    /empty, Red, or Yellow/,
  );
  assert.throws(
    () => chooseMove(position([[RED], [0]], { currentPlayer: YELLOW, connect: 1 })),
    /obey gravity/,
  );
  assert.throws(
    () => chooseMove(position(emptyBoard(4, 4), { connect: 3 }), {
      difficulty: 'medium',
      maximumDepth: 0,
    }),
    /Maximum search depth/,
  );
  assert.throws(
    () => chooseMove(position(emptyBoard(4, 4), { connect: 3, chaosMode: true }), {
      difficulty: 'medium',
      quiescenceDepth: Number.NaN,
    }),
    /Quiescence depth/,
  );
  assert.throws(
    () => chooseMove(position(emptyBoard(4, 4), { connect: 3, chaosMode: true }), {
      difficulty: 'medium',
      chaosTransformBudget: -1,
    }),
    /Chaos transform budget/,
  );
  assert.throws(
    () => chooseMove(position(emptyBoard()), { difficulty: 'impossible' }),
    /Unknown AI difficulty/,
  );
  assert.throws(
    () => chooseMove(position(emptyBoard(), { chaosMode: true }), {
      difficulty: 'medium',
      chaosExactEmptyThreshold: 43,
    }),
    /Chaos exact empty threshold/,
  );
  assert.throws(
    () => chooseMove(position(emptyBoard(), { chaosMode: true }), {
      difficulty: 'medium',
      chaosMaximumStates: 2_000_001,
    }),
    /Chaos exact state limit/,
  );
  assert.throws(
    () => chooseMove({
      ...position(emptyBoard(4, 4), { connect: 3, chaosMode: true }),
      repetitionCounts: [['position', '2']],
    }, { difficulty: 'easy' }),
    /non-negative integers/,
  );
  assert.throws(
    () => chooseMove({
      ...position(emptyBoard(4, 4), { connect: 3, chaosMode: true }),
      repetitionCounts: 1,
    }, { difficulty: 'easy' }),
    /map, entry array, or object/,
  );
  const twoReds = emptyBoard(4, 4);
  twoReds[3][0] = RED;
  twoReds[3][1] = RED;
  assert.throws(
    () => chooseMove(position(twoReds, { currentPlayer: YELLOW, connect: 3 })),
    /differ by more than one piece/,
  );

  const wrongTurn = emptyBoard(4, 4);
  wrongTurn[3][0] = RED;
  assert.throws(
    () => chooseMove(position(wrongTurn, { currentPlayer: RED, connect: 3 })),
    /side to move/,
  );

  const winnerRetained = emptyBoard();
  winnerRetained[5] = [RED, RED, RED, RED, YELLOW, YELLOW, YELLOW];
  winnerRetained[4][4] = YELLOW;
  const retained = chooseMove(position(winnerRetained, { currentPlayer: RED }));
  assert.equal(retained.action, null);
  assert.equal(retained.solver, 'terminal');
  assert.ok(retained.score > 0);

  const bothWin = emptyBoard();
  bothWin[5] = [RED, RED, RED, RED, YELLOW, YELLOW, YELLOW];
  bothWin[4] = [YELLOW, YELLOW, YELLOW, YELLOW, RED, RED, RED];
  assert.throws(
    () => chooseMove(position(bothWin, { currentPlayer: RED })),
    /both players/,
  );
});

test('Easy AI validates custom random sources', () => {
  assert.throws(
    () => chooseMove(position(emptyBoard()), { difficulty: 'easy', random: 1 }),
    /must be a function/,
  );
  assert.throws(
    () => chooseMove(position(emptyBoard()), {
      difficulty: 'easy',
      random: () => Number.NaN,
    }),
    /finite number/,
  );
  const result = chooseMove(position(emptyBoard()), {
    difficulty: 'easy',
    random: () => 1,
  });
  assert.ok(result.action);
});
