import assert from 'node:assert/strict';
import test from 'node:test';

import {
  CHAOS_DRAW,
  CHAOS_LOSS,
  CHAOS_WIN,
  buildChaosGraph,
  canonicalChaosPosition,
  mirrorChaosAction,
  solveChaosGraph,
  solveChaosPosition,
} from '../src/chaos-solver.js';
import {
  ACTION_DROP,
  ACTION_FLIP,
  ACTION_ROTATE_CCW,
  ACTION_ROTATE_CW,
  RED,
  YELLOW,
  applyAction,
  boardDimensions,
  boardToString,
  createBoard,
  legalActions,
  otherPlayer,
  positionKey,
  resolveActionOutcome,
} from '../src/engine.js';

function directedGraph(edgeLists) {
  const nodes = edgeLists.map((edges, index) => ({
    board: [],
    key: String(index),
    edges: edges.map((edge) => ({ ...edge })),
    predecessors: [],
  }));
  for (let parent = 0; parent < nodes.length; parent += 1) {
    for (let edge = 0; edge < nodes[parent].edges.length; edge += 1) {
      const child = nodes[parent].edges[edge].next;
      if (child >= 0) nodes[child].predecessors.push({ parent, edge });
    }
  }
  return { nodes, root: 0, rootMirrored: false, rootColumns: 1 };
}

function terminal(value, type = ACTION_FLIP) {
  return { action: { type }, terminal: value, next: -1 };
}

function successor(next, type = ACTION_FLIP) {
  return { action: { type }, terminal: null, next };
}

test('retrograde solving leaves closed cycles as draws', () => {
  const graph = directedGraph([
    [successor(1)],
    [successor(0)],
  ]);
  const solved = solveChaosGraph(graph);
  assert.deepEqual([...solved.values], [CHAOS_DRAW, CHAOS_DRAW]);
});

test('retrograde ranks choose a finite winning exit instead of cycling', () => {
  const graph = directedGraph([
    [successor(1), terminal(CHAOS_WIN, ACTION_DROP)],
    [successor(0)],
  ]);
  const solved = solveChaosGraph(graph);
  assert.deepEqual([...solved.values], [CHAOS_WIN, CHAOS_LOSS]);
  assert.deepEqual([...solved.ranks], [1, 2]);
  assert.equal(solved.bestEdges[0], 1);
});

test('losses propagate only after every action is proved losing', () => {
  const graph = directedGraph([
    [terminal(CHAOS_LOSS), successor(1)],
    [terminal(CHAOS_WIN)],
  ]);
  const solved = solveChaosGraph(graph);
  assert.deepEqual([...solved.values], [CHAOS_LOSS, CHAOS_WIN]);
  assert.deepEqual([...solved.ranks], [2, 1]);
});

function mirrorBoard(board) {
  return board.map((row) => [...row].reverse());
}

function actionResult(board, action) {
  return applyAction(board, action, RED).board;
}

test('horizontal symmetry maps every Chaos action correctly', () => {
  const board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, RED, 0, 0, 0],
    [YELLOW, RED, 0, YELLOW, 0, RED, 0],
  ];
  const actions = [
    { type: ACTION_DROP, column: 1 },
    { type: ACTION_FLIP },
    { type: ACTION_ROTATE_CW },
    { type: ACTION_ROTATE_CCW },
  ];

  for (const actionOnMirror of actions) {
    const mapped = mirrorChaosAction(actionOnMirror, board[0].length);
    const fromOriginal = mirrorBoard(actionResult(board, mapped));
    const fromMirror = actionResult(mirrorBoard(board), actionOnMirror);
    assert.equal(boardToString(fromOriginal), boardToString(fromMirror));
  }
});

function historyKey(board, player, connect, repetitions) {
  const counts = [...repetitions.entries()]
    .filter(([, count]) => count > 0)
    .sort(([first], [second]) => first.localeCompare(second));
  return `${positionKey(board, player, connect, true)}|${JSON.stringify(counts)}`;
}

function solveWithLiteralThreefold(board, player, connect, repetitions, memo) {
  const key = historyKey(board, player, connect, repetitions);
  if (memo.has(key)) return memo.get(key);

  let best = CHAOS_LOSS;
  for (const action of legalActions(board, true)) {
    const result = applyAction(board, action, player);
    const outcome = resolveActionOutcome(
      result.board,
      connect,
      player,
      action.type,
      action.type === ACTION_DROP ? { row: result.row, column: result.column } : null,
    );
    let value;
    if (outcome.status === 'won') value = outcome.winner === player ? CHAOS_WIN : CHAOS_LOSS;
    else if (outcome.status === 'draw') value = CHAOS_DRAW;
    else {
      const nextPlayer = otherPlayer(player);
      const nextKey = positionKey(result.board, nextPlayer, connect, true);
      const previous = repetitions.get(nextKey) ?? 0;
      if (previous + 1 >= 3) value = CHAOS_DRAW;
      else {
        repetitions.set(nextKey, previous + 1);
        try {
          value = -solveWithLiteralThreefold(
            result.board,
            nextPlayer,
            connect,
            repetitions,
            memo,
          );
        } finally {
          if (previous === 0) repetitions.delete(nextKey);
          else repetitions.set(nextKey, previous);
        }
      }
    }
    if (value > best) best = value;
    if (best === CHAOS_WIN) break;
  }
  memo.set(key, best);
  return best;
}

test('loopy retrograde value matches literal threefold play on a complete tiny game', () => {
  const board = createBoard(2, 2);
  const rootKey = positionKey(board, RED, 2, true);
  const literal = solveWithLiteralThreefold(
    board,
    RED,
    2,
    new Map([[rootKey, 1]]),
    new Map(),
  );
  const solved = solveChaosPosition({
    board,
    currentPlayer: RED,
    connect: 2,
    chaosMode: true,
  });
  assert.equal(solved.value, literal);
  assert.equal(solved.value, CHAOS_WIN);
});


function collectReachableFreshPositions(rows, cols, connect) {
  const queue = [{ board: createBoard(rows, cols), player: RED }];
  const seen = new Set();
  const positions = [];

  while (queue.length > 0) {
    const current = queue.shift();
    const key = `${current.player}|${current.board.length}x${current.board[0].length}|${boardToString(current.board)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    positions.push(current);

    for (const action of legalActions(current.board, true)) {
      const result = applyAction(current.board, action, current.player);
      const outcome = resolveActionOutcome(
        result.board,
        connect,
        current.player,
        action.type,
        action.type === ACTION_DROP ? { row: result.row, column: result.column } : null,
      );
      if (outcome.status === 'playing') {
        queue.push({ board: result.board, player: otherPlayer(current.player) });
      }
    }
  }

  return positions;
}

test('ranked retrograde agrees with literal threefold play on every reachable 2x3 state', () => {
  const positions = collectReachableFreshPositions(2, 3, 2);
  assert.equal(positions.length, 96);

  for (const { board, player } of positions) {
    const rootKey = positionKey(board, player, 2, true);
    const literal = solveWithLiteralThreefold(
      board,
      player,
      2,
      new Map([[rootKey, 1]]),
      new Map(),
    );
    const solved = solveChaosPosition({
      board,
      currentPlayer: player,
      connect: 2,
      chaosMode: true,
    });
    assert.equal(
      solved.value,
      literal,
      `value mismatch for ${player}|${boardToString(board)}`,
    );
  }
});

test('the empty 3x3 Connect-3 Chaos game is an exact draw', () => {
  const solved = solveChaosPosition({
    board: createBoard(3, 3),
    currentPlayer: RED,
    connect: 3,
    chaosMode: true,
  });
  assert.equal(solved.value, CHAOS_DRAW);
  assert.ok(solved.action);
  assert.equal(solved.graph.states, 628);
});

test('the selected exact strategy strictly reduces rank in won positions', () => {
  const graph = buildChaosGraph({
    board: createBoard(3, 4),
    currentPlayer: RED,
    connect: 3,
    chaosMode: true,
  });
  const solved = solveChaosGraph(graph);

  for (let index = 0; index < graph.nodes.length; index += 1) {
    const value = solved.values[index];
    const edgeIndex = solved.bestEdges[index];
    assert.ok(edgeIndex >= 0, `missing best edge for state ${index}`);
    const edge = graph.nodes[index].edges[edgeIndex];
    const childValue = edge.next >= 0 ? solved.values[edge.next] : null;
    const actionValue = edge.terminal ?? (childValue === CHAOS_DRAW ? CHAOS_DRAW : -childValue);
    assert.equal(actionValue, value);
    if (value === CHAOS_WIN && edge.next >= 0) {
      assert.equal(solved.values[edge.next], CHAOS_LOSS);
      assert.ok(solved.ranks[edge.next] < solved.ranks[index]);
    }
  }
});

test('canonical positions normalize the side to move and horizontal reflection', () => {
  const board = [
    [0, 0, 0],
    [0, 0, RED],
    [0, YELLOW, RED],
  ];
  const first = canonicalChaosPosition(board, RED);
  const swapped = board.map((row) => row.map((cell) => (
    cell === RED ? YELLOW : cell === YELLOW ? RED : 0
  )));
  const second = canonicalChaosPosition(swapped, YELLOW);
  assert.equal(first.key, second.key);
  assert.deepEqual(boardDimensions(first.board), { rows: 3, cols: 3 });
});

test('state limits fail closed before returning an unproved move', () => {
  assert.throws(
    () => solveChaosPosition({
      board: createBoard(4, 4),
      currentPlayer: RED,
      connect: 3,
      chaosMode: true,
    }, { maximumStates: 10 }),
    (error) => error?.code === 'CHAOS_GRAPH_LIMIT',
  );
});
