import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildChaosProofGraph,
  solveChaosProofPosition,
} from '../src/chaos-proof.js';
import {
  CHAOS_DRAW,
  CHAOS_LOSS,
  CHAOS_WIN,
  mirrorChaosAction,
  solveChaosPosition,
} from '../src/chaos-solver.js';
import {
  ACTION_DROP,
  ACTION_FLIP,
  RED,
  YELLOW,
  applyAction,
  boardToString,
  createBoard,
  legalActions,
  otherPlayer,
  resolveActionOutcome,
  sameAction,
} from '../src/engine.js';

function actionOutcome(board, action, player, connect) {
  const result = applyAction(board, action, player);
  const outcome = resolveActionOutcome(
    result.board,
    connect,
    player,
    action.type,
    action.type === ACTION_DROP ? { row: result.row, column: result.column } : null,
  );
  return { result, outcome };
}

function actionValue(board, action, player, connect, memo) {
  const { result, outcome } = actionOutcome(board, action, player, connect);
  if (outcome.status === 'draw') return CHAOS_DRAW;
  if (outcome.status === 'won') return outcome.winner === player ? CHAOS_WIN : CHAOS_LOSS;
  const nextPlayer = otherPlayer(player);
  const key = `${nextPlayer}|${boardToString(result.board)}`;
  let childValue = memo.get(key);
  if (childValue === undefined) {
    childValue = solveChaosPosition({
      board: result.board,
      currentPlayer: nextPlayer,
      connect,
      chaosMode: true,
    }).value;
    memo.set(key, childValue);
  }
  return childValue === CHAOS_DRAW ? CHAOS_DRAW : -childValue;
}

function collectReachablePositions(rows, cols, connect) {
  const queue = [{ board: createBoard(rows, cols), player: RED }];
  const seen = new Set();
  const positions = [];

  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const current = queue[cursor];
    const key = `${current.player}|${boardToString(current.board)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    positions.push(current);

    for (const action of legalActions(current.board, true)) {
      const { result, outcome } = actionOutcome(
        current.board,
        action,
        current.player,
        connect,
      );
      if (outcome.status === 'playing') {
        queue.push({ board: result.board, player: otherPlayer(current.player) });
      }
    }
  }
  return positions;
}

function mirrorBoard(board) {
  return board.map((row) => [...row].reverse());
}

function findBound(proof, action) {
  return proof.actionBounds.find((entry) => sameAction(entry.action, action));
}

test('a complete drop horizon converges to the exact loopy-game value', () => {
  for (const [rows, cols, connect, expected] of [
    [2, 2, 2, CHAOS_WIN],
    [3, 3, 3, CHAOS_DRAW],
  ]) {
    const board = createBoard(rows, cols);
    const proof = solveChaosProofPosition({
      board,
      currentPlayer: RED,
      connect,
      chaosMode: true,
    }, {
      dropDepth: rows * cols,
      maximumStates: 20_000,
    });
    const exact = solveChaosPosition({
      board,
      currentPlayer: RED,
      connect,
      chaosMode: true,
    });

    assert.equal(proof.solved, true);
    assert.equal(proof.lowerValue, expected);
    assert.equal(proof.upperValue, expected);
    assert.equal(proof.value, exact.value);
    assert.ok(proof.action);
    assert.equal(proof.graph.frontierEdges, 0);
  }
});

test('bounded values bracket exact root and action values on every reachable 2x3 state', () => {
  const positions = collectReachablePositions(2, 3, 2);
  const exactMemo = new Map();
  assert.equal(positions.length, 96);

  for (const { board, player } of positions) {
    const key = `${player}|${boardToString(board)}`;
    let exactValue = exactMemo.get(key);
    if (exactValue === undefined) {
      exactValue = solveChaosPosition({
        board,
        currentPlayer: player,
        connect: 2,
        chaosMode: true,
      }).value;
      exactMemo.set(key, exactValue);
    }

    for (const dropDepth of [1, 2, 3]) {
      const proof = solveChaosProofPosition({
        board,
        currentPlayer: player,
        connect: 2,
        chaosMode: true,
      }, { dropDepth, maximumStates: 20_000 });

      assert.ok(proof.lowerValue <= exactValue, `${key} depth ${dropDepth} lower bound`);
      assert.ok(exactValue <= proof.upperValue, `${key} depth ${dropDepth} upper bound`);
      if (proof.solved) assert.equal(proof.value, exactValue);

      for (const action of legalActions(board, true)) {
        const bound = findBound(proof, action);
        assert.ok(bound, `missing ${JSON.stringify(action)} at ${key}`);
        const exactActionValue = actionValue(board, action, player, 2, exactMemo);
        assert.ok(bound.lower <= exactActionValue, `${key} ${JSON.stringify(action)} lower`);
        assert.ok(exactActionValue <= bound.upper, `${key} ${JSON.stringify(action)} upper`);
      }
    }
  }
});

test('the two-drop proof rejects the known opening horizon blunders', () => {
  const board = createBoard(6, 7);
  board[5] = [YELLOW, RED, 0, RED, RED, 0, YELLOW];
  const before = board.map((row) => [...row]);
  const proof = solveChaosProofPosition({
    board,
    currentPlayer: YELLOW,
    connect: 4,
    chaosMode: true,
  }, { dropDepth: 2, maximumStates: 20_000 });

  assert.equal(proof.solved, false);
  assert.deepEqual(proof.action, { type: ACTION_DROP, column: 2 });
  assert.equal(findBound(proof, { type: ACTION_DROP, column: 2 }).upper, CHAOS_WIN);
  assert.equal(findBound(proof, { type: ACTION_DROP, column: 3 }).upper, CHAOS_LOSS);
  assert.equal(findBound(proof, { type: ACTION_FLIP }).upper, CHAOS_LOSS);
  assert.equal(proof.provenLosingActions.length, 7);
  assert.deepEqual(board, before);
});

test('horizontal reflection preserves bounds and maps the selected action', () => {
  const board = createBoard(6, 7);
  board[5] = [YELLOW, RED, 0, RED, RED, 0, YELLOW];
  const first = solveChaosProofPosition({
    board,
    currentPlayer: YELLOW,
    connect: 4,
    chaosMode: true,
  }, { dropDepth: 2, maximumStates: 20_000 });
  const second = solveChaosProofPosition({
    board: mirrorBoard(board),
    currentPlayer: YELLOW,
    connect: 4,
    chaosMode: true,
  }, { dropDepth: 2, maximumStates: 20_000 });

  assert.deepEqual(second.action, mirrorChaosAction(first.action, 7));
  for (const entry of first.actionBounds) {
    const mirrored = mirrorChaosAction(entry.action, 7);
    const counterpart = findBound(second, mirrored);
    assert.ok(counterpart);
    assert.equal(counterpart.lower, entry.lower);
    assert.equal(counterpart.upper, entry.upper);
  }
});

test('the graph tracks root-side parity and deterministic frontier metadata', () => {
  const graph = buildChaosProofGraph({
    board: createBoard(2, 3),
    currentPlayer: RED,
    connect: 2,
    chaosMode: true,
  }, { dropDepth: 2, maximumStates: 1_000 });

  assert.equal(graph.nodes[graph.root].aiTurn, true);
  assert.ok(graph.nodes.some((node) => node.aiTurn === false));
  assert.ok(graph.frontierEdges > 0);
  assert.ok(graph.frontierStates > 0);
  for (const node of graph.nodes) {
    for (const edge of node.edges) {
      if (!edge.frontier) continue;
      assert.equal(edge.frontierLower, node.aiTurn ? CHAOS_LOSS : CHAOS_WIN);
      assert.equal(edge.frontierUpper, node.aiTurn ? CHAOS_WIN : CHAOS_LOSS);
    }
  }
});

test('bounded proof limits fail closed', () => {
  assert.throws(
    () => solveChaosProofPosition({
      board: createBoard(4, 4),
      currentPlayer: RED,
      connect: 3,
      chaosMode: true,
    }, { dropDepth: 3, maximumStates: 10 }),
    (error) => error?.code === 'CHAOS_PROOF_GRAPH_LIMIT' && error.states === 10,
  );
});
