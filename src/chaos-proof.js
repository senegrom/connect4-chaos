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
  hasWinFrom,
  legalActions,
  resolveActionOutcome,
  sameAction,
} from './engine.js';
import {
  CHAOS_DRAW,
  CHAOS_LOSS,
  CHAOS_WIN,
  canonicalChaosPosition,
  mirrorChaosAction,
  solveChaosGraph,
} from './chaos-solver.js';

const DEFAULT_DROP_DEPTH = 2;
const DEFAULT_MAXIMUM_STATES = 150_000;

function now() {
  return globalThis.performance?.now?.() ?? Date.now();
}

function cloneAction(action) {
  return action ? { ...action } : null;
}

function actionKey(action) {
  return action?.type === ACTION_DROP ? `${action.type}:${action.column}` : action?.type ?? '';
}

function actionPreference(action, columns) {
  if (action?.type === ACTION_DROP) {
    return 1_000 - Math.abs(action.column - (columns - 1) / 2) * 20;
  }
  if (action?.type === ACTION_FLIP) return 3;
  if (action?.type === ACTION_ROTATE_CW) return 2;
  if (action?.type === ACTION_ROTATE_CCW) return 1;
  return 0;
}

function compareActions(first, second, columns) {
  return actionPreference(second, columns) - actionPreference(first, columns)
    || actionKey(first).localeCompare(actionKey(second));
}

function validatePosition(position) {
  if (!position || !Array.isArray(position.board) || position.board.length === 0) {
    throw new TypeError('A non-empty Chaos board is required.');
  }
  const { rows, cols } = boardDimensions(position.board);
  if (cols === 0 || position.board.some((row) => !Array.isArray(row) || row.length !== cols)) {
    throw new TypeError('The Chaos board must be rectangular.');
  }
  if (rows < 1 || cols < 1 || rows * cols > 42) {
    throw new RangeError('The bounded Chaos proof supports rectangular boards with at most 42 cells.');
  }
  if (position.currentPlayer !== RED && position.currentPlayer !== YELLOW) {
    throw new RangeError('Current player must be Red or Yellow.');
  }
  if (!Number.isInteger(position.connect)
      || position.connect < 1
      || position.connect > Math.max(rows, cols)) {
    throw new RangeError('Connect length must fit the Chaos board.');
  }

  for (let column = 0; column < cols; column += 1) {
    let foundPiece = false;
    for (let row = 0; row < rows; row += 1) {
      const cell = position.board[row][column];
      if (cell !== EMPTY && cell !== RED && cell !== YELLOW) {
        throw new RangeError('Board cells must be empty, Red, or Yellow.');
      }
      if (cell === EMPTY && foundPiece) {
        throw new RangeError('Chaos board pieces must obey gravity.');
      }
      if (cell !== EMPTY) foundPiece = true;
    }
  }
}

function boardWinner(board, connect) {
  let winner = EMPTY;
  for (const player of [RED, YELLOW]) {
    let won = false;
    for (let row = 0; row < board.length && !won; row += 1) {
      for (let column = 0; column < board[row].length; column += 1) {
        if (board[row][column] === player
            && hasWinFrom(board, row, column, player, connect)) {
          won = true;
          break;
        }
      }
    }
    if (!won) continue;
    if (winner !== EMPTY) {
      throw new RangeError('A searchable Chaos position cannot contain wins for both players.');
    }
    winner = player;
  }
  return winner;
}

function proofOptions(position, options) {
  const boardCells = position.board.length * position.board[0].length;
  const dropDepth = options.dropDepth ?? DEFAULT_DROP_DEPTH;
  if (!Number.isInteger(dropDepth) || dropDepth < 1 || dropDepth > boardCells) {
    throw new RangeError(`dropDepth must be an integer from 1 through ${boardCells}.`);
  }
  const maximumStates = options.maximumStates ?? DEFAULT_MAXIMUM_STATES;
  if (!Number.isInteger(maximumStates) || maximumStates < 1 || maximumStates > 2_000_000) {
    throw new RangeError('maximumStates must be an integer from 1 through 2,000,000.');
  }
  return { dropDepth, maximumStates };
}

function edgeOutcome(result, action, connect) {
  const outcome = resolveActionOutcome(
    result.board,
    connect,
    RED,
    action.type,
    action.type === ACTION_DROP ? { row: result.row, column: result.column } : null,
  );
  if (outcome.status === 'draw') return CHAOS_DRAW;
  if (outcome.status === 'won') return outcome.winner === RED ? CHAOS_WIN : CHAOS_LOSS;
  return null;
}

function createNode(board, key, dropsLeft, aiTurn) {
  return {
    board,
    key,
    dropsLeft,
    aiTurn,
    edges: [],
    predecessors: [],
    edgeIndices: new Map(),
  };
}

function addEdge(node, identity, edge, action) {
  const existingIndex = node.edgeIndices.get(identity);
  if (existingIndex !== undefined) {
    const existing = node.edges[existingIndex];
    if (!existing.actions.some((candidate) => sameAction(candidate, action))) {
      existing.actions.push(cloneAction(action));
    }
    return;
  }
  node.edgeIndices.set(identity, node.edges.length);
  node.edges.push({ ...edge, action: cloneAction(action), actions: [cloneAction(action)] });
}

/**
 * Builds a finite loopy graph through a fixed number of future drops.
 * Transformations do not consume the horizon, so transform-only cycles are
 * represented exactly. A non-terminal drop at the horizon becomes an unknown
 * frontier edge whose value is bounded separately by the solver.
 */
export function buildChaosProofGraph(position, options = {}) {
  validatePosition(position);
  if (boardWinner(position.board, position.connect) !== EMPTY) {
    throw new RangeError('A searchable Chaos position cannot already be won.');
  }
  const { dropDepth, maximumStates } = proofOptions(position, options);
  const rootCanonical = canonicalChaosPosition(position.board, position.currentPlayer);
  const rootKey = `${dropDepth}:1:${rootCanonical.key}`;
  const nodes = [createNode(rootCanonical.board, rootKey, dropDepth, true)];
  const indices = new Map([[rootKey, 0]]);
  const frontierKeys = new Set();
  let frontierEdges = 0;
  let edgeCount = 0;

  for (let cursor = 0; cursor < nodes.length; cursor += 1) {
    const node = nodes[cursor];
    for (const action of legalActions(node.board, true)) {
      const result = applyAction(node.board, action, RED);
      if (!result) continue;
      const terminal = edgeOutcome(result, action, position.connect);
      if (terminal !== null) {
        addEdge(node, `terminal:${terminal}`, {
          terminal,
          next: -1,
          frontier: false,
        }, action);
        continue;
      }

      const child = canonicalChaosPosition(result.board, YELLOW);
      const nextDropsLeft = node.dropsLeft - (action.type === ACTION_DROP ? 1 : 0);
      if (nextDropsLeft === 0) {
        const identity = `frontier:${child.key}`;
        const before = node.edges.length;
        addEdge(node, identity, {
          terminal: CHAOS_DRAW,
          next: -1,
          frontier: true,
          frontierKey: `${!node.aiTurn ? 1 : 0}:${child.key}`,
          frontierLower: node.aiTurn ? CHAOS_LOSS : CHAOS_WIN,
          frontierUpper: node.aiTurn ? CHAOS_WIN : CHAOS_LOSS,
        }, action);
        if (node.edges.length > before) {
          frontierEdges += 1;
          frontierKeys.add(`${!node.aiTurn ? 1 : 0}:${child.key}`);
        }
        continue;
      }

      const childAiTurn = !node.aiTurn;
      const childKey = `${nextDropsLeft}:${childAiTurn ? 1 : 0}:${child.key}`;
      let childIndex = indices.get(childKey);
      if (childIndex === undefined) {
        if (nodes.length >= maximumStates) {
          const error = new RangeError(
            `Bounded Chaos proof exceeded the ${maximumStates.toLocaleString()}-state safety limit.`,
          );
          error.code = 'CHAOS_PROOF_GRAPH_LIMIT';
          error.states = nodes.length;
          error.dropDepth = dropDepth;
          throw error;
        }
        childIndex = nodes.length;
        indices.set(childKey, childIndex);
        nodes.push(createNode(child.board, childKey, nextDropsLeft, childAiTurn));
      }
      addEdge(node, `next:${childKey}`, {
        terminal: null,
        next: childIndex,
        frontier: false,
      }, action);
    }
  }

  for (let parent = 0; parent < nodes.length; parent += 1) {
    delete nodes[parent].edgeIndices;
    edgeCount += nodes[parent].edges.length;
    for (let edge = 0; edge < nodes[parent].edges.length; edge += 1) {
      const child = nodes[parent].edges[edge].next;
      if (child >= 0) nodes[child].predecessors.push({ parent, edge });
    }
  }

  return {
    nodes,
    root: 0,
    rootMirrored: rootCanonical.mirrored,
    rootColumns: rootCanonical.cols,
    dropDepth,
    edges: edgeCount,
    frontierEdges,
    frontierStates: frontierKeys.size,
  };
}

function solveWithFrontierBound(graph, bound) {
  for (const node of graph.nodes) {
    for (const edge of node.edges) {
      if (edge.frontier) {
        edge.terminal = bound === 'lower' ? edge.frontierLower : edge.frontierUpper;
      }
    }
  }
  try {
    return solveChaosGraph(graph);
  } finally {
    for (const node of graph.nodes) {
      for (const edge of node.edges) {
        if (edge.frontier) edge.terminal = CHAOS_DRAW;
      }
    }
  }
}

function edgeValue(edge, solved, bound) {
  if (edge.frontier) return bound === 'lower' ? edge.frontierLower : edge.frontierUpper;
  if (edge.terminal !== null) return edge.terminal;
  const childValue = solved.values[edge.next];
  return childValue === CHAOS_DRAW ? CHAOS_DRAW : -childValue;
}

function preferredAlias(edge, columns) {
  return [...edge.actions].sort((first, second) => compareActions(first, second, columns))[0] ?? null;
}

function mappedAction(action, graph) {
  const selected = graph.rootMirrored
    ? mirrorChaosAction(action, graph.rootColumns)
    : cloneAction(action);
  return cloneAction(selected);
}

function chooseExactAction(graph, lower, upper, exactValue, bounds) {
  const root = graph.nodes[graph.root];
  if (exactValue === CHAOS_WIN) {
    const edgeIndex = lower.bestEdges[graph.root];
    if (edgeIndex >= 0) return preferredAlias(root.edges[edgeIndex], graph.rootColumns);
  } else if (exactValue === CHAOS_LOSS) {
    const edgeIndex = upper.bestEdges[graph.root];
    if (edgeIndex >= 0) return preferredAlias(root.edges[edgeIndex], graph.rootColumns);
  }

  const exactCandidates = bounds
    .filter((entry) => entry.lower === exactValue && entry.upper === exactValue)
    .map((entry) => entry.canonicalAction)
    .sort((first, second) => compareActions(first, second, graph.rootColumns));
  return exactCandidates[0] ?? null;
}

function compareBounds(first, second, columns) {
  return second.lower - first.lower
    || second.upper - first.upper
    || compareActions(first.canonicalAction, second.canonicalAction, columns);
}

/**
 * Solves the bounded graph twice: once assuming every horizon state is a loss
 * for the root AI, and once assuming every horizon state is an AI win. The
 * resulting root-relative values are rigorous lower and upper W/D/L bounds.
 */
export function solveChaosProofPosition(position, options = {}) {
  const start = now();
  const graph = buildChaosProofGraph(position, options);
  const lower = solveWithFrontierBound(graph, 'lower');
  const upper = solveWithFrontierBound(graph, 'upper');
  const lowerValue = lower.values[graph.root];
  const upperValue = upper.values[graph.root];
  if (lowerValue > upperValue) {
    throw new Error('Bounded Chaos proof produced inconsistent value bounds.');
  }

  const root = graph.nodes[graph.root];
  const canonicalBounds = [];
  for (const edge of root.edges) {
    const lowerEdgeValue = edgeValue(edge, lower, 'lower');
    const upperEdgeValue = edgeValue(edge, upper, 'upper');
    if (lowerEdgeValue > upperEdgeValue) {
      throw new Error('Bounded Chaos proof produced inconsistent action bounds.');
    }
    for (const action of edge.actions) {
      canonicalBounds.push({
        canonicalAction: cloneAction(action),
        lower: lowerEdgeValue,
        upper: upperEdgeValue,
        frontier: edge.frontier,
      });
    }
  }

  const solved = lowerValue === upperValue;
  const exactValue = solved ? lowerValue : null;
  let canonicalAction = solved
    ? chooseExactAction(graph, lower, upper, exactValue, canonicalBounds)
    : [...canonicalBounds].sort((first, second) => (
      compareBounds(first, second, graph.rootColumns)
    ))[0]?.canonicalAction ?? null;
  if (!canonicalAction && root.edges.length > 0) {
    canonicalAction = preferredAlias(root.edges[0], graph.rootColumns);
  }

  const actionBounds = canonicalBounds.map((entry) => ({
    action: mappedAction(entry.canonicalAction, graph),
    lower: entry.lower,
    upper: entry.upper,
    frontier: entry.frontier,
  })).sort((first, second) => (
    second.lower - first.lower
    || second.upper - first.upper
    || compareActions(first.action, second.action, graph.rootColumns)
  ));
  const action = mappedAction(canonicalAction, graph);
  const selectedBound = actionBounds.find((entry) => sameAction(entry.action, action)) ?? null;
  const rankSource = exactValue === CHAOS_LOSS ? upper : lower;
  const elapsedMs = now() - start;

  return {
    action,
    value: solved ? exactValue : CHAOS_DRAW,
    score: solved ? exactValue : CHAOS_DRAW,
    lowerValue,
    upperValue,
    solved,
    solver: 'chaos-bounded-proof',
    depth: graph.dropDepth,
    nodes: graph.nodes.length,
    elapsedMs,
    principalVariation: action ? [{ ...action }] : [],
    actionBounds,
    selectedBound,
    provenWinningActions: actionBounds
      .filter((entry) => entry.lower === CHAOS_WIN)
      .map((entry) => cloneAction(entry.action)),
    certifiedNonLosingActions: actionBounds
      .filter((entry) => entry.lower >= CHAOS_DRAW)
      .map((entry) => cloneAction(entry.action)),
    provenLosingActions: actionBounds
      .filter((entry) => entry.upper === CHAOS_LOSS)
      .map((entry) => cloneAction(entry.action)),
    graph: {
      states: graph.nodes.length,
      edges: graph.edges,
      frontierEdges: graph.frontierEdges,
      frontierStates: graph.frontierStates,
      rank: solved ? rankSource.ranks[graph.root] : 0,
      lower: {
        value: lowerValue,
        wins: lower.wins,
        draws: lower.draws,
        losses: lower.losses,
      },
      upper: {
        value: upperValue,
        wins: upper.wins,
        draws: upper.draws,
        losses: upper.losses,
      },
    },
  };
}
