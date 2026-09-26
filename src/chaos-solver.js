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
  hasWinFrom,
  resolveActionOutcome,
} from './engine.js';

export const CHAOS_WIN = 1;
export const CHAOS_DRAW = 0;
export const CHAOS_LOSS = -1;

const UNKNOWN = 2;
const DEFAULT_MAX_STATES = 2_000_000;

function swapPlayers(board) {
  return board.map((row) => row.map((cell) => {
    if (cell === RED) return YELLOW;
    if (cell === YELLOW) return RED;
    return EMPTY;
  }));
}

function mirrorBoard(board) {
  return board.map((row) => [...row].reverse());
}

export function mirrorChaosAction(action, columns) {
  if (!action) return null;
  if (action.type === ACTION_DROP) {
    return { type: ACTION_DROP, column: columns - 1 - action.column };
  }
  if (action.type === ACTION_ROTATE_CW) return { type: ACTION_ROTATE_CCW };
  if (action.type === ACTION_ROTATE_CCW) return { type: ACTION_ROTATE_CW };
  return { type: action.type };
}

function normalizedForMover(board, currentPlayer) {
  return currentPlayer === RED ? board : swapPlayers(board);
}

export function canonicalChaosPosition(board, currentPlayer = RED) {
  const moverBoard = normalizedForMover(board, currentPlayer);
  const { rows, cols } = boardDimensions(moverBoard);
  const normal = boardToString(moverBoard);
  const mirroredBoard = mirrorBoard(moverBoard);
  const mirrored = boardToString(mirroredBoard);
  const useMirror = mirrored < normal;
  return {
    board: useMirror ? mirroredBoard : moverBoard.map((row) => [...row]),
    key: `${rows}x${cols}:${useMirror ? mirrored : normal}`,
    mirrored: useMirror,
    rows,
    cols,
  };
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
  if (outcome.status === 'won') {
    return outcome.winner === RED ? CHAOS_WIN : CHAOS_LOSS;
  }
  return null;
}

// Cell digits with the colours swapped: the child of an edge is seen from the
// next mover's side, and the next mover is Yellow in the relative colours.
const SWAPPED_CODE = [48, 50, 49];   // '0', '2', '1'
const SLASH = 47;
const SWAPPED_CELL = [EMPTY, YELLOW, RED];

/**
 * The canonical key of `board` seen by the next mover - colours swapped, and
 * mirrored when that sorts first - built in one pass without copying the
 * board. Equal to canonicalChaosPosition(swapPlayers(board), RED).key, which
 * cost four board copies and two strings for every edge, most of which lead
 * to a state already in the graph. Built from character codes, the key is
 * one flat string rather than a chain of a character per link.
 */
function nextCanonicalKey(board) {
  const rows = board.length;
  const cols = board[0].length;
  const length = rows * (cols + 1) - 1;
  const normal = new Array(length);
  const mirrored = new Array(length);
  let at = 0;
  let order = 0;          // < 0: the mirror image sorts first
  for (let row = 0; row < rows; row += 1) {
    if (row > 0) {
      normal[at] = SLASH;
      mirrored[at] = SLASH;
      at += 1;
    }
    const cells = board[row];
    for (let column = 0; column < cols; column += 1) {
      normal[at] = SWAPPED_CODE[cells[column]];
      mirrored[at] = SWAPPED_CODE[cells[cols - 1 - column]];
      if (order === 0) order = mirrored[at] - normal[at];
      at += 1;
    }
  }
  const useMirror = order < 0;
  return {
    key: `${rows}x${cols}:${String.fromCharCode.apply(null, useMirror ? mirrored : normal)}`,
    mirror: useMirror,
  };
}

/** The board a nextCanonicalKey() describes, built only for a new state. */
function nextCanonicalBoard(board, mirror) {
  const cols = board[0].length;
  return board.map((cells) => Array.from(cells, (_cell, column) => (
    SWAPPED_CELL[cells[mirror ? cols - 1 - column : column]])));
}

function createNode(board, key) {
  return {
    board,
    key,
    edges: [],
    predecessors: [],
  };
}

// One frozen action object per move, shared by every edge: a graph of 200,000
// states held two million of them.
const DROP_ACTIONS = Object.freeze(Array.from({ length: 10 },
  (_unused, column) => Object.freeze({ type: ACTION_DROP, column })));
const TRANSFORM_ACTIONS = Object.freeze([ACTION_FLIP, ACTION_ROTATE_CW, ACTION_ROTATE_CCW]
  .map((type) => Object.freeze({ type })));

function isFull(board) {
  return board[0].every((cell) => cell !== EMPTY);
}

const LINE_DIRECTIONS = [[0, 1], [1, 0], [1, 1], [1, -1]];

/** Which colours have a line of `connect`: bit 1 Red, bit 2 Yellow. One scan
 * from each run's first cell, stopping once both colours have one. */
function lineOwners(board, connect) {
  const rows = board.length;
  const cols = board[0].length;
  let owners = 0;
  for (let row = 0; row < rows; row += 1) {
    const cells = board[row];
    for (let column = 0; column < cols; column += 1) {
      const player = cells[column];
      if (player === EMPTY || (owners & player) !== 0) continue;
      for (const [rowStep, columnStep] of LINE_DIRECTIONS) {
        const lastRow = row + rowStep * (connect - 1);
        const lastColumn = column + columnStep * (connect - 1);
        if (lastRow >= rows || lastColumn < 0 || lastColumn >= cols) continue;
        let length = 1;
        while (length < connect && board[row + rowStep * length][column + columnStep * length] === player) {
          length += 1;
        }
        if (length === connect) {
          owners |= player;
          break;
        }
      }
      if (owners === (RED | YELLOW)) return owners;
    }
  }
  return owners;
}

/**
 * resolveActionOutcome() for a transformation by Red, answering only whether
 * each colour has a line: both is a loss for the mover, one is its owner's.
 * The engine's version gathers every winning cell of both colours, and
 * checking lines was most of the time spent building a graph.
 */
function transformOutcome(board, connect) {
  const owners = lineOwners(board, connect);
  if (owners !== 0) return owners === RED ? CHAOS_WIN : CHAOS_LOSS;
  return isFull(board) ? CHAOS_DRAW : null;
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
    throw new RangeError('The exact Chaos solver supports rectangular boards with at most 42 cells.');
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

export function buildChaosGraph(position, options = {}) {
  validatePosition(position);
  if (boardWinner(position.board, position.connect) !== EMPTY) {
    throw new RangeError('A searchable Chaos position cannot already be won.');
  }
  const maximumStates = options.maximumStates ?? DEFAULT_MAX_STATES;
  if (!Number.isInteger(maximumStates) || maximumStates < 1) {
    throw new RangeError('maximumStates must be a positive integer.');
  }

  const rootCanonical = canonicalChaosPosition(position.board, position.currentPlayer);
  const nodes = [createNode(rootCanonical.board, rootCanonical.key)];
  const indices = new Map([[rootCanonical.key, 0]]);

  for (let cursor = 0; cursor < nodes.length; cursor += 1) {
    const node = nodes[cursor];
    const { board } = node;
    // Each state is expanded once, in order; its board is not needed after.
    node.board = null;
    // Actions reaching the same outcome or state are one edge. Terminal
    // outcomes are -3..-1 here and states their index.
    const seenEdges = [];
    const addTerminal = (action, terminal) => {
      if (seenEdges.includes(terminal - 2)) return;
      seenEdges.push(terminal - 2);
      node.edges.push({ action, terminal, next: -1 });
    };
    const addChild = (action, childBoard) => {
      const child = nextCanonicalKey(childBoard);
      let childIndex = indices.get(child.key);
      if (childIndex !== undefined && seenEdges.includes(childIndex)) return;
      if (childIndex === undefined) {
        if (nodes.length >= maximumStates) {
          const error = new RangeError(
            `Exact Chaos graph exceeded the ${maximumStates.toLocaleString()}-state safety limit.`,
          );
          error.code = 'CHAOS_GRAPH_LIMIT';
          error.states = nodes.length;
          throw error;
        }
        childIndex = nodes.length;
        indices.set(child.key, childIndex);
        nodes.push(createNode(nextCanonicalBoard(childBoard, child.mirror), child.key));
      }
      seenEdges.push(childIndex);
      node.edges.push({ action, terminal: null, next: childIndex });
    };

    // A drop is tried in place and undone: only the mover's new piece can
    // complete a line, and a full board after it is a draw.
    for (let column = 0; column < board[0].length; column += 1) {
      let row = board.length - 1;
      while (row >= 0 && board[row][column] !== EMPTY) row -= 1;
      if (row < 0) continue;
      board[row][column] = RED;
      try {
        if (hasWinFrom(board, row, column, RED, position.connect)) addTerminal(DROP_ACTIONS[column], CHAOS_WIN);
        else if (isFull(board)) addTerminal(DROP_ACTIONS[column], CHAOS_DRAW);
        else addChild(DROP_ACTIONS[column], board);
      } finally {
        board[row][column] = EMPTY;
      }
    }
    if (isFull(board)) continue;
    for (const action of TRANSFORM_ACTIONS) {
      const { board: transformed } = applyAction(board, action, RED);
      const terminal = transformOutcome(transformed, position.connect);
      if (terminal !== null) addTerminal(action, terminal);
      else addChild(action, transformed);
    }
  }

  for (let parent = 0; parent < nodes.length; parent += 1) {
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
  };
}

export function solveChaosGraph(graph) {
  const { nodes } = graph;
  const values = new Int8Array(nodes.length);
  values.fill(UNKNOWN);
  const ranks = new Uint32Array(nodes.length);
  const bestEdges = new Int32Array(nodes.length);
  bestEdges.fill(-1);
  const losingActions = new Uint16Array(nodes.length);
  const maximumWinningChildRank = new Uint32Array(nodes.length);
  const heap = [];

  function push(index) {
    heap.push(index);
    let child = heap.length - 1;
    while (child > 0) {
      const parent = Math.floor((child - 1) / 2);
      if (ranks[heap[parent]] <= ranks[index]) break;
      heap[child] = heap[parent];
      child = parent;
    }
    heap[child] = index;
  }

  function pop() {
    const first = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      let parent = 0;
      while (true) {
        const left = parent * 2 + 1;
        if (left >= heap.length) break;
        const right = left + 1;
        const smaller = right < heap.length && ranks[heap[right]] < ranks[heap[left]]
          ? right
          : left;
        if (ranks[last] <= ranks[heap[smaller]]) break;
        heap[parent] = heap[smaller];
        parent = smaller;
      }
      heap[parent] = last;
    }
    return first;
  }

  for (let index = 0; index < nodes.length; index += 1) {
    const node = nodes[index];
    let winningEdge = -1;
    let losses = 0;
    let lossEdge = -1;
    for (let edgeIndex = 0; edgeIndex < node.edges.length; edgeIndex += 1) {
      const edge = node.edges[edgeIndex];
      if (edge.terminal === CHAOS_WIN && winningEdge < 0) winningEdge = edgeIndex;
      else if (edge.terminal === CHAOS_LOSS) {
        losses += 1;
        if (lossEdge < 0) lossEdge = edgeIndex;
      }
    }
    losingActions[index] = losses;

    if (winningEdge >= 0) {
      values[index] = CHAOS_WIN;
      ranks[index] = 1;
      bestEdges[index] = winningEdge;
      push(index);
    } else if (node.edges.length > 0 && node.edges.length === losses) {
      values[index] = CHAOS_LOSS;
      ranks[index] = 1;
      bestEdges[index] = lossEdge;
      push(index);
    }
  }

  while (heap.length > 0) {
    const child = pop();
    const childValue = values[child];

    for (const predecessor of nodes[child].predecessors) {
      const { parent, edge } = predecessor;
      if (values[parent] !== UNKNOWN) continue;
      if (childValue === CHAOS_LOSS) {
        values[parent] = CHAOS_WIN;
        ranks[parent] = ranks[child] + 1;
        bestEdges[parent] = edge;
        push(parent);
      } else if (childValue === CHAOS_WIN) {
        losingActions[parent] += 1;
        if (ranks[child] >= maximumWinningChildRank[parent]) {
          maximumWinningChildRank[parent] = ranks[child];
          bestEdges[parent] = edge;
        }
        if (losingActions[parent] === nodes[parent].edges.length) {
          values[parent] = CHAOS_LOSS;
          ranks[parent] = maximumWinningChildRank[parent] + 1;
          push(parent);
        }
      }
    }
  }

  let wins = 0;
  let draws = 0;
  let losses = 0;
  for (let index = 0; index < values.length; index += 1) {
    if (values[index] === UNKNOWN) {
      values[index] = CHAOS_DRAW;
      draws += 1;
    } else if (values[index] === CHAOS_WIN) wins += 1;
    else losses += 1;
  }
  for (let index = 0; index < values.length; index += 1) {
    if (values[index] !== CHAOS_DRAW) continue;
    const node = nodes[index];
    bestEdges[index] = node.edges.findIndex((edge) => (
      edge.terminal === CHAOS_DRAW
      || (edge.next >= 0 && values[edge.next] === CHAOS_DRAW)
    ));
  }

  return { values, ranks, bestEdges, wins, draws, losses };
}

function edgeValue(edge, values) {
  if (edge.terminal !== null) return edge.terminal;
  const child = values[edge.next];
  return child === CHAOS_DRAW ? CHAOS_DRAW : -child;
}

function actionPreference(action, columns) {
  if (action.type === ACTION_DROP) {
    return 100 - Math.abs(action.column - (columns - 1) / 2) * 10;
  }
  if (action.type === ACTION_FLIP) return 2;
  return 1;
}

function selectAction(node, targetValue, values, columns) {
  const candidates = node.edges.filter((edge) => edgeValue(edge, values) === targetValue);
  candidates.sort((first, second) => (
    actionPreference(second.action, columns) - actionPreference(first.action, columns)
  ));
  return candidates[0]?.action ?? null;
}

export function solveChaosPosition(position, options = {}) {
  const start = globalThis.performance?.now?.() ?? Date.now();
  const graph = buildChaosGraph(position, options);
  const solved = solveChaosGraph(graph);
  const value = solved.values[graph.root];
  const rootNode = graph.nodes[graph.root];
  // A won root plays its fastest win and a lost one its longest defence.
  // Every drawing action is as good as another, and the first in move order
  // was always the leftmost column: a drawn root prefers the centre.
  const preferredEdge = value === CHAOS_DRAW ? -1 : solved.bestEdges[graph.root];
  let action = preferredEdge >= 0
    ? rootNode.edges[preferredEdge].action
    : selectAction(rootNode, value, solved.values, graph.rootColumns);
  // Edges share frozen action objects; the caller gets its own.
  action = graph.rootMirrored ? mirrorChaosAction(action, graph.rootColumns) : action && { ...action };
  const elapsedMs = (globalThis.performance?.now?.() ?? Date.now()) - start;
  return {
    action,
    value,
    score: value,
    solved: true,
    solver: 'chaos-exact-graph',
    depth: solved.ranks[graph.root],
    nodes: graph.nodes.length,
    elapsedMs,
    principalVariation: action ? [{ ...action }] : [],
    graph: {
      states: graph.nodes.length,
      wins: solved.wins,
      draws: solved.draws,
      losses: solved.losses,
      rank: solved.ranks[graph.root],
    },
  };
}
