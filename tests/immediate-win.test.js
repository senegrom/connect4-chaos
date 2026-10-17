import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

import {
  ACTION_DROP, ACTION_FLIP, ACTION_ROTATE_CCW, ACTION_ROTATE_CW, RED, YELLOW,
  applyAction, boardDimensions, createBoard, immediateWinningActions, resolveActionOutcome, sameAction,
} from '../src/engine.js';
import { chooseMove, preferImmediateWin } from '../src/ai.js';
import { choosePreparedAction, choosePreparedMove } from '../src/ai-worker.js';
import { searchSummary } from '../src/analysis-state.js';
import { neuralSearchInfo } from '../src/search-info.js';

// A player that can end the game now should end it. Winning in five instead
// is worth the same game-theoretically, but it leaves the person opposite
// believing they still have a game.

/** Red to move on 6x7 with three in a row on the bottom, at columns 0-2. */
function redWinsAtThree() {
  const board = createBoard(6, 7);
  for (const column of [0, 1, 2]) {
    board[5][column] = RED;
    board[5][column + 4] = YELLOW;      // keep the piece counts equal
  }
  board[4][0] = YELLOW;
  board[4][1] = RED;
  return { board, currentPlayer: RED, connect: 4, chaosMode: false };
}

function wins(position, action) {
  const applied = applyAction(position.board, action, position.currentPlayer);
  const outcome = resolveActionOutcome(applied.board, position.connect, position.currentPlayer,
    action.type, action.type === ACTION_DROP ? { row: applied.row, column: applied.column } : null);
  return outcome.status === 'won' && outcome.winner === position.currentPlayer;
}

test('the winning drop is available and really wins', () => {
  const position = redWinsAtThree();
  const winning = immediateWinningActions(position.board, RED, 4, false);
  assert.deepEqual(winning, [{ type: ACTION_DROP, column: 3 }]);
  assert.ok(wins(position, winning[0]));
});

for (const difficulty of ['easy', 'medium', 'hard', 'brutal']) {
  test(`${difficulty} plays the win in one rather than any other move`, () => {
    const position = redWinsAtThree();
    const result = chooseMove(position, { difficulty, random: () => 0 });
    assert.ok(wins(position, result.action),
      `${difficulty} played ${JSON.stringify(result.action)} with a mate in one available`);
  });
}

test('the prepared route (Perfect and the certified policies) plays the win in one', () => {
  const position = redWinsAtThree();
  const result = choosePreparedAction(position, { difficulty: 'brutal' });
  assert.ok(wins(position, result.action));
});

// The override must leave a route's own reporting alone: only the move
// changes, never the solver, the proof or the statistics.
test('preferring the win keeps every other field the route reported', () => {
  const position = redWinsAtThree();
  const reported = {
    action: { type: ACTION_DROP, column: 6 }, score: 1234, depth: 9, nodes: 42,
    solver: 'chaos-exact-graph', solved: true,
  };
  const result = preferImmediateWin(position, reported);
  assert.deepEqual(result.action, { type: ACTION_DROP, column: 3 });
  for (const key of ['score', 'depth', 'nodes', 'solver', 'solved']) {
    assert.deepEqual(result[key], reported[key], key);
  }
});

test('a winning move already chosen is left exactly as it was', () => {
  const position = redWinsAtThree();
  const chosen = { action: { type: ACTION_DROP, column: 3 }, score: 7 };
  assert.equal(preferImmediateWin(position, chosen), chosen);
});

test('a position with no win in one is not touched', () => {
  const position = { board: createBoard(6, 7), currentPlayer: RED, connect: 4, chaosMode: false };
  const chosen = { action: { type: ACTION_DROP, column: 0 }, score: 0 };
  assert.equal(preferImmediateWin(position, chosen), chosen);
});

// In Chaos a transform can complete a line, so it counts as a winning move -
// but one that completes the opponent's line instead must never be offered.
test('a transform that wins counts, and one that loses does not', () => {
  const board = createBoard(4, 4);
  board[3][0] = RED; board[3][1] = RED; board[3][2] = RED;
  board[3][3] = YELLOW; board[2][0] = YELLOW; board[2][1] = YELLOW;
  const position = { board, currentPlayer: RED, connect: 4, chaosMode: true };
  const winning = immediateWinningActions(board, RED, 4, true);
  for (const action of winning) {
    assert.ok(wins(position, action), `${JSON.stringify(action)} does not actually win`);
  }
  for (const action of [{ type: ACTION_FLIP }, { type: ACTION_ROTATE_CW },
    { type: ACTION_ROTATE_CCW }]) {
    if (wins(position, action)) {
      assert.ok(winning.some((win) => sameAction(win, action)),
        `${JSON.stringify(action)} wins but was not offered`);
    }
  }
});

// The neural opponent's network is a 133 MB download on first use. A win in
// one needs none of it, so a first-time player is not asked to fetch it.
test('the neural opponent plays a win in one without downloading or loading its network', async () => {
  const source = readFileSync(new URL('../src/neural-app.js', import.meta.url), 'utf8');
  const asked = [];
  const context = { immediateWinningActions, neuralLoadState: () => 'idle',
    requestDownload: async () => { asked.push('consent'); return true; },
    showDownloadProgress: () => { asked.push('progress'); return { update() {}, note() {}, close() {} }; },
    loadNeuralNetwork: async () => { asked.push('network'); throw new Error('no network in this test'); } };
  vm.runInNewContext(source.slice(source.indexOf('export async function')).replace('export ', ''), context);
  let played = null;
  await context.runNeuralRequest({ controller: new AbortController(), position: redWinsAtThree() }, {
    isCurrent: () => true, shouldStop: () => false, onSearch() {}, onFraction() {},
    finish: (result) => { played = result; }, fail: (message) => assert.fail(message),
  });
  assert.deepEqual(asked, []);
  assert.deepEqual(played.action, { type: ACTION_DROP, column: 3 });
  assert.equal(played.evaluations, 0);
  assert.equal(played.backend, null, 'no backend ran');
  assert.equal(played.outcome, undefined, 'the network was not asked, so it has no estimate to show');
  assert.equal(played.networkOutcome, undefined, 'nor an output of its own');
});

/** The AI details line app.js's renderSearchInfo writes for a finished move,
 * with that function run alone. */
function analysisLine(opponent) {
  const source = readFileSync(new URL('../src/app.js', import.meta.url), 'utf8');
  const start = source.indexOf('function renderSearchInfo(');
  const elements = { searchInfo: { textContent: '' } };
  const state = { config: { opponent }, aiThinking: false, aiError: null, liveSearch: null, lastSearch: null };
  const context = { elements, state, neuralSearchInfo,
    numberFormatter: new Intl.NumberFormat('en', { maximumFractionDigits: 0 }) };
  vm.runInNewContext(source.slice(start, source.indexOf('\n}\n', start) + 2), context);
  return (result) => {
    state.lastSearch = searchSummary(result);
    context.renderSearchInfo();
    return elements.searchInfo.textContent;
  };
}

test('the analysis line reports that win without claiming a device ran the network', () => {
  const line = analysisLine('neural');
  assert.equal(line({ action: { type: ACTION_DROP, column: 3 }, score: 1, depth: 1, nodes: 0,
    evaluations: 0, elapsedMs: 0, solver: 'neural', solved: false, backend: null }), 'Neural opponent · Immediate win');
  assert.equal(line({ score: 0.2, depth: 0, nodes: 64, evaluations: 65, elapsedMs: 4_500,
    solver: 'neural', solved: false, backend: 'webgpu' }), 'Neural network · 64 simulations · on the GPU · 4.5s');
});

/** Red to move on 6x7 Chaos wins at once with drop 6. Six cells are empty,
 * so the exact endgame takes the position at every searched level. */
function chaosEndgameWin() {
  const board = [[2, 2, 0, 0, 0, 0, 0], [2, 1, 1, 2, 1, 2, 0], [2, 2, 1, 1, 2, 1, 1],
    [1, 1, 2, 2, 1, 2, 2], [1, 2, 1, 1, 2, 2, 1], [2, 1, 2, 1, 1, 2, 1]];
  return { board, currentPlayer: RED, connect: 4, chaosMode: true };
}

// The exact Chaos endgame plays a win on the spot before it builds a graph.
// Its result still names the graph solver, and the line read "Exact Chaos
// retrograde · 0 states · forced result within 1 plies".
test('the analysis line reports a Chaos endgame win on the spot without a retrograde analysis', () => {
  const line = analysisLine('brutal');
  for (const difficulty of ['medium', 'hard', 'brutal']) {
    const result = choosePreparedAction(chaosEndgameWin(), { difficulty, aiPlayer: RED });
    assert.equal(result.solver, 'chaos-exact-graph', difficulty);
    assert.equal(line(result), 'Exact Chaos endgame · Immediate win', difficulty);
  }
  // A graph that was built and solved keeps its own line.
  assert.equal(line({ solver: 'chaos-exact-graph', solved: true, score: 1, depth: 3, nodes: 1_198 }),
    'Exact Chaos retrograde · 1,198 states · forced result within 3 plies');
  assert.equal(line({ solver: 'chaos-exact-graph', solved: true, score: 0, depth: 0, nodes: 628 }),
    'Exact Chaos retrograde · 628 states · cycle-safe draw analysis');
});

/** The progress app.js's renderStatus writes beside "AI is thinking…" for
 * the search's latest report, with that function run alone. */
function thinkingLine() {
  const source = readFileSync(new URL('../src/app.js', import.meta.url), 'utf8');
  const start = source.indexOf('function renderStatus(');
  const elements = { statusDisc: {}, statusText: {}, aiErrorText: {}, moveInfo: {},
    thinkingIndicator: { classList: { toggle() {} } }, thinkingProgress: { textContent: '' } };
  const state = { status: 'playing', currentPlayer: RED, aiThinking: true, aiError: null, liveSearch: null,
    board: createBoard(6, 7), moveCount: 0, config: { connect: 4 } };
  const context = { elements, state, RED, boardDimensions, playerClass: () => 'red',
    statusMessage: () => 'AI is thinking…', renderThinkingBar() {}, currentRepetitionCount: () => 1,
    numberFormatter: new Intl.NumberFormat('en', { maximumFractionDigits: 0 }) };
  vm.runInNewContext(source.slice(start, source.indexOf('\n}\n', start) + 2), context);
  return (progress) => {
    state.liveSearch = progress;
    context.renderStatus();
    return elements.thinkingProgress.textContent;
  };
}

// The worker posts each report the search makes, and the page shows the
// latest as its live search until the move arrives (ai-worker.js, app.js's
// handleAiWorkerMessage). The win on the spot reports its result once, just
// before it is posted, and the status line read "Depth 1 · 0 positions".
test('the status line reports a Chaos endgame win on the spot while the AI thinks', async () => {
  const line = thinkingLine();
  for (const difficulty of ['medium', 'hard', 'brutal']) {
    const reports = [];
    const result = await choosePreparedMove(chaosEndgameWin(), { difficulty, aiPlayer: RED,
      // A copy, as postMessage hands it to the page.
      onIteration: (progress) => reports.push(structuredClone(progress)) });
    assert.deepEqual(reports, [result], difficulty);
    assert.equal(line(reports[0]), 'Immediate win', difficulty);
  }
  // A graph that was built keeps the search's line.
  assert.equal(line({ solver: 'chaos-exact-graph', solved: true, score: 1, depth: 3, nodes: 1_198 }),
    'Depth 3 · 1,198 positions');
});
