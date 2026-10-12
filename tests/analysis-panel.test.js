import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

import * as analysis from '../src/analysis-state.js';
import { evaluateBoard } from '../src/board-evaluation.js';
import * as engine from '../src/engine.js';
import { bestAction } from '../src/neural-search.js';
import * as storage from '../src/round-storage.js';
import { createScoreStore } from '../src/score-store.js';
import { RELOAD_MESSAGE } from '../src/site-build.js';

// The analysis panel as the page runs it: the real round controller, Undo,
// moves, the AI's hand-off and the neural request's callbacks, with the
// neural request itself scripted by the test. Storage is in memory.
const source = readFileSync(new URL('../src/app.js', import.meta.url), 'utf8');
const between = (from, to) => source.slice(source.indexOf(from), source.indexOf(to));
const code = [
  between('function saveRound()', 'function applySettingsAndStartRound()'),
  between('function canHumanAct()', 'let resetScoreConfirmTimer'),
  between('function setAnalysisMode(', 'function renderAiRecovery('),
  between('function renderEvaluation(', 'function renderSearchInfo('),
  between('async function performAction(', '\nfunction isLegalAiAction('),
  between('function isLegalAiAction(', 'function disposeAiWorker('),
  between('let resumeNeuralOnVisible', 'function renderAiState('),
  between('function finishAiRequest(', "/** Easy's own search"),
  between('async function runNeuralMove(', 'function postToWorker(')
    .replace("import('./neural-app.js')", 'importNeuralApp()'),
  between('const RELOAD_REQUEST_KEY', 'function switchToBrutal('),
  source.slice(source.lastIndexOf('const savedRound =')),
].join('\n');

const mapStorage = (data) => ({
  getItem: (key) => data.get(key) ?? null,
  setItem: (key, value) => data.set(key, value),
  removeItem: (key) => data.delete(key),
});
let nextId = 0;

/** One load of the page. Each neural turn waits in `turns` for the test to
 * report progress through its callbacks and to finish it. */
function page(config, data = new Map(), tabData = new Map()) {
  const state = { config: engine.normalizeConfig(config), scores: { 1: 0, 2: 0, draw: 0 }, history: [], version: 0,
    aiRequestId: 0, estimateFor: null, gameFirstLayout: false, touchHintDismissed: false };
  const turns = [];
  const elements = {
    evaluationPanel: { hidden: true, dataset: {} },
    evaluationLabel: { textContent: '' },
    evaluationDescription: { textContent: '' },
    evaluationBalance: { hidden: false, style: { setProperty(name, value) { this[name] = value; } },
      setAttribute(name, value) { this[name] = value; } },
    exactResult: { hidden: true },
    exactBadge: { textContent: '' },
    exactResultText: { textContent: '' },
    selectedColumnStatus: { textContent: '' },
    reloadPageButton: {},
    boardFrame: { classList: { add() {}, remove() {} } },
  };
  const context = { ...engine, ...storage, ...analysis, evaluateBoard, RELOAD_MESSAGE, state, elements,
    AbortController, document: { hidden: false },
    resultId: () => `round-${++nextId}`, populateSettingsForm() {}, setSettingsExpanded() {}, renderGuidance() {},
    renderStatus() {}, renderActions() {}, renderAiRecovery() {}, renderSearchInfo() {}, showResultDialog() {},
    closeResultDialog() {}, clearBoardAnimations() {}, refreshScores() {}, scoreWarning() {},
    disposeAiWorker() {}, invalidateNeuralNetwork() {}, retainIdleAiWorker() {}, postToWorker() {},
    fallbackOrStop: (_request, message) => context.stopAiWithError(message),
    animationPlan: () => null, pause: async () => {},
    scoreStore: createScoreStore({ indexedDB: null }),
    acceptScore(result) { state.scores = result.scores; return result.receipt; },
    isAiGame: () => state.config.opponent !== 'human',
    pushSnapshot(receipt = null) { state.history.push({ ...storage.makeSnapshot(state), scoreReceipt: receipt }); },
    restoreSnapshot(snapshot) { storage.restoreSnapshot(state, snapshot); },
    saveJson: (key, value) => data.set(key, JSON.stringify(value)),
    sessionStorage: mapStorage(tabData),
    roundStore: storage.createRoundStore({ sharedStorage: () => mapStorage(data), tabStorage: () => mapStorage(tabData) }),
    siteBuild: { ensureCurrent: async () => {}, reload: async () => {} },
    importNeuralApp: async () => ({
      runNeuralRequest: (request, callbacks) => new Promise((resolve) => turns.push({ request, ...callbacks, resolve })),
    }),
    enableCrossOriginIsolation: async () => false,
    // Every render the page makes reaches the panel.
    renderAll: () => context.renderEvaluation(),
    renderAiState: () => context.renderEvaluation(),
  };
  vm.createContext(context);
  vm.runInContext(code, context);
  const panel = () => ({
    mode: elements.evaluationPanel.dataset.analysisMode,
    label: elements.evaluationLabel.textContent,
    description: elements.evaluationDescription.textContent,
    share: elements.evaluationBalance.style['--you-share'],
    shown: !elements.evaluationBalance.hidden,
    text: elements.evaluationBalance['aria-label'],
  });
  return { state, data, tabData, turns, context, panel };
}

// Lets a move's awaits, and a request's start, run to completion.
const settle = async () => {
  for (let round = 0; round < 5; round += 1) await new Promise((resolve) => setImmediate(resolve));
};

/** The panel as it reads for the heuristic: the board alone decides it. */
function heuristicFor(board) {
  const score = evaluateBoard(board, 4, engine.YELLOW);
  const share = 1 / (1 + Math.exp(-Math.max(-12_000, Math.min(12_000, score)) / 1_800));
  const label = score > 2_500 ? 'AI ahead' : score > 500 ? 'AI slight edge' : score < -2_500 ? 'You are ahead'
    : score < -500 ? 'You have a slight edge' : 'Even';
  return { mode: 'heuristic', label, description: 'Heuristic position estimate',
    share: `${Math.round((1 - share) * 100)}%`, shown: true, text: `Heuristic position estimate: ${label.toLowerCase()}` };
}

/** One neural turn, which the page has just asked for: progress, then the move. */
async function aiTurn(game, { live, column, outcome, nodes = 16 }) {
  await settle();
  const turn = game.turns.at(-1);
  assert.deepEqual(game.panel(), heuristicFor(game.state.board), 'nothing from the network yet');
  turn.onSearch({ solver: 'neural-searching', note: 'Neural search · up to 16 simulations on wasm' });
  assert.equal(game.panel().mode, 'heuristic', 'starting is not an estimate');
  turn.onFraction(0.5, { outcome: live, nodes: 8 });
  const thinking = game.panel();
  turn.finish({ action: { type: 'drop', column }, score: 0, depth: 0, nodes, evaluations: nodes + 1, elapsedMs: 120,
    solver: 'neural', solved: false, backend: 'wasm', outcome });
  turn.resolve();
  await settle();
  return thinking;
}

test('a neural game shows the network\'s estimate while it thinks and for the position its move made', async () => {
  const game = page({ opponent: 'neural', startingPlayer: 1 });
  assert.deepEqual(game.panel(), heuristicFor(game.state.board));
  await game.context.performAction({ type: 'drop', column: 3 });
  // The live estimate is the AI's win, draw and loss in the position it is to move in.
  const thinking = await aiTurn(game, { live: [0.5, 0.2, 0.3], column: 3, outcome: [0.6, 0.25, 0.15] });
  assert.deepEqual(thinking, { mode: 'neural', label: 'AI 50% · Draw 20% · You 30%',
    description: 'Neural network estimate after 8 simulations', share: '40%', shown: true,
    text: 'Neural network estimate after 8 simulations: an AI win 50 percent, a draw 20 percent, your win 30 percent' });
  assert.equal(game.state.moveCount, 2);
  assert.deepEqual(game.panel(), { mode: 'neural', label: 'AI 60% · Draw 25% · You 15%',
    description: 'Neural network estimate after 16 simulations', share: '27.5%', shown: true,
    text: 'Neural network estimate after 16 simulations: an AI win 60 percent, a draw 25 percent, your win 15 percent' });

  // The human's move leaves that estimate behind: the heuristic shows until
  // the network speaks about the new position (aiTurn checks it).
  await game.context.performAction({ type: 'drop', column: 2 });
  await aiTurn(game, { live: [0.4, 0.3, 0.3], column: 2, outcome: [0.45, 0.3, 0.25] });
  assert.equal(game.state.moveCount, 4);
  assert.equal(game.panel().label, 'AI 45% · Draw 30% · You 25%');

  // Undo returns to the position after the first AI move. Its snapshot still
  // holds the estimate made for it, but Undo begins a new version.
  await game.context.undoTurn();
  assert.equal(game.state.moveCount, 2);
  assert.deepEqual(game.state.lastSearch.outcome, [0.6, 0.25, 0.15]);
  assert.deepEqual(game.panel(), heuristicFor(game.state.board));

  // So does a reload that restores the round, whose save holds it too.
  const reloaded = page({ opponent: 'neural', startingPlayer: 1 }, game.data, game.tabData);
  assert.equal(reloaded.state.moveCount, 2);
  assert.deepEqual(reloaded.state.lastSearch.outcome, [0.6, 0.25, 0.15]);
  assert.deepEqual(reloaded.panel(), heuristicFor(reloaded.state.board));
});

test('a new round or new rules leave no estimate, nor a request they cancelled', async () => {
  const game = page({ opponent: 'neural', startingPlayer: 1 });
  await game.context.performAction({ type: 'drop', column: 3 });
  await aiTurn(game, { live: [0.5, 0.2, 0.3], column: 3, outcome: [0.6, 0.25, 0.15] });
  assert.equal(game.panel().mode, 'neural');
  game.context.startRound();
  assert.deepEqual(game.panel(), heuristicFor(game.state.board));
  await game.context.performAction({ type: 'drop', column: 0 });
  await settle();
  const cancelled = game.turns.at(-1);
  cancelled.onFraction(0.5, { outcome: [0.9, 0.05, 0.05], nodes: 8 });
  assert.equal(game.panel().mode, 'neural');
  game.context.startRound({ ...game.state.config, rows: 7 });
  assert.deepEqual(game.panel(), heuristicFor(game.state.board));
  // A late report from the request the new rules cancelled must not land on
  // the next request's position.
  await game.context.performAction({ type: 'drop', column: 0 });
  await settle();
  cancelled.onFraction(0.75, { outcome: [0.9, 0.05, 0.05], nodes: 12 });
  assert.deepEqual(game.panel(), heuristicFor(game.state.board));
});

test('an error, a finished game and other opponents keep their own panels', async () => {
  const game = page({ opponent: 'neural', startingPlayer: 1 });
  await game.context.performAction({ type: 'drop', column: 3 });
  await settle();
  game.turns.at(-1).onFraction(0.5, { outcome: [0.5, 0.2, 0.3], nodes: 8 });
  game.context.stopAiWithError('The neural opponent failed: lost. Retry to restart it.');
  assert.deepEqual([game.panel().mode, game.panel().label], ['error', 'Analysis unavailable']);
  // Retry asks again; the estimate of the failed request is not shown.
  game.context.requestAiMove();
  assert.deepEqual(game.panel(), heuristicFor(game.state.board));

  // The network's estimate of a finished game gives way to its result.
  const finished = page({ opponent: 'neural', startingPlayer: 1 });
  for (const column of [0, 1, 0, 1, 0, 1]) {
    if (finished.state.currentPlayer === engine.RED) await finished.context.performAction({ type: 'drop', column });
    else await aiTurn(finished, { live: [0.1, 0.1, 0.8], column, outcome: [0.05, 0.05, 0.9] });
  }
  await finished.context.performAction({ type: 'drop', column: 0 });
  assert.equal(finished.state.status, 'won');
  assert.deepEqual([finished.panel().mode, finished.panel().label], ['exact', 'Round result']);

  // Every other opponent shows the heuristic, whatever a result carries.
  const brutal = page({ opponent: 'brutal', startingPlayer: 1 });
  await brutal.context.performAction({ type: 'drop', column: 3 });
  brutal.context.finishAiRequest(brutal.state.aiRequest, { result: { action: { type: 'drop', column: 3 },
    solver: 'bitboard', solved: false, score: 0, depth: 9, nodes: 1_000, outcome: [0.6, 0.25, 0.15] } });
  await settle();
  assert.equal(brutal.state.moveCount, 2);
  assert.deepEqual(brutal.panel(), heuristicFor(brutal.state.board));
});

test('the estimate for the position a transform makes waits until the board has turned', async () => {
  const game = page({ opponent: 'neural', startingPlayer: 1, chaosMode: true });
  await game.context.performAction({ type: 'drop', column: 3 });
  await settle();
  const turn = game.turns.at(-1);
  turn.onFraction(0.5, { outcome: [0.5, 0.2, 0.3], nodes: 8 });
  // The flip turns the old board out, then the new one in.
  const held = [];
  game.context.animationPlan = (action) => (action.type === 'flip'
    ? { outClass: 'anim-flip-out', inClass: 'anim-flip-in', outMs: 320, inMs: 420 } : null);
  game.context.pause = () => new Promise((resolve) => held.push(resolve));
  turn.finish({ action: { type: 'flip' }, score: 0, depth: 0, nodes: 16, evaluations: 17, elapsedMs: 120,
    solver: 'neural', solved: false, backend: 'wasm', outcome: [0.7, 0.2, 0.1] });
  await settle();
  assert.equal(game.state.moveCount, 1, 'the old board is still showing');
  game.context.renderEvaluation();
  assert.deepEqual(game.panel(), heuristicFor(game.state.board), 'whatever renders the panel meanwhile');
  held.shift()();
  await settle();
  assert.equal(game.state.moveCount, 2);
  assert.equal(game.panel().label, 'AI 70% · Draw 20% · You 10%');
  held.shift()();
  await settle();
  assert.equal(game.state.currentPlayer, engine.RED);
  assert.equal(game.panel().label, 'AI 70% · Draw 20% · You 10%');
});

// A GPU that fails mid-search moves the network to the CPU, and the request
// says so through onSearch (neural-app.js) while the search waits for it.
// The search and the position are the same, so its estimate still holds.
test('a note from the search under way keeps the estimate it has reported', async () => {
  const game = page({ opponent: 'neural', startingPlayer: 1 });
  await game.context.performAction({ type: 'drop', column: 3 });
  await settle();
  const turn = game.turns.at(-1);
  turn.onSearch({ solver: 'neural-searching', note: 'Neural search · up to 16 simulations on webgpu' });
  turn.onFraction(0.5, { outcome: [0.5, 0.2, 0.3], nodes: 8 });
  const thinking = game.panel();
  assert.equal(thinking.label, 'AI 50% · Draw 20% · You 30%');
  for (const note of ['Neural search · the GPU failed; reading the network for the CPU (10.0 of 106.0 MB)',
    'Neural search · the GPU failed; starting the network on wasm', 'Neural search · up to 16 simulations on wasm']) {
    turn.onSearch({ solver: 'neural-searching', note });
    assert.equal(game.state.liveSearch.note, note);
    assert.deepEqual(game.panel(), thinking, note);
  }
  turn.onFraction(0.75, { outcome: [0.6, 0.2, 0.2], nodes: 12 });
  assert.equal(game.panel().label, 'AI 60% · Draw 20% · You 20%');

  // A new request has nothing to keep, even for the same position: Retry.
  game.context.stopAiWithError('The neural opponent failed: lost. Retry to restart it.');
  game.context.requestAiMove();
  await settle();
  game.turns.at(-1).onSearch({ solver: 'neural-searching', note: 'Neural search · up to 16 simulations on wasm' });
  assert.deepEqual(game.panel(), heuristicFor(game.state.board));
});

// A win in one is played before the network is loaded or asked anything
// (neural-app.js), so while its disc drops the panel keeps its heuristic.
// Then the result takes over.
test('a win in one claims no network estimate while its disc drops', async () => {
  const game = page({ opponent: 'neural', startingPlayer: 1 });
  // The AI stacks column 1 while Red plays elsewhere.
  for (const column of [0, 1, 0, 1, 6, 1]) {
    if (game.state.currentPlayer === engine.RED) await game.context.performAction({ type: 'drop', column });
    else await aiTurn(game, { live: [0.5, 0.2, 0.3], column, outcome: [0.6, 0.25, 0.15] });
  }
  const neural = readFileSync(new URL('../src/neural-app.js', import.meta.url), 'utf8');
  const request = vm.createContext({ immediateWinningActions: engine.immediateWinningActions,
    neuralLoadState: () => assert.fail('a win in one needs no network') });
  vm.runInContext(neural.slice(neural.indexOf('export async function')).replace('export ', ''), request);
  game.context.importNeuralApp = async () => ({ runNeuralRequest: request.runNeuralRequest });
  const held = [];
  game.context.pause = () => new Promise((resolve) => held.push(resolve));
  const move = game.context.performAction({ type: 'drop', column: 5 });
  await settle();
  held.shift()();                               // Red's disc lands, and the AI answers at once
  await move;
  await settle();
  assert.equal(game.state.moveCount, 8);
  assert.equal(game.state.status, 'playing', 'the AI\'s disc is still dropping');
  assert.deepEqual(game.panel(), heuristicFor(game.state.board));
  held.shift()();
  await settle();
  assert.equal(game.state.winner, engine.YELLOW);
  assert.deepEqual([game.panel().mode, game.panel().label], ['exact', 'Round result']);
});

// What the page hears from a neural request: the search's estimate as it
// runs, and with the move the estimate for the position the move makes.
test('a neural request passes on the search\'s estimate, then the chosen move\'s', async () => {
  const neural = readFileSync(new URL('../src/neural-app.js', import.meta.url), 'utf8');
  const actions = [{ type: 'drop', column: 0 }, { type: 'drop', column: 1 }];
  const context = vm.createContext({ immediateWinningActions: () => [], neuralLoadState: () => 'ready',
    loadNeuralNetwork: async () => ({ backend: 'wasm' }), waitFor: (promise) => promise, simulationsFor: () => 8,
    recordSearch() {}, searchOverran: () => false, invalidateNeuralNetwork() {}, bestAction,
    performance: { now: () => 0 },
    async searchPosition(_position, _evaluate, { onProgress }) {
      onProgress(4, 8, [0.5, 0.3, 0.2]);
      return { actions, visits: [1, 7], policy: [0.125, 0.875], actionValues: [-0.6, 0.5], terminalValues: [null, null],
        value: 0.36, completedSimulations: 8, evaluations: 9, outcome: [0.55, 0.26, 0.19],
        actionOutcomes: [[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]], networkOutcome: [0.4, 0.3, 0.3] };
    } });
  vm.runInContext(neural.slice(neural.indexOf('export async function')).replace('export ', ''), context);
  const heard = [];
  let played = null;
  await context.runNeuralRequest({ controller: new AbortController(),
    position: { board: engine.createBoard(6, 7), currentPlayer: engine.YELLOW, connect: 4, chaosMode: false } }, {
    isCurrent: () => true, shouldStop: () => false, onSearch() {}, onFraction: (...args) => heard.push(args),
    finish: (result) => { played = result; }, fail: (message) => assert.fail(message) });
  // Through JSON: what the request builds comes from another realm's Object.
  assert.deepEqual(JSON.parse(JSON.stringify(heard)), [[0.5, { outcome: [0.5, 0.3, 0.2], nodes: 4 }]]);
  assert.equal(played.action, actions[1]);
  assert.deepEqual(JSON.parse(JSON.stringify(played.outcome)), [0.6, 0.3, 0.1]);
});
