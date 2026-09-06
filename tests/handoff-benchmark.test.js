import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import * as engine from '../src/engine.js';
import { choosePreparedMove } from '../src/ai-worker.js';
import { createEvaluator, playGame } from '../scripts/neural-vs-brutal.mjs';

const { RED, YELLOW, createBoard, applyAction, otherPlayer, positionKey, legalActions, sameAction } = engine;
const flip = { type: 'flip' };
const noOp = () => {};
const app = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');
function definition(name) {
  const start = app.search(new RegExp(`^(?:async )?function ${name}\\(`, 'm'));
  assert.notEqual(start, -1, `missing controller ${name}`);
  return app.slice(start, app.indexOf('\n}\n', start) + 2);
}
function positionAfter(columns) {
  let board = createBoard(6, 7), currentPlayer = RED;
  const counts = new Map([[positionKey(board, currentPlayer, 4, true), 1]]);
  for (const column of columns) {
    board = applyAction(board, { type: 'drop', column }, currentPlayer).board;
    currentPlayer = otherPlayer(currentPlayer);
    const key = positionKey(board, currentPlayer, 4, true);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return { board, currentPlayer, connect: 4, chaosMode: true, startingPlayer: RED, repetitionCounts: [...counts] };
}

test('off-policy handoff uses general Brutal without weakening strict certificate errors', async () => {
  const position = positionAfter([0, 0, 0]);
  const options = { difficulty: 'brutal', aiPlayer: YELLOW, timeBudgetMs: 10, useChaosProof: false };
  await assert.rejects(choosePreparedMove(position, options), /does not cover this reachable position/);
  const result = await choosePreparedMove(position, { ...options, useChaosPolicy: false });
  assert.ok(legalActions(position.board, true).some((a) => sameAction(a, result.action)));
  const covered = await choosePreparedMove(positionAfter([0, 3, 0]), options);
  assert.ok(covered.action);
});

test('prepared moves retain Classic book setup and signal search readiness', async () => {
  let ready = 0;
  const result = await choosePreparedMove({ board: createBoard(6, 7), currentPlayer: RED,
    startingPlayer: RED, connect: 4, chaosMode: false }, {
    difficulty: 'brutal', aiPlayer: RED, timeBudgetMs: 10, onSearchStart() { ready += 1; },
  });
  assert.equal(ready, 1);
  assert.equal(result.action.type, 'drop');
  assert.match(result.solver, /book/);
});

test('prepared move does not search an already cancelled request', async () => {
  const controller = new AbortController(); controller.abort();
  await assert.rejects(choosePreparedMove(positionAfter([0, 0, 0]), {
    difficulty: 'brutal', useChaosPolicy: false, signal: controller.signal,
    onSearchStart() { assert.fail('cancelled request started searching'); },
  }), { name: 'AbortError' });
});

for (const neuralPlays of [RED, YELLOW]) {
  test(`benchmark stops at four flips and supplies both histories (Neural=${neuralPlays})`, async () => {
    const positions = [], options = [];
    const result = await playGame(createBoard(6, 7), 4, true, neuralPlays, {
      openingPlies: 0, maxPlies: 10,
      neuralSearch: async (p) => { positions.push(p); return { actions: [flip], visits: [1] }; },
      brutalMove: async (p, o) => { positions.push(p); options.push(o); return { action: flip }; },
    });
    assert.equal(result, 0);
    assert.equal(positions.length, 4);
    assert.deepEqual(positions.map((p) => new Map(p.repetitionCounts).get(positionKey(p.board, p.currentPlayer, 4, true))), [1, 1, 2, 2]);
    assert.deepEqual(positions.map((p) => p.currentPlayer), [RED, YELLOW, RED, YELLOW]);
    assert.ok(options.every((o) => o.useChaosPolicy === true));
  });
}

test('four opening flips end before either opponent is asked for an extra turn', async () => {
  let plies = 0;
  assert.equal(await playGame(createBoard(6, 7), 4, true, RED, {
    chooseOpening: () => { plies += 1; return flip; },
    neuralSearch: () => assert.fail('requested fifth ply'), brutalMove: () => assert.fail('requested fifth ply'),
  }), 0);
  assert.equal(plies, 4);
});

test('randomized openings select prepared general Brutal, with updated repetition history', async () => {
  let captured;
  assert.equal(await playGame(createBoard(6, 7), 4, true, RED, {
    openingPlies: 1, maxPlies: 2, chooseOpening: () => flip,
    brutalMove: async (position, options) => { captured = { position, options }; return { action: flip }; },
  }), null);
  assert.equal(captured.options.useChaosPolicy, false);
  assert.equal(captured.options.difficulty, 'brutal');
  assert.equal(captured.position.repetitionCounts.length, 2);
});

test('evaluator forwards repetition flags and clears them on the next position', async () => {
  const observed = [];
  const ort = { Tensor: class { constructor(_type, data, dims) { this.data = data; this.dims = dims; } } };
  const session = { async run({ planes }) {
    observed.push([planes.data[500], planes.data[600]]);
    assert.deepEqual(planes.dims, [1, 7, 10, 10]);
    return { policy: { data: new Float32Array(13) }, value: { data: new Float32Array(3) }, q: { data: new Float32Array(39) } };
  } };
  const evaluate = createEvaluator(ort, session);
  for (const repetition of [2, 1, 0]) await evaluate(createBoard(6, 7), RED, [], 4, true, repetition);
  assert.deepEqual(observed, [[1, 1], [1, 0], [0, 0]]);
});

test('benchmark rejects invalid moves and distinguishes a ply cap from a draw', async () => {
  await assert.rejects(playGame(createBoard(6, 7), 4, true, YELLOW, {
    openingPlies: 0, brutalMove: async () => ({ action: null }),
  }), /illegal move/);
  assert.equal(await playGame(createBoard(6, 7), 4, true, RED, { maxPlies: 1, chooseOpening: () => flip }), null);
});

test('handoff controller saves general mode and puts it on the very next request', () => {
  const state = { ...positionAfter([0, 0, 0]), config: { rows: 6, cols: 7, connect: 4,
    chaosMode: true, startingPlayer: RED, opponent: 'neural' }, version: 1, aiRequestId: 3,
    status: 'playing', busy: false, aiThinking: false, history: [{}], useChaosPolicy: true };
  let saved, request, cancelled = 0;
  const context = vm.createContext({ ...engine, state, AbortController,
    SETTINGS_KEY: 'settings', populateSettingsForm: noOp, saveJson: noOp, renderAll: noOp, renderAiState: noOp,
    cancelAiSearch() { cancelled += 1; state.aiThinking = false; },
    saveRound() { saved = state.useChaosPolicy; }, isAiGame: () => true,
    postToWorker(r) { request = r; },
  });
  vm.runInContext(`${definition('requestAiMove')}\n${definition('switchToBrutal')}\nswitchToBrutal();`, context);
  assert.equal(cancelled, 1);
  assert.equal(saved, false);
  assert.equal(state.config.opponent, 'brutal');
  assert.equal(request.options.useChaosPolicy, false);
});

for (const actionType of ['rotateCW', 'rotateCCW', 'flip']) {
  test(`old ${actionType} completion cannot remove a restarted animation`, async () => {
    // Controlled pauses are stronger than a wall-clock race: both animations
    // have entered their incoming phase when the obsolete callback is released.
    const actualType = actionType === 'rotateCW' ? engine.ACTION_ROTATE_CW
      : actionType === 'rotateCCW' ? engine.ACTION_ROTATE_CCW : engine.ACTION_FLIP;
    const classes = new Set(), pauses = [];
    const state = { board: createBoard(6, 7), config: { connect: 4, chaosMode: true },
      currentPlayer: RED, status: 'playing', busy: false, version: 0, moveCount: 0,
      touchHintDismissed: true, repetitionCounts: new Map() };
    const context = vm.createContext({ ...engine, state, elements: { boardFrame: { classList: {
      add: (c) => classes.add(c), remove: (c) => classes.delete(c),
    } } }, canHumanAct: () => true, renderAll: noOp, renderStatus: noOp, renderActions: noOp,
      clearBoardAnimations: () => classes.clear(),
      animationPlan: () => ({ outClass: 'out', inClass: 'in', outMs: 1, inMs: 2 }),
      pause: () => new Promise((resolve) => pauses.push(resolve)),
    });
    vm.runInContext(definition('performAction'), context);
    const first = context.performAction({ type: actualType });
    pauses.shift()(); await new Promise(setImmediate);
    assert.equal(classes.has('in'), true);
    state.version += 1; state.busy = false; state.board = createBoard(6, 7); classes.clear();
    const second = context.performAction({ type: actualType });
    const finishOld = pauses.shift();
    pauses.shift()(); await new Promise(setImmediate);
    assert.equal(classes.has('in'), true);
    finishOld(); await first;
    assert.equal(classes.has('in'), true);
    assert.equal(state.busy, true);
    // Cancel the second round to finish the harness without unrelated UI work.
    state.version += 1; pauses.shift()(); await second;
  });
}
