import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import * as engine from '../src/engine.js';
import * as storage from '../src/round-storage.js';
import { createScoreStore } from '../src/score-store.js';

// Exercise the real startup/save/restore controller. Rendering and physical
// storage are replaced; rule validation and snapshots remain production code.
const source = readFileSync(new URL('../src/app.js', import.meta.url), 'utf8');
const controller = source.slice(source.indexOf('function saveRound()'), source.indexOf('function applySettingsAndStartRound()'));
const perform = source.slice(source.indexOf('async function performAction('), source.indexOf('\nfunction isLegalAiAction('));
const boot = source.slice(source.lastIndexOf('const savedRound ='));
let nextId = 0;
const mapStorage = (data) => ({
  getItem: (key) => data.get(key) ?? null,
  setItem: (key, value) => data.set(key, value),
  removeItem: (key) => data.delete(key),
});
function page(config, data = new Map(), tabData = new Map()) {
  const state = { config: engine.normalizeConfig(config), scores: { 1: 0, 2: 0, draw: 0 },
    history: [], version: 0, gameFirstLayout: false, touchHintDismissed: false };
  const aiCalls = [];
  const context = { ...engine, ...storage, state,
    resultId: () => `round-${++nextId}`, populateSettingsForm() {}, renderAll() {}, setSettingsExpanded() {},
    renderGuidance() {}, renderStatus() {}, renderActions() {}, disposeAiWorker() {}, showResultDialog() {},
    animationPlan: () => null, pause: async () => {},
    canHumanAct: () => state.status === 'playing' && !state.busy,
    scoreStore: createScoreStore({ indexedDB: null }), scoreWarning() {},
    acceptScore(result) { state.scores = result.scores; return result.receipt; },
    cancelAiSearch() { state.aiThinking = false; }, closeResultDialog() {}, clearBoardAnimations() {}, refreshScores() {},
    requestAiMove() { aiCalls.push(state.moveCount); state.aiThinking = true; },
    stopAiWithError(message) { state.aiError = message; state.aiThinking = false; },
    isAiGame: () => state.config.opponent !== 'human',
    pushSnapshot(receipt = null) { state.history.push({ ...storage.makeSnapshot(state), scoreReceipt: receipt }); },
    restoreSnapshot(snapshot, options) { storage.restoreSnapshot(state, snapshot, options); },
    loadJson: (key, fallback) => data.has(key) ? JSON.parse(data.get(key)) : fallback,
    saveJson: (key, value) => data.set(key, JSON.stringify(value)),
    localStorage: mapStorage(data),
    roundStore: storage.createRoundStore({ sharedStorage: () => mapStorage(data), tabStorage: () => mapStorage(tabData) }),
  };
  vm.createContext(context);
  vm.runInContext(controller + '\n' + perform + '\n' + boot, context);
  return { state, data, tabData, aiCalls, context,
    drop: (column) => context.performAction({ type: 'drop', column }) };
}

test('an opening neural crash restores move zero without relaunching inference', () => {
  const config = { opponent: 'neural', startingPlayer: 2 };
  const first = page(config);
  assert.deepEqual(first.aiCalls, [0], 'a fresh game starts normally');
  const saved = first.data.get(storage.ROUND_KEY);
  assert.ok(saved, 'the opening position must be saved before inference');
  for (let reload = 0; reload < 3; reload++) {
    const restored = page(config, first.data, first.tabData);
    assert.deepEqual(restored.aiCalls, [], 'reload cannot loop through neural startup');
    assert.equal(restored.state.moveCount, 0);
    assert.equal(restored.state.currentPlayer, 2);
    assert.match(restored.state.aiError, /board is restored.*Retry/);
    assert.equal(restored.data.get(storage.ROUND_KEY), saved);
  }
});

test('a neural reload preserves a rotated board, repetition history and Undo snapshots', () => {
  const config = { opponent: 'neural', startingPlayer: 1, chaosMode: true };
  const first = page(config);
  let state = first.state;
  for (const action of [{ type: 'drop', column: 3 }, { type: 'rotateCW' }, { type: 'drop', column: 2 }]) {
    const result = engine.applyAction(state.board, action, state.currentPlayer);
    state.board = result.board;
    state.lastMover = state.currentPlayer;
    state.currentPlayer = engine.otherPlayer(state.currentPlayer);
    state.moveCount++;
    state.selectedColumn = 2;
    state.lastMove = action.type === 'drop' ? { row: result.row, column: result.column } : null;
    state.repetitionCounts.set(engine.positionKey(state.board, state.currentPlayer, 4, true), 1);
    first.context.pushSnapshot();
  }
  first.context.saveRound();
  const history = JSON.stringify(state.history);
  const restored = page(config, first.data, first.tabData);
  assert.deepEqual(restored.aiCalls, []);
  assert.equal(JSON.stringify(restored.state.history), history);
  assert.deepEqual(restored.state.board, state.board);
  assert.deepEqual(structuredClone([...restored.state.repetitionCounts]), structuredClone([...state.repetitionCounts]));
  // The existing Retry handler invokes requestAiMove without resetting state.
  restored.context.requestAiMove();
  assert.deepEqual(restored.aiCalls, [3]);
});

test('human turns restore normally and other AI opponents still resume automatically', () => {
  for (const config of [{ opponent: 'neural', startingPlayer: 1 }, { opponent: 'medium', startingPlayer: 2 }]) {
    const first = page(config);
    const restored = page(config, first.data, first.tabData);
    assert.deepEqual(restored.aiCalls, config.startingPlayer === 2 ? [0] : []);
    assert.equal(restored.state.aiError, null);
  }
});

test('finishing an older tab does not erase another tab\'s saved round', async () => {
  const config = { opponent: 'human' }, data = new Map();
  const first = page(config, data);
  for (const column of [0, 1, 0, 1, 0, 1]) await first.drop(column);
  const second = page(config, data);
  second.context.startRound();
  await second.drop(3);
  const saved = data.get(storage.ROUND_KEY);
  assert.notEqual(first.state.roundId, second.state.roundId);
  await first.drop(0);
  assert.equal(first.state.status, 'won');
  const restored = page(config, data, second.tabData);
  assert.equal(restored.state.moveCount, 1);
  assert.equal(restored.state.roundId, second.state.roundId);
  assert.equal(data.get(storage.ROUND_KEY), saved);
});

test('interleaved tabs reload their own board, rules and Undo history', async () => {
  const data = new Map(), first = page({ opponent: 'human' }, data);
  await first.drop(0);
  const second = page(first.state.config, data);
  second.context.startRound({ opponent: 'human', rows: 4, cols: 4, chaosMode: true });
  await second.drop(3);
  await first.drop(1);
  for (const tab of [second, first]) {
    const sharedConfig = JSON.parse(data.get(storage.SETTINGS_KEY));
    const restored = page(sharedConfig, data, tab.tabData);
    assert.equal(restored.state.roundId, tab.state.roundId);
    assert.deepEqual(restored.state.config, tab.state.config);
    assert.equal(JSON.stringify(restored.state.history), JSON.stringify(tab.state.history));
  }
});

test('a finished tab reloads a new round even when another tab has an active save', async () => {
  const config = { opponent: 'human' }, data = new Map();
  const first = page(config, data);
  for (const column of [0, 1, 0, 1, 0, 1, 0]) await first.drop(column);
  const second = page(config, data);
  await second.drop(3);
  const restored = page(config, data, first.tabData);
  assert.equal(restored.state.moveCount, 0);
  assert.notEqual(restored.state.roundId, second.state.roundId);
  assert.equal(page(config, data, second.tabData).state.moveCount, 1);
});

test('legacy shared saves migrate to tab recovery on first load', async () => {
  const config = { opponent: 'human' }, original = page(config);
  await original.drop(2);
  const restored = page(config, original.data);
  assert.equal(restored.state.moveCount, 1);
  assert.equal(restored.state.roundId, original.state.roundId);
  assert.equal(restored.tabData.get(storage.ROUND_KEY), original.data.get(storage.ROUND_KEY));
});

test('round recovery tolerates blocked storage and only clears its own shared save', () => {
  const data = new Map();
  const blocked = () => { throw new DOMException('Storage blocked', 'SecurityError'); };
  const store = storage.createRoundStore({ sharedStorage: () => mapStorage(data), tabStorage: blocked });
  store.save({ roundId: 'other-round' });
  store.clear('finished-round');
  assert.equal(store.read().roundId, 'other-round');
  store.clear('other-round');
  assert.equal(store.read(), null);
  const tab = new Map();
  const tabOnly = storage.createRoundStore({ sharedStorage: blocked, tabStorage: () => mapStorage(tab) });
  tabOnly.save({ roundId: 'tab-round' });
  assert.equal(tabOnly.read().roundId, 'tab-round');
  tabOnly.clear('tab-round');
  assert.equal(tabOnly.read(), null);
});
