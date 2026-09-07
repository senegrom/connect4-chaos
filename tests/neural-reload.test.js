import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import * as engine from '../src/engine.js';
import * as storage from '../src/round-storage.js';

// Exercise the real startup/save/restore controller. Rendering and physical
// storage are replaced; rule validation and snapshots remain production code.
const source = readFileSync(new URL('../src/app.js', import.meta.url), 'utf8');
const controller = source.slice(source.indexOf('function saveRound()'), source.indexOf('function applySettingsAndStartRound()'));
const boot = source.slice(source.lastIndexOf('const savedRound ='));
function page(config, data = new Map()) {
  const state = { config: engine.normalizeConfig(config), scores: { 1: 0, 2: 0, draw: 0 },
    history: [], version: 0, gameFirstLayout: false, touchHintDismissed: false };
  const aiCalls = [];
  const context = { ...engine, ...storage, state,
    resultId: () => 'round-1', populateSettingsForm() {}, renderAll() {}, setSettingsExpanded() {},
    cancelAiSearch() { state.aiThinking = false; }, closeResultDialog() {}, clearBoardAnimations() {}, refreshScores() {},
    requestAiMove() { aiCalls.push(state.moveCount); state.aiThinking = true; },
    stopAiWithError(message) { state.aiError = message; state.aiThinking = false; },
    isAiGame: () => state.config.opponent !== 'human',
    pushSnapshot() { state.history.push(storage.makeSnapshot(state)); },
    restoreSnapshot(snapshot, options) { storage.restoreSnapshot(state, snapshot, options); },
    loadJson: (key, fallback) => data.has(key) ? JSON.parse(data.get(key)) : fallback,
    saveJson: (key, value) => data.set(key, JSON.stringify(value)),
    localStorage: { removeItem: (key) => data.delete(key) },
  };
  vm.createContext(context);
  vm.runInContext(controller + '\n' + boot, context);
  return { state, data, aiCalls, context };
}

test('an opening neural crash restores move zero without relaunching inference', () => {
  const config = { opponent: 'neural', startingPlayer: 2 };
  const first = page(config);
  assert.deepEqual(first.aiCalls, [0], 'a fresh game starts normally');
  const saved = first.data.get(storage.ROUND_KEY);
  assert.ok(saved, 'the opening position must be saved before inference');
  for (let reload = 0; reload < 3; reload++) {
    const restored = page(config, first.data);
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
  const restored = page(config, first.data);
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
    const restored = page(config, first.data);
    assert.deepEqual(restored.aiCalls, config.startingPlayer === 2 ? [0] : []);
    assert.equal(restored.state.aiError, null);
  }
});
