import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

import {
  ACTION_DROP, ACTION_FLIP, ACTION_ROTATE_CCW, ACTION_ROTATE_CW, RED, YELLOW,
  boardDimensions, createBoard, getDropRow,
} from '../src/engine.js';

const app = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');

function definition(name) {
  const start = app.search(new RegExp(`^function ${name}\\(`, 'm'));
  assert.notEqual(start, -1, `missing ${name} in app.js`);
  return app.slice(start, app.indexOf('\n}\n', start) + 2);
}

/** A whole line of app.js, by its start: a variable or a listener's registration. */
function line(start) {
  const found = app.split('\n').find((text) => text.startsWith(start));
  assert.ok(found?.endsWith(';'), `app.js has no one-line statement starting ${start}`);
  return found;
}

class Target {
  listeners = {};
  addEventListener(type, listener) { (this.listeners[type] ??= []).push(listener); }
  dispatch(type, event) { for (const listener of this.listeners[type] ?? []) listener(event); }
}
class Button extends Target {
  click() { this.dispatch('click', {}); }
}
class Field extends Target {}

// One press of a key is one action, however long the key stays down. This is
// the page's key handling as app.js has it: its two keydown handlers, the
// functions they call and the lines that register them, with the round's
// moves, Undo and New round recorded in `done` instead of played. `thinking`
// is the AI's search while it is the AI's turn.
function page({ opponent = 'neural', chaosMode = false, thinking = null } = {}) {
  // A third place that listens for keys needs its own look at held ones.
  assert.equal(app.split(".addEventListener('key").length - 1, 2, 'the board and the document listen for keys');
  const done = [];
  const document = Object.assign(new Target(), { body: new Target() });
  document.activeElement = document.body;
  const board = Object.assign(new Target(), {
    focus() { document.activeElement = this; this.dispatch('focus', {}); },
  });
  const elements = {
    board, moveNowButton: new Button(), flipButton: new Button(),
    thinkingBarRow: { hidden: true }, thinkingBar: { value: 0, removeAttribute() { this.value = null; } },
    selectedColumnStatus: { textContent: '' },
    resultDialog: { open: false }, rulesDialog: { open: false }, downloadDialog: { open: false },
  };
  const state = {
    config: { opponent, chaosMode }, status: 'playing', busy: false, board: createBoard(6, 7), selectedColumn: 3,
    currentPlayer: thinking ? YELLOW : RED, aiThinking: Boolean(thinking), aiRequest: thinking ? { id: 1 } : null,
    liveSearch: thinking, moveNowRequested: false,
  };
  const context = vm.createContext({
    state, elements, document, RED, ACTION_DROP, ACTION_FLIP, ACTION_ROTATE_CCW, ACTION_ROTATE_CW,
    boardDimensions, getDropRow, HTMLButtonElement: Button, HTMLInputElement: Field,
    HTMLSelectElement: class {}, HTMLTextAreaElement: class {},
    isLegalAiAction: (action) => Boolean(action), renderBoardSelection() {},
    performAction: (action) => { done.push({ ...action }); },
    undoTurn: () => { done.push('undo'); }, restartRound: () => { done.push('new round'); },
  });
  vm.runInContext([
    ...['isAiGame', 'canHumanAct', 'renderThinkingBar', 'moveNow', 'announceSelectedColumn', 'chooseColumn',
      'moveSelectedColumn', 'dropSelectedColumn', 'isTypingTarget', 'handleBoardKeydown', 'handleGlobalKeydown',
    ].map(definition),
    line('const pressedOn = '), line("elements.board.addEventListener('keydown', "),
    line("elements.board.addEventListener('focus', "), line("document.addEventListener('keydown', "),
    line("elements.moveNowButton.addEventListener('click', "), line("elements.flipButton.addEventListener('click', "),
  ].join('\n'), context);
  // The browser's part. A keydown goes to the element with the focus and on
  // to the document. Unless the page cancels it, a button takes an Enter as
  // a click and a field as its form's submission, and a button that sees a
  // Space go down clicks when the key comes up. While a key stays down, its
  // keydown repeats.
  let armed = null;
  const keydown = (key, { repeat = false, shiftKey = false } = {}) => {
    const target = document.activeElement;
    const event = { key, repeat, shiftKey, ctrlKey: false, metaKey: false, altKey: false, target,
      defaultPrevented: false, preventDefault() { this.defaultPrevented = true; } };
    target.dispatch('keydown', event);
    document.dispatch('keydown', event);
    if (!event.defaultPrevented && key === 'Enter' && target instanceof Button) target.click();
    if (!event.defaultPrevented && key === 'Enter' && target instanceof Field) done.push('submit');
    if (!event.defaultPrevented && key === ' ' && target instanceof Button) armed = target;
    return event;
  };
  const held = (key, options) => keydown(key, { ...options, repeat: true });
  const keyup = (key) => {
    if (key === ' ' && armed === document.activeElement) armed.click();
    armed = null;
  };
  return { state, elements, document, context, done, keydown, held, keyup };
}

// Enter presses a button as the key goes down. Since Move now disables itself
// and hands its focus to the board, the repeats of an Enter held on it went
// to the board, which dropped a disc for the player at the first one after
// the AI's move.
test('an Enter still held from Move now drops no disc when the AI has moved', () => {
  const { state, elements, document, context, done, keydown, held } = page({
    thinking: { solver: 'neural-searching', note: 'Neural search · up to 512 simulations on webgpu', fraction: 40 / 512 } });
  context.renderThinkingBar();
  document.activeElement = elements.moveNowButton;
  keydown('Enter');
  assert.equal(state.moveNowRequested, true, 'the keydown presses Move now');
  assert.equal(document.activeElement, elements.board, 'and the board takes the focus the disabled button held');
  for (let repeat = 0; repeat < 3; repeat += 1) held('Enter');
  // The search stops, the AI's disc lands and the turn is the player's.
  Object.assign(state, { aiThinking: false, aiRequest: null, liveSearch: null, currentPlayer: RED });
  assert.equal(context.canHumanAct(), true);
  for (let repeat = 0; repeat < 3; repeat += 1) assert.equal(held('Enter').defaultPrevented, true);
  assert.deepEqual(done, [], 'the key is still the press that asked for the AI\'s move');
  // Released and pressed again, it is the player's own move.
  keydown('Enter');
  assert.deepEqual(done, [{ type: ACTION_DROP, column: 3 }]);
});

// The board played for the other player too, or again after every reply, for
// as long as the key that dropped a disc stayed down.
test('Enter or Space held on the board drops one disc', () => {
  for (const key of ['Enter', ' ']) {
    const { elements, done, keydown, held } = page({ opponent: 'human' });
    elements.board.focus();
    assert.equal(keydown(key).defaultPrevented, true);
    assert.deepEqual(done, [{ type: ACTION_DROP, column: 3 }]);
    // The disc has landed and the next move can be made.
    for (let repeat = 0; repeat < 3; repeat += 1) {
      assert.equal(held(key).defaultPrevented, true, 'a held Space does not scroll the page either');
    }
    assert.deepEqual(done, [{ type: ACTION_DROP, column: 3 }], JSON.stringify(key));
  }
});

// Holding an arrow is how a wide board is crossed, and choosing a column
// plays nothing.
test('the arrows, Home and End repeat: they choose a column and drop nothing', () => {
  const { state, elements, done, held } = page({ opponent: 'human' });
  elements.board.focus();
  held('ArrowRight');
  held('ArrowRight');
  assert.equal(state.selectedColumn, 5);
  assert.equal(elements.selectedColumnStatus.textContent, 'Column 6 selected. Press Enter or Space to drop.');
  held('Home');
  assert.equal(state.selectedColumn, 0);
  held('End');
  assert.equal(state.selectedColumn, 6);
  held('ArrowLeft');
  assert.equal(state.selectedColumn, 5);
  assert.deepEqual(done, []);
});

// F and R play a move: held down, they played for both players in turn, and
// a third return to the same position ends a Chaos round as a draw.
test('a shortcut key held down acts once', () => {
  const { done, keydown, held } = page({ opponent: 'human', chaosMode: true });
  const expected = [];
  for (const [key, shiftKey, action] of [['f', false, { type: ACTION_FLIP }],
    ['r', false, { type: ACTION_ROTATE_CW }], ['R', true, { type: ACTION_ROTATE_CCW }],
    ['u', false, 'undo'], ['n', false, 'new round']]) {
    assert.equal(keydown(key, { shiftKey }).defaultPrevented, true);
    expected.push(action);
    assert.deepEqual(done, expected, key);
    for (let repeat = 0; repeat < 3; repeat += 1) {
      assert.equal(held(key, { shiftKey }).defaultPrevented, true, 'the key is still the page\'s');
    }
    assert.deepEqual(done, expected, `${key} held`);
  }
});

// To a button every repeat of a held Enter is another click, and the
// transform buttons play a move at each: a button that was still focused
// when its move was over played the next one too.
test('an Enter held on a button presses it once', () => {
  const { elements, document, done, keydown, held } = page({ opponent: 'human', chaosMode: true });
  document.activeElement = elements.flipButton;
  keydown('Enter');
  assert.deepEqual(done, [{ type: ACTION_FLIP }]);
  for (let repeat = 0; repeat < 3; repeat += 1) assert.equal(held('Enter').defaultPrevented, true);
  assert.deepEqual(done, [{ type: ACTION_FLIP }], 'the other player has not flipped too');
  // Pressed again, it flips again.
  keydown('Enter');
  assert.deepEqual(done, [{ type: ACTION_FLIP }, { type: ACTION_FLIP }]);
});

// A dialog focuses one of its buttons as it opens, and the download prompt
// the one that agrees. Opened by the AI's turn while the key that played the
// move was still down, it was answered by that key: by Enter at its next
// repeat, by Space as it came up. A settings field that Change rules or
// Change opponent focuses took a held Enter as its form's submission, which
// starts a round.
test('a key held while the focus moves does nothing to what it was not pressed on', () => {
  const { elements, document, done, keydown, held, keyup } = page({ opponent: 'human' });
  const drop = { type: ACTION_DROP, column: 3 };
  const download = new Button();
  download.addEventListener('click', () => done.push('download'));
  for (const key of ['Enter', ' ']) {
    done.length = 0;
    elements.board.focus();
    keydown(key);
    document.activeElement = download;
    for (let repeat = 0; repeat < 3; repeat += 1) assert.equal(held(key).defaultPrevented, true);
    keyup(key);
    assert.deepEqual(done, [drop], `${JSON.stringify(key)} played the move and answered nothing`);
  }
  const rows = new Field();
  const changeRules = new Button();
  changeRules.addEventListener('click', () => { document.activeElement = rows; });
  done.length = 0;
  document.activeElement = changeRules;
  keydown('Enter');
  assert.equal(document.activeElement, rows);
  for (let repeat = 0; repeat < 3; repeat += 1) held('Enter');
  // Another key pressed meanwhile does not make the held one the field's.
  keydown('Shift');
  held('Enter');
  assert.deepEqual(done, [], 'the field is not submitted by the key that opened it');
  // A press on the element itself is its own: Enter submits the field, and a
  // button clicks once for a Space however long it is held.
  keydown('Enter');
  assert.deepEqual(done, ['submit']);
  document.activeElement = download;
  keydown(' ');
  for (let repeat = 0; repeat < 3; repeat += 1) assert.equal(held(' ').defaultPrevented, false);
  assert.deepEqual(done, ['submit'], 'not before the key comes up');
  keyup(' ');
  assert.deepEqual(done, ['submit', 'download']);
});
