import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const app = await readFile(new URL('../src/app.js', import.meta.url), 'utf8');

function definition(name) {
  const start = app.search(new RegExp(`^function ${name}\\(`, 'm'));
  assert.notEqual(start, -1, `missing ${name} in app.js`);
  return app.slice(start, app.indexOf('\n}\n', start) + 2);
}

// Move now beside the thinking bar, as app.js's renderThinkingBar and moveNow
// draw and handle it and its announceSelectedColumn tells a screen reader. A
// neural search that hears the press before its minimum runs on to it
// (neural-app.js), which on a slow phone takes seconds, so the press has to
// show, and be heard, at once. `focused` says whether the button held the
// focus: a click in Safari never gives it. Any move a search names is legal.
function page(liveSearch, { focused = true } = {}) {
  const button = { hidden: true, disabled: false, textContent: 'Move now', title: 'Play the best move found so far' };
  const elements = {
    thinkingBarRow: { hidden: true },
    thinkingBar: { value: 0, removeAttribute() { this.value = null; } },
    moveNowButton: button,
    // The board announces itself as it takes the focus, as app.js wires it.
    board: { focused: 0, focus() { this.focused += 1; context.announceSelectedColumn(); } },
    selectedColumnStatus: { textContent: '' },
  };
  const state = { aiThinking: true, aiRequest: { id: 1 }, liveSearch, moveNowRequested: false };
  const context = vm.createContext({ state, elements, document: { activeElement: focused ? button : null },
    isLegalAiAction: (action) => Boolean(action), canHumanAct: () => false,
    cancelAiSearch: () => assert.fail('a neural search stops by itself'),
    performAction: () => assert.fail('a neural search plays its own move') });
  vm.runInContext(['renderThinkingBar', 'moveNow', 'announceSelectedColumn'].map(definition).join('\n'), context);
  const shown = () => [button.hidden, button.disabled, button.textContent];
  const heard = () => elements.selectedColumnStatus.textContent;
  return { state, elements, context, shown, heard };
}

const searching = (note, fraction) => ({ solver: 'neural-searching', note, fraction });
// The thinking line is hidden from screen readers, so this is all they hear
// of the press until the move.
const pressed = 'Move now: the AI moves once its search has its minimum. Board controls are temporarily unavailable.';
// What the button promises when hovered: a neural search plays the best move
// so far only once it has its minimum, and once pressed it waits for it.
const offer = 'Play the best move found so far, once the search has its minimum';
const waiting = 'The AI moves once its search has its minimum';

test('Move now on a neural search says the move is coming until it comes', () => {
  const { state, elements, context, shown, heard } = page(
    searching('Neural search · up to 512 simulations on wasm', 2 / 512));
  context.renderThinkingBar();
  assert.deepEqual(shown(), [false, false, 'Move now']);
  assert.equal(elements.moveNowButton.title, offer);
  context.announceSelectedColumn();
  assert.equal(heard(), 'AI is thinking. Board controls are temporarily unavailable.');
  context.moveNow();
  assert.equal(state.moveNowRequested, true, 'the search hears it at its next simulation');
  assert.deepEqual(shown(), [false, true, 'Moving…']);
  assert.equal(elements.moveNowButton.title, waiting);
  assert.equal(elements.board.focused, 1, 'the focus the disabled button held goes on to the board');
  assert.equal(heard(), pressed, 'the board says so as it takes the focus');
  // The search runs on to its minimum and says so; the button stays as it is.
  state.liveSearch = searching('Moving after 16 simulations, the minimum, on wasm', 3 / 16);
  context.renderThinkingBar();
  assert.equal(elements.thinkingBar.value, 3 / 16);
  assert.deepEqual(shown(), [false, true, 'Moving…']);
  assert.equal(elements.moveNowButton.title, waiting);
  // The next request starts unpressed (requestAiMove), and so does the button.
  state.moveNowRequested = false;
  state.liveSearch = searching('Neural search · up to 512 simulations on wasm');
  context.renderThinkingBar();
  assert.deepEqual(shown(), [false, false, 'Move now']);
  assert.equal(elements.moveNowButton.title, offer);
});

// Any other search has no minimum: Move now plays its best move so far at
// once, and the button says only that, after a neural search too (the
// opponent can change between moves).
test('Move now on another search promises its best move so far at once', () => {
  const { state, elements, context, shown } = page(searching('Neural search · up to 512 simulations on wasm'));
  context.renderThinkingBar();
  assert.equal(elements.moveNowButton.title, offer);
  state.liveSearch = { solver: 'bitboard', depth: 6, maximumDepth: 12, action: { type: 'drop', column: 3 } };
  context.renderThinkingBar();
  assert.deepEqual(shown(), [false, false, 'Move now']);
  assert.equal(elements.moveNowButton.title, 'Play the best move found so far');
});

test('Move now moves no focus the button did not hold, and is heard all the same', () => {
  const { elements, context, shown, heard } = page(searching('Neural search · up to 16 simulations on wasm', 0.5),
    { focused: false });
  context.renderThinkingBar();
  context.moveNow();
  assert.deepEqual(shown(), [false, true, 'Moving…']);
  assert.equal(elements.board.focused, 0);
  assert.equal(heard(), pressed);
});
