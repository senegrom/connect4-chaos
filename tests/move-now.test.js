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
// draw and handle it. A neural search that hears the press before its
// minimum runs on to it (neural-app.js), which on a slow phone takes seconds,
// so the press has to show at once. `focused` says whether the button held
// the focus: a click in Safari never gives it.
function page(liveSearch, { focused = true } = {}) {
  const button = { hidden: true, disabled: false, textContent: 'Move now' };
  const elements = {
    thinkingBarRow: { hidden: true },
    thinkingBar: { value: 0, removeAttribute() { this.value = null; } },
    moveNowButton: button,
    board: { focused: 0, focus() { this.focused += 1; } },
  };
  const state = { aiThinking: true, aiRequest: { id: 1 }, liveSearch, moveNowRequested: false };
  const context = vm.createContext({ state, elements, document: { activeElement: focused ? button : null },
    isLegalAiAction: () => false,
    cancelAiSearch: () => assert.fail('a neural search stops by itself'),
    performAction: () => assert.fail('a neural search plays its own move') });
  vm.runInContext(['renderThinkingBar', 'moveNow'].map(definition).join('\n'), context);
  const shown = () => [button.hidden, button.disabled, button.textContent];
  return { state, elements, context, shown };
}

const searching = (note, fraction) => ({ solver: 'neural-searching', note, fraction });

test('Move now on a neural search says the move is coming until it comes', () => {
  const { state, elements, context, shown } = page(searching('Neural search · up to 512 simulations on wasm', 2 / 512));
  context.renderThinkingBar();
  assert.deepEqual(shown(), [false, false, 'Move now']);
  context.moveNow();
  assert.equal(state.moveNowRequested, true, 'the search hears it at its next simulation');
  assert.deepEqual(shown(), [false, true, 'Moving…']);
  assert.equal(elements.board.focused, 1, 'the focus the disabled button held goes on to the board');
  // The search runs on to its minimum and says so; the button stays as it is.
  state.liveSearch = searching('Neural search · moving after 16 simulations, the minimum, on wasm', 3 / 16);
  context.renderThinkingBar();
  assert.equal(elements.thinkingBar.value, 3 / 16);
  assert.deepEqual(shown(), [false, true, 'Moving…']);
  // The next request starts unpressed (requestAiMove), and so does the button.
  state.moveNowRequested = false;
  state.liveSearch = searching('Neural search · up to 512 simulations on wasm');
  context.renderThinkingBar();
  assert.deepEqual(shown(), [false, false, 'Move now']);
});

test('Move now moves no focus the button did not hold', () => {
  const { elements, context, shown } = page(searching('Neural search · up to 16 simulations on wasm', 0.5),
    { focused: false });
  context.renderThinkingBar();
  context.moveNow();
  assert.deepEqual(shown(), [false, true, 'Moving…']);
  assert.equal(elements.board.focused, 0);
});
