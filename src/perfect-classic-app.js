import './app.js';
import { supportsPerfectClassicConfig } from './engine.js';

const SETTINGS_KEY = 'connect4-chaos.settings.v1';
const rowsInput = document.querySelector('#rowsInput');
const columnsInput = document.querySelector('#colsInput');
const connectInput = document.querySelector('#connectInput');
const chaosInput = document.querySelector('#chaosInput');
const opponentInput = document.querySelector('#opponentInput');
const perfectOption = document.querySelector('#perfectOpponentOption');
const opponentHint = document.querySelector('#opponentHint');

const PERFECT_HINT = 'Game-theoretically optimal play on classic Connect Four boards from 4×4 through 7×7.';
let restorePerfectAfterRuleChange = false;
let applying = false;

function selectedRulesSupportPerfect() {
  return supportsPerfectClassicConfig(
    Number.parseInt(rowsInput?.value ?? '', 10),
    Number.parseInt(columnsInput?.value ?? '', 10),
    Number.parseInt(connectInput?.value ?? '', 10),
    chaosInput?.checked === true,
  );
}

function savedRoundRequestedPerfect() {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? 'null');
    return saved?.opponent === 'perfect'
      && supportsPerfectClassicConfig(
        Number(saved.rows),
        Number(saved.cols),
        Number(saved.connect),
        Boolean(saved.chaosMode),
      );
  } catch {
    return false;
  }
}

function applyPerfectAvailability({ restoreSaved = false } = {}) {
  if (applying || !perfectOption || !opponentInput) return;
  applying = true;
  try {
    const available = selectedRulesSupportPerfect();
    if (perfectOption.disabled === available) perfectOption.disabled = !available;
    const title = available
      ? 'Uses a verified optimal policy with an exact endgame handoff.'
      : 'Perfect AI supports non-Chaos Connect Four boards from 4×4 through 7×7.';
    if (perfectOption.title !== title) perfectOption.title = title;

    if (available && (restorePerfectAfterRuleChange || (restoreSaved && savedRoundRequestedPerfect()))) {
      opponentInput.value = 'perfect';
    } else if (!available && opponentInput.value === 'perfect') {
      opponentInput.value = 'brutal';
    }
    restorePerfectAfterRuleChange = false;

    if (opponentInput.value === 'perfect' && opponentHint) {
      opponentHint.textContent = PERFECT_HINT;
    }
  } finally {
    applying = false;
  }
}

const ruleInputs = [rowsInput, columnsInput, connectInput, chaosInput].filter(Boolean);
for (const input of ruleInputs) {
  input.addEventListener('input', () => queueMicrotask(applyPerfectAvailability));
  input.addEventListener('change', () => queueMicrotask(applyPerfectAvailability));
}

// Remember a selected Perfect opponent before app.js processes a rule change;
// restore it when the new rules remain inside the verified policy matrix.
document.addEventListener('input', (event) => {
  if (ruleInputs.includes(event.target) && opponentInput?.value === 'perfect') {
    restorePerfectAfterRuleChange = true;
  }
}, true);
document.addEventListener('change', (event) => {
  if (ruleInputs.includes(event.target) && opponentInput?.value === 'perfect') {
    restorePerfectAfterRuleChange = true;
  }
}, true);

opponentInput?.addEventListener('change', () => queueMicrotask(applyPerfectAvailability));

// app.js refreshes the option while rounds and forms are initialised. Reapply
// the broader verified matrix immediately after those attribute updates.
if (perfectOption) {
  const observer = new MutationObserver(() => queueMicrotask(applyPerfectAvailability));
  observer.observe(perfectOption, { attributes: true, attributeFilter: ['disabled', 'title'] });
}

applyPerfectAvailability({ restoreSaved: true });
