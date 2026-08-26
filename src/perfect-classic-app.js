import './app.js';
import {
  YELLOW,
  supportsPerfectChaosConfig,
  supportsPerfectClassicConfig,
  supportsPerfectConfig,
} from './engine.js';
import {
  findPerfectClassicPolicy,
  loadPerfectClassicManifest,
  perfectClassicRole,
} from './perfect-classic-policy.js';
import {
  findPerfectChaosCompletePolicy,
  loadPerfectChaosCompleteManifest,
  perfectChaosCompleteRole,
} from './perfect-chaos-complete.js';

const SETTINGS_KEY = 'connect4-chaos.settings.v1';
const rowsInput = document.querySelector('#rowsInput');
const columnsInput = document.querySelector('#colsInput');
const connectInput = document.querySelector('#connectInput');
const chaosInput = document.querySelector('#chaosInput');
const opponentInput = document.querySelector('#opponentInput');
const startingPlayerInput = document.querySelector('#startingPlayerInput');
const perfectOption = document.querySelector('#perfectOpponentOption');
const opponentHint = document.querySelector('#opponentHint');

const PERFECT_HINT = 'Game-theoretically optimal play using a verified policy and exact endgame solver.';
let restorePerfectAfterRuleChange = false;
let applying = false;
let manifest = null;
let manifestLoaded = false;
let manifestError = null;
let chaosManifest = null;
let chaosManifestLoaded = false;
let chaosManifestError = null;

function selectedRules() {
  return {
    rows: Number.parseInt(rowsInput?.value ?? '', 10),
    columns: Number.parseInt(columnsInput?.value ?? '', 10),
    connect: Number.parseInt(connectInput?.value ?? '', 10),
    chaosMode: chaosInput?.checked === true,
    startingPlayer: Number.parseInt(startingPlayerInput?.value ?? '', 10),
  };
}

function standardPerfectAvailable(rules) {
  return rules.rows === 6
    && rules.columns === 7
    && rules.connect === 4
    && rules.chaosMode === false;
}

function selectedPolicyEntry(rules) {
  if (!manifestLoaded || !manifest || standardPerfectAvailable(rules)) return null;
  const role = perfectClassicRole(rules.startingPlayer, YELLOW);
  if (role === null) return null;
  return findPerfectClassicPolicy(
    manifest,
    rules.rows,
    rules.columns,
    rules.connect,
    role,
  );
}

function selectedChaosPolicyEntry(rules) {
  if (!chaosManifestLoaded || !chaosManifest) return null;
  if (!supportsPerfectChaosConfig(
    rules.rows,
    rules.columns,
    rules.connect,
    rules.chaosMode,
  )) return null;
  const role = perfectChaosCompleteRole(rules.startingPlayer, YELLOW);
  if (role === null) return null;
  return findPerfectChaosCompletePolicy(
    chaosManifest,
    rules.rows,
    rules.columns,
    rules.connect,
    role,
  );
}

function selectedRulesSupportPerfect() {
  const rules = selectedRules();
  if (rules.chaosMode) {
    return selectedChaosPolicyEntry(rules) !== null;
  }
  if (!supportsPerfectClassicConfig(
    rules.rows,
    rules.columns,
    rules.connect,
    rules.chaosMode,
  )) return false;
  return standardPerfectAvailable(rules) || selectedPolicyEntry(rules) !== null;
}

function savedRoundRequestedPerfect() {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? 'null');
    if (saved?.opponent !== 'perfect') return false;
    const rules = {
      rows: Number(saved.rows),
      columns: Number(saved.cols),
      connect: Number(saved.connect),
      chaosMode: Boolean(saved.chaosMode),
      startingPlayer: Number(saved.startingPlayer),
    };
    if (!supportsPerfectConfig(
      rules.rows,
      rules.columns,
      rules.connect,
      rules.chaosMode,
    )) return false;
    if (rules.chaosMode) return selectedChaosPolicyEntry(rules) !== null;
    if (standardPerfectAvailable(rules)) return true;
    if (!manifestLoaded || !manifest) return false;
    const role = perfectClassicRole(rules.startingPlayer, YELLOW);
    return role !== null && findPerfectClassicPolicy(
      manifest,
      rules.rows,
      rules.columns,
      rules.connect,
      role,
    ) !== null;
  } catch {
    return false;
  }
}

function availabilityTitle(available, rules) {
  if (available) {
    if (rules.chaosMode) {
      return 'Uses a completely solved Chaos Mode certificate — no search, no handoff.';
    }
    return standardPerfectAvailable(rules)
      ? 'Uses the verified standard 6×7 strategy and exact endgame solver.'
      : 'Uses a verified optimal policy with an exact endgame handoff.';
  }
  if (!supportsPerfectConfig(
    rules.rows,
    rules.columns,
    rules.connect,
    rules.chaosMode,
  )) {
    return rules.chaosMode
      ? 'Perfect AI supports Chaos Mode only on boards with a committed complete solution.'
      : 'Perfect AI supports non-Chaos Connect Four boards from 4×4 through 7×7.';
  }
  if (rules.chaosMode) {
    if (!chaosManifestLoaded) return 'Loading the verified Chaos certificate catalog…';
    if (chaosManifestError) return 'The verified Chaos certificate catalog could not be loaded.';
    return 'A verified Chaos certificate for this board and starting role is not installed yet.';
  }
  if (!manifestLoaded) return 'Loading the verified Perfect policy catalog…';
  if (manifestError) return 'The verified Perfect policy catalog could not be loaded.';
  return 'A verified policy for this board and starting role is not installed yet.';
}

// Assigning select.value fires no change event, so app.js would never refresh
// the difficulty hint or the starting-player label for the replacement
// opponent, leaving a Perfect description beside a downgraded selection.
function setOpponent(value) {
  if (opponentInput.value === value) return;
  opponentInput.value = value;
  opponentInput.dispatchEvent(new Event('change', { bubbles: true }));
}

function applyPerfectAvailability({ restoreSaved = false } = {}) {
  if (applying || !perfectOption || !opponentInput) return;
  applying = true;
  try {
    const rules = selectedRules();
    const available = selectedRulesSupportPerfect();
    if (perfectOption.disabled === available) perfectOption.disabled = !available;
    const title = availabilityTitle(available, rules);
    if (perfectOption.title !== title) perfectOption.title = title;

    if (available && (restorePerfectAfterRuleChange || (restoreSaved && savedRoundRequestedPerfect()))) {
      setOpponent('perfect');
    } else if (!available && opponentInput.value === 'perfect') {
      setOpponent('brutal');
    }
    restorePerfectAfterRuleChange = false;

    if (opponentInput.value === 'perfect' && opponentHint) {
      opponentHint.textContent = PERFECT_HINT;
    }
  } finally {
    applying = false;
  }
}

const ruleInputs = [
  rowsInput,
  columnsInput,
  connectInput,
  chaosInput,
  startingPlayerInput,
].filter(Boolean);
for (const input of ruleInputs) {
  input.addEventListener('input', () => queueMicrotask(applyPerfectAvailability));
  input.addEventListener('change', () => queueMicrotask(applyPerfectAvailability));
}

// Remember a selected Perfect opponent before app.js processes a rule change;
// restore it only when the replacement rules have a committed policy.
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
// catalog-backed availability immediately after those attribute updates.
if (perfectOption) {
  const observer = new MutationObserver(() => queueMicrotask(applyPerfectAvailability));
  observer.observe(perfectOption, { attributes: true, attributeFilter: ['disabled', 'title'] });
}

applyPerfectAvailability({ restoreSaved: true });
loadPerfectClassicManifest().then((loaded) => {
  manifest = loaded;
  manifestLoaded = true;
  manifestError = null;
  applyPerfectAvailability({ restoreSaved: true });
}).catch((error) => {
  manifest = null;
  manifestLoaded = true;
  manifestError = error;
  applyPerfectAvailability();
});
loadPerfectChaosCompleteManifest().then((loaded) => {
  chaosManifest = loaded;
  chaosManifestLoaded = true;
  chaosManifestError = null;
  applyPerfectAvailability({ restoreSaved: true });
}).catch((error) => {
  chaosManifest = null;
  chaosManifestLoaded = true;
  chaosManifestError = error;
  applyPerfectAvailability();
});
