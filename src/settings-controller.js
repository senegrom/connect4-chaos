import { YELLOW, normalizeConfig, supportsPerfectConfig } from './engine.js';
import { findPerfectClassicPolicy, loadPerfectClassicManifest, perfectClassicRole } from './perfect-classic-policy.js';
import { findPerfectChaosCompletePolicy, loadPerfectChaosCompleteManifest, perfectChaosCompleteRole } from './perfect-chaos-complete.js';
import { waitFor } from './async-control.js';
import { DOWNLOAD_BYTES } from './neural-runtime.js';
import { formatBytes } from './download-gate.js';

export const DIFFICULTY_HINTS = Object.freeze({
  human: 'Two people share this device.',
  easy: 'Quick and forgiving, with basic wins and blocks.',
  medium: 'Responsive tactical play with solid planning.',
  hard: 'Plans further ahead and may think a little longer.',
  brutal: 'The deepest general search, with a certified Chaos policy and exact late-game solving.',
  perfect: 'Game-theoretically optimal play using a verified policy and exact endgame solver.',
  neural: `A trained network with a look-ahead search, run on your device after an approximately ${formatBytes(DOWNLOAD_BYTES.model + DOWNLOAD_BYTES.runtime)} download, normally cached by your browser.`,
});

/** One source of truth for geometry, starting role, catalog state and copy. */
export function perfectCapability(rules, catalogs = {}) {
  const unavailable = (message, status = 'unavailable') => ({ available: false, status, message });
  if (!supportsPerfectConfig(rules.rows, rules.cols, rules.connect, rules.chaosMode)) {
    return unavailable('Perfect AI requires a board with a committed complete solution.');
  }
  if (!rules.chaosMode && rules.rows === 6 && rules.cols === 7 && rules.connect === 4) {
    return { available: true, status: 'ready', message: 'Uses the verified standard 6×7 strategy and exact endgame solver.' };
  }
  const catalog = rules.chaosMode ? catalogs.chaos : catalogs.classic;
  if (!catalog || catalog.status === 'loading') return unavailable('Loading the verified policy catalog…', 'loading');
  if (catalog.status === 'error') return unavailable('The verified policy catalog could not be loaded. Reopen settings or change a rule to retry.', 'error');
  const role = (rules.chaosMode ? perfectChaosCompleteRole : perfectClassicRole)(rules.startingPlayer, YELLOW);
  const entry = role === null ? null : (rules.chaosMode ? findPerfectChaosCompletePolicy : findPerfectClassicPolicy)(
    catalog.manifest, rules.rows, rules.cols, rules.connect, role,
  );
  return entry
    ? { available: true, status: 'ready', message: rules.chaosMode
      ? 'Uses a completely solved Chaos Mode certificate — no search, no handoff.'
      : 'Uses a verified optimal policy with an exact endgame handoff.' }
    : unavailable('A verified policy for this board and starting role is not installed yet.');
}

/** This controller alone writes the opponent selection and availability. */
export function createSettingsController(elements, loaders = {}) {
  const catalogs = {};
  const catalogLoaders = {
    classic: loaders.classic ?? loadPerfectClassicManifest,
    chaos: loaders.chaos ?? loadPerfectChaosCompleteManifest,
  };
  const submit = elements.settingsForm.querySelector('[type="submit"]');
  const rules = () => ({
    rows: Number.parseInt(elements.rowsInput.value, 10),
    cols: Number.parseInt(elements.colsInput.value, 10),
    connect: Number.parseInt(elements.connectInput.value, 10),
    chaosMode: elements.chaosInput.checked,
    startingPlayer: Number.parseInt(elements.startingPlayerInput.value, 10),
  });
  const numericFields = [
    [elements.rowsInput, 'Rows'],
    [elements.colsInput, 'Columns'],
    [elements.connectInput, 'Connect'],
  ];
  function validateNumericFields({ report = false } = {}) {
    let firstInvalid = null;
    for (const [input, label] of numericFields) {
      const value = input.valueAsNumber;
      const minimum = Number(input.min);
      const maximum = Number(input.max);
      let message = '';
      if (input.value.trim() === '' || !Number.isFinite(value)) message = `${label} is required.`;
      else if (!Number.isInteger(value)) message = `${label} must be a whole number.`;
      else if (value < minimum || value > maximum) message = `${label} must be between ${minimum} and ${maximum}.`;
      input.setCustomValidity(message);
      if (message) input.setAttribute('aria-invalid', 'true');
      else input.removeAttribute('aria-invalid');
      const note = input.parentElement?.querySelector('.field-error');
      if (note) note.textContent = message;
      if (message && firstInvalid === null) firstInvalid = input;
    }
    if (report && firstInvalid) firstInvalid.reportValidity();
    return firstInvalid === null;
  }
  function refresh() {
    const current = rules();
    const rows = Math.max(4, Math.min(10, current.rows || 6));
    const cols = Math.max(4, Math.min(10, current.cols || 7));
    const maximum = Math.min(6, Math.max(rows, cols));
    elements.connectInput.max = String(maximum);
    // Geometry changes must not silently rewrite a rule the user chose.
    // Validation explains the conflict until they explicitly change Connect.
    const fieldsValid = validateNumericFields();
    const capability = perfectCapability(rules(), catalogs);
    const opponent = elements.opponentInput.value;
    // Catalog availability must never rewrite the user's opponent choice.
    // A transient error can disable applying Perfect, but the editor remains
    // truthful about what the user selected and what the active round uses.
    elements.perfectOpponentOption.disabled = !capability.available && opponent !== 'perfect';
    elements.perfectOpponentOption.title = capability.message;
    if (submit) submit.disabled = !fieldsValid
      || (opponent === 'perfect' && !capability.available);
    elements.yellowStarterOption.textContent = opponent === 'human' ? 'Yellow' : 'AI (Yellow)';
    elements.opponentHint.textContent = opponent === 'perfect' && !capability.available
      ? capability.message : DIFFICULTY_HINTS[opponent] ?? DIFFICULTY_HINTS.medium;
    return capability;
  }
  async function load(name, { force = false } = {}) {
    const pending = { status: 'loading' };
    catalogs[name] = pending;
    refresh();
    try {
      const manifest = await waitFor(catalogLoaders[name](undefined, { force }), {
        timeoutMs: 10_000, label: 'Policy catalog',
      });
      if (catalogs[name] === pending) catalogs[name] = { status: 'ready', manifest };
    } catch {
      if (catalogs[name] === pending) catalogs[name] = { status: 'error' };
    }
    refresh();
  }
  function retryCatalogs() {
    // Only user interaction retries failures: rendering an error must not
    // create a request loop. Pending loads are deduplicated, and force
    // bypasses a cached request that outlived the controller's timeout.
    return Promise.all(Object.keys(catalogLoaders)
      .filter((name) => catalogs[name]?.status === 'error')
      .map((name) => load(name, { force: true })));
  }
  const refreshAfterInteraction = () => {
    void retryCatalogs();
    refresh();
  };
  for (const input of [elements.rowsInput, elements.colsInput, elements.connectInput,
    elements.chaosInput, elements.startingPlayerInput, elements.opponentInput]) {
    input.addEventListener('input', refreshAfterInteraction);
    input.addEventListener('change', refreshAfterInteraction);
  }
  // Opening the editor or focusing a field also recovers a disabled Perfect
  // option, without requiring a different rule or a page reload.
  elements.settingsToggle?.addEventListener('click', retryCatalogs);
  elements.settingsForm.addEventListener('focusin', retryCatalogs);
  const ready = Promise.all(Object.keys(catalogLoaders).map((name) => load(name)));
  return {
    ready,
    refresh,
    retryCatalogs,
    acceptCatalog(name, manifest) {
      if (name !== 'classic' && name !== 'chaos') throw new Error('Unknown policy catalog');
      catalogs[name] = { status: 'ready', manifest };
      refresh();
    },
    canApply: ({ report = false } = {}) => {
      const fieldsValid = validateNumericFields({ report });
      if (!fieldsValid) return false;
      const perfectValid = elements.opponentInput.value !== 'perfect' || perfectCapability(rules(), catalogs).available;
      if (!perfectValid && report) elements.opponentInput.focus();
      return perfectValid;
    },
    read: () => normalizeConfig({ ...rules(), opponent: elements.opponentInput.value }),
    populate(config) {
      elements.rowsInput.value = String(config.rows);
      elements.colsInput.value = String(config.cols);
      elements.connectInput.value = String(config.connect);
      elements.opponentInput.value = config.opponent;
      elements.startingPlayerInput.value = String(config.startingPlayer);
      elements.chaosInput.checked = config.chaosMode;
      refresh();
    },
  };
}
