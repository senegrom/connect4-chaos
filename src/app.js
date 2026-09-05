import { chooseMove, evaluateBoard } from './ai.js';
import {
  ACTION_DROP,
  ACTION_FLIP,
  ACTION_ROTATE_CCW,
  ACTION_ROTATE_CW,
  EMPTY,
  RED,
  YELLOW,
  applyAction,
  boardDimensions,
  canDrop,
  cloneBoard,
  createBoard,
  getDropRow,
  legalActions,
  normalizeConfig,
  otherPlayer,
  positionKey,
  resolveActionOutcome,
  supportsPerfectConfig,
} from './engine.js';

const SETTINGS_KEY = 'connect4-chaos.settings.v1';
const SCORES_KEY = 'connect4-chaos.scores.v1';
const ROUND_KEY = 'connect4-chaos.round.v1';
const DIFFICULTY_LABELS = Object.freeze({
  human: 'Human',
  easy: 'Easy AI',
  medium: 'Medium AI',
  hard: 'Hard AI',
  brutal: 'Brutal AI',
  perfect: 'Perfect AI',
  neural: 'Neural AI',
});
const DIFFICULTY_HINTS = Object.freeze({
  human: 'Two people share this device.',
  easy: 'Quick and forgiving, with basic wins and blocks.',
  medium: 'Responsive tactical play with solid planning.',
  hard: 'Plans further ahead and may think a little longer.',
  brutal: 'The deepest general search, with a certified Chaos policy and exact late-game solving.',
  perfect: 'Game-theoretically optimal play where a verified certificate exists.',
  neural: 'A trained network with a look-ahead search, run on your device after a one-time 73 MB download.',
});
const COLUMN_CLASSES = Array.from({ length: 7 }, (_, index) => `cols-${index + 4}`);
const ANIMATION_CLASSES = [
  'anim-flip-out',
  'anim-flip-in',
  'anim-cw-out',
  'anim-cw-in',
  'anim-ccw-out',
  'anim-ccw-in',
];
const TRANSFORM_ANIMATIONS = Object.freeze({
  [ACTION_FLIP]: Object.freeze({
    outClass: 'anim-flip-out',
    inClass: 'anim-flip-in',
    outMs: 320,
    inMs: 420,
  }),
  [ACTION_ROTATE_CW]: Object.freeze({
    outClass: 'anim-cw-out',
    inClass: 'anim-cw-in',
    outMs: 280,
    inMs: 360,
  }),
  [ACTION_ROTATE_CCW]: Object.freeze({
    outClass: 'anim-ccw-out',
    inClass: 'anim-ccw-in',
    outMs: 280,
    inMs: 360,
  }),
});

const elements = {
  setupPanel: document.querySelector('#setupPanel'),
  setupTitle: document.querySelector('#setupTitle'),
  settingsBody: document.querySelector('#settingsBody'),
  settingsToggle: document.querySelector('#settingsToggle'),
  activeRulesSummary: document.querySelector('#activeRulesSummary'),
  settingsForm: document.querySelector('#settingsForm'),
  rowsInput: document.querySelector('#rowsInput'),
  colsInput: document.querySelector('#colsInput'),
  connectInput: document.querySelector('#connectInput'),
  opponentInput: document.querySelector('#opponentInput'),
  perfectOpponentOption: document.querySelector('#perfectOpponentOption'),
  startingPlayerInput: document.querySelector('#startingPlayerInput'),
  yellowStarterOption: document.querySelector('#yellowStarterOption'),
  opponentHint: document.querySelector('#opponentHint'),
  chaosInput: document.querySelector('#chaosInput'),
  redScore: document.querySelector('#redScore'),
  yellowScore: document.querySelector('#yellowScore'),
  drawScore: document.querySelector('#drawScore'),
  yellowScoreLabel: document.querySelector('#yellowScoreLabel'),
  resetScoreButton: document.querySelector('#resetScoreButton'),
  matchGrid: document.querySelector('.match-grid'),
  evaluationPanel: document.querySelector('#evaluationPanel'),
  evaluationLabel: document.querySelector('#evaluationLabel'),
  evaluationDescription: document.querySelector('#evaluationDescription'),
  evaluationBalance: document.querySelector('#evaluationBalance'),
  exactResult: document.querySelector('#exactResult'),
  exactBadge: document.querySelector('#exactBadge'),
  exactResultText: document.querySelector('#exactResultText'),
  searchInfo: document.querySelector('#searchInfo'),
  aiRecovery: document.querySelector('#aiRecovery'),
  retryAiButton: document.querySelector('#retryAiButton'),
  switchBrutalButton: document.querySelector('#switchBrutalButton'),
  undoAiButton: document.querySelector('#undoAiButton'),
  statusDisc: document.querySelector('#statusDisc'),
  statusText: document.querySelector('#statusText'),
  aiErrorText: document.querySelector('#aiErrorText'),
  thinkingBarRow: document.querySelector('#thinkingBarRow'),
  thinkingBar: document.querySelector('#thinkingBar'),
  moveNowButton: document.querySelector('#moveNowButton'),
  downloadDialog: document.querySelector('#downloadDialog'),
  moveInfo: document.querySelector('#moveInfo'),
  gamePanel: document.querySelector('#gamePanel'),
  thinkingIndicator: document.querySelector('#thinkingIndicator'),
  thinkingProgress: document.querySelector('#thinkingProgress'),
  transformToolbar: document.querySelector('#transformToolbar'),
  restartButton: document.querySelector('#restartButton'),
  undoButton: document.querySelector('#undoButton'),
  flipButton: document.querySelector('#flipButton'),
  rotateCcwButton: document.querySelector('#rotateCcwButton'),
  rotateCwButton: document.querySelector('#rotateCwButton'),
  chaosActions: [...document.querySelectorAll('.chaos-action')],
  chaosKeyboardHelp: document.querySelector('#chaosKeyboardHelp'),
  boardFrame: document.querySelector('#boardFrame'),
  columnControls: document.querySelector('#columnControls'),
  ghostDisc: document.querySelector('#ghostDisc'),
  board: document.querySelector('#gameBoard'),
  touchHelp: document.querySelector('#touchHelp'),
  selectedColumnStatus: document.querySelector('#selectedColumnStatus'),
  resultDialog: document.querySelector('#resultDialog'),
  dialogDisc: document.querySelector('#dialogDisc'),
  dialogTitle: document.querySelector('#dialogTitle'),
  dialogMessage: document.querySelector('#dialogMessage'),
  reviewBoardButton: document.querySelector('#reviewBoardButton'),
  changeRulesButton: document.querySelector('#changeRulesButton'),
  playAgainButton: document.querySelector('#playAgainButton'),
  rulesButton: document.querySelector('#rulesButton'),
  rulesDialog: document.querySelector('#rulesDialog'),
  closeRulesButton: document.querySelector('#closeRulesButton'),
  rulesDoneButton: document.querySelector('#rulesDoneButton'),
};

const reducedMotion = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
const coarsePointer = globalThis.matchMedia?.('(pointer: coarse)') ?? { matches: false };
const numberFormatter = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });
const hadSavedSettings = storageHasValue(SETTINGS_KEY);
// On a phone the full setup form would push the board off the first screen,
// so small screens start compact even on a first visit.
const narrowViewport = globalThis.matchMedia?.('(max-width: 39rem)').matches ?? false;

const state = {
  config: normalizeConfig(loadJson(SETTINGS_KEY, {})),
  board: [],
  currentPlayer: RED,
  status: 'playing',
  winner: EMPTY,
  winningCells: [],
  simultaneousWin: false,
  drawReason: null,
  lastMove: null,
  lastMover: null,
  moveCount: 0,
  selectedColumn: 0,
  repetitionCounts: new Map(),
  scores: normalizeScores(loadJson(SCORES_KEY, {})),
  history: [],
  busy: false,
  aiThinking: false,
  aiWorker: null,
  aiRequest: null,
  aiRequestId: 0,
  version: 0,
  dropAnimation: null,
  lastSearch: null,
  liveSearch: null,
  aiError: null,
  moveNowRequested: false,
  gameFirstLayout: hadSavedSettings || narrowViewport,
  touchHintDismissed: false,
};

function storageHasValue(key) {
  try {
    return localStorage.getItem(key) !== null;
  } catch {
    return false;
  }
}

function loadJson(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return (value ? JSON.parse(value) : null) ?? fallback;
  } catch {
    return fallback;
  }
}

function saveJson(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // The game remains fully usable when storage is unavailable.
  }
}

function normalizeScores(scores) {
  return {
    [RED]: Math.max(0, Number.parseInt(scores?.[RED] ?? scores?.red ?? 0, 10) || 0),
    [YELLOW]: Math.max(0, Number.parseInt(scores?.[YELLOW] ?? scores?.yellow ?? 0, 10) || 0),
    draw: Math.max(0, Number.parseInt(scores?.draw ?? 0, 10) || 0),
  };
}

function isAiGame() {
  return state.config.opponent !== 'human';
}

function playerName(player) {
  if (player === RED) return 'Red';
  return isAiGame() ? 'AI (Yellow)' : 'Yellow';
}

function playerClass(player) {
  return player === RED ? 'red' : 'yellow';
}

function readSettingsForm() {
  return normalizeConfig({
    rows: elements.rowsInput.value,
    cols: elements.colsInput.value,
    connect: elements.connectInput.value,
    opponent: elements.opponentInput.value,
    startingPlayer: elements.startingPlayerInput.value,
    chaosMode: elements.chaosInput.checked,
  });
}

function populateSettingsForm(config) {
  elements.rowsInput.value = String(config.rows);
  elements.colsInput.value = String(config.cols);
  elements.connectInput.value = String(config.connect);
  elements.opponentInput.value = config.opponent;
  elements.startingPlayerInput.value = String(config.startingPlayer);
  elements.chaosInput.checked = config.chaosMode;
  updateConnectLimit();
  updatePerfectAvailability();
}

function updateConnectLimit() {
  const rows = Math.max(4, Math.min(10, Number.parseInt(elements.rowsInput.value, 10) || 6));
  const cols = Math.max(4, Math.min(10, Number.parseInt(elements.colsInput.value, 10) || 7));
  const maximum = Math.min(6, Math.max(rows, cols));
  elements.connectInput.max = String(maximum);
  if ((Number.parseInt(elements.connectInput.value, 10) || 4) > maximum) {
    elements.connectInput.value = String(maximum);
  }
}

function updateOpponentLabels() {
  const opponent = elements.opponentInput.value;
  const aiSelected = opponent !== 'human';
  elements.yellowStarterOption.textContent = aiSelected ? 'AI (Yellow)' : 'Yellow';
  elements.opponentHint.textContent = DIFFICULTY_HINTS[opponent] ?? DIFFICULTY_HINTS.medium;
}

// A conservative first pass: perfect-classic-app.js refines this once the
// verified catalogs have loaded and it knows which boards are actually
// installed.
function formSupportsPerfect() {
  return supportsPerfectConfig(
    Number.parseInt(elements.rowsInput.value, 10),
    Number.parseInt(elements.colsInput.value, 10),
    Number.parseInt(elements.connectInput.value, 10),
    elements.chaosInput.checked,
  );
}

function updatePerfectAvailability() {
  const available = formSupportsPerfect();
  elements.perfectOpponentOption.disabled = !available;
  elements.perfectOpponentOption.title = available
    ? 'Uses a machine-verified strategy and exact endgame solver.'
    : 'Perfect AI requires a board with a committed exact solution.';
  if (!available && elements.opponentInput.value === 'perfect') {
    elements.opponentInput.value = 'brutal';
  }
  updateOpponentLabels();
}

function activeRulesText(config = state.config) {
  const opponent = DIFFICULTY_LABELS[config.opponent] ?? DIFFICULTY_LABELS.medium;
  const chaos = config.chaosMode ? ' · Chaos' : '';
  return `${config.rows}×${config.cols} · Connect ${config.connect} · ${opponent}${chaos}`;
}

function renderActiveRulesSummary() {
  elements.activeRulesSummary.textContent = activeRulesText();
}

function renderLayout() {
  document.body.classList.toggle('game-first', state.gameFirstLayout);
  elements.setupPanel.classList.toggle('is-collapsed', elements.settingsBody.hidden);
}

function setSettingsExpanded(expanded, focusToggle = false) {
  elements.settingsBody.hidden = !expanded;
  elements.settingsToggle.setAttribute('aria-expanded', String(expanded));
  elements.settingsToggle.textContent = expanded ? 'Hide settings' : 'Change rules';
  elements.setupTitle.textContent = expanded ? 'Choose the rules' : 'Current rules';
  renderLayout();
  if (focusToggle) elements.settingsToggle.focus();
}

function makeSnapshot() {
  return {
    board: cloneBoard(state.board),
    currentPlayer: state.currentPlayer,
    status: state.status,
    winner: state.winner,
    winningCells: state.winningCells.map((cell) => [...cell]),
    simultaneousWin: state.simultaneousWin,
    drawReason: state.drawReason,
    lastMove: state.lastMove ? { ...state.lastMove } : null,
    lastMover: state.lastMover,
    moveCount: state.moveCount,
    selectedColumn: state.selectedColumn,
    repetitionCounts: [...state.repetitionCounts.entries()],
    scores: { ...state.scores },
    lastSearch: state.lastSearch ? { ...state.lastSearch } : null,
  };
}

function restoreSnapshot(snapshot) {
  state.board = cloneBoard(snapshot.board);
  state.currentPlayer = snapshot.currentPlayer;
  state.status = snapshot.status;
  state.winner = snapshot.winner;
  state.winningCells = snapshot.winningCells.map((cell) => [...cell]);
  state.simultaneousWin = snapshot.simultaneousWin;
  state.drawReason = snapshot.drawReason;
  state.lastMove = snapshot.lastMove ? { ...snapshot.lastMove } : null;
  state.lastMover = snapshot.lastMover;
  state.moveCount = snapshot.moveCount;
  state.selectedColumn = snapshot.selectedColumn;
  state.repetitionCounts = new Map(snapshot.repetitionCounts);
  state.scores = { ...snapshot.scores };
  state.lastSearch = snapshot.lastSearch ? { ...snapshot.lastSearch } : null;
  state.liveSearch = null;
  state.dropAnimation = null;
  state.aiError = null;
}

function pushSnapshot() {
  state.history.push(makeSnapshot());
}

// --- the round in progress, kept across a crash or reload ----------------------

/**
 * Stores the round so a tab that crashes or reloads comes back to the same
 * board rather than an empty one. A finished round has nothing to restore.
 */
function saveRound() {
  if (state.status !== 'playing' || state.history.length < 2) {
    clearRound();
    return;
  }
  saveJson(ROUND_KEY, {
    version: 1,
    config: state.config,
    touchHintDismissed: state.touchHintDismissed,
    history: state.history,
  });
}

function clearRound() {
  try {
    localStorage.removeItem(ROUND_KEY);
  } catch {
    // Nothing to clear when storage is unavailable.
  }
}

function sameConfig(a, b) {
  return a.rows === b.rows
    && a.cols === b.cols
    && a.connect === b.connect
    && a.opponent === b.opponent
    && a.startingPlayer === b.startingPlayer
    && a.chaosMode === b.chaosMode;
}

function validSnapshot(snapshot, config) {
  if (!snapshot || !Array.isArray(snapshot.board) || !Array.isArray(snapshot.board[0])) return false;
  // A Chaos rotation transposes the board, so either orientation is valid there.
  const rows = snapshot.board.length;
  const cols = snapshot.board[0].length;
  const upright = rows === config.rows && cols === config.cols;
  const transposed = config.chaosMode && rows === config.cols && cols === config.rows;
  if (!upright && !transposed) return false;
  const cells = snapshot.board.every((row) => Array.isArray(row)
    && row.length === cols
    && row.every((cell) => cell === EMPTY || cell === RED || cell === YELLOW));
  if (!cells) return false;
  if (snapshot.currentPlayer !== RED && snapshot.currentPlayer !== YELLOW) return false;
  return Array.isArray(snapshot.winningCells)
    && Array.isArray(snapshot.repetitionCounts)
    && Boolean(snapshot.scores) && typeof snapshot.scores === 'object';
}

/** Resumes a saved round when it matches the current rules; true when it did. */
function restoreSavedRound(saved) {
  if (!saved || saved.version !== 1 || !Array.isArray(saved.history) || saved.history.length < 2) return false;
  const config = normalizeConfig(saved.config ?? {});
  const last = saved.history[saved.history.length - 1];
  if (!sameConfig(config, state.config) || !validSnapshot(last, config) || last.status !== 'playing') {
    return false;
  }
  if (!saved.history.every((snapshot) => validSnapshot(snapshot, config))) return false;
  cancelAiSearch();
  state.version += 1;
  state.history = saved.history;
  restoreSnapshot(last);
  state.touchHintDismissed = Boolean(saved.touchHintDismissed);
  state.busy = false;
  state.aiThinking = false;
  state.aiError = null;
  // startRound cleared the stored round before the restore; keep it for
  // the next reload, until a move replaces it.
  saveRound();
  renderAll();
  if (isAiGame() && state.currentPlayer === YELLOW) requestAiMove();
  return true;
}

function currentRepetitionCount() {
  const key = positionKey(
    state.board,
    state.currentPlayer,
    state.config.connect,
    state.config.chaosMode,
  );
  return state.repetitionCounts.get(key) ?? 0;
}

function startRound(config = state.config, options = {}) {
  const {
    collapseSettings = state.gameFirstLayout,
    scrollToGame = false,
    activateGameFirst = false,
  } = options;

  cancelAiSearch();
  closeResultDialog();
  clearBoardAnimations();

  state.config = normalizeConfig(config);
  if (activateGameFirst) state.gameFirstLayout = true;
  populateSettingsForm(state.config);
  saveJson(SETTINGS_KEY, state.config);

  state.version += 1;
  state.board = createBoard(state.config.rows, state.config.cols);
  state.currentPlayer = state.config.startingPlayer;
  state.status = 'playing';
  state.winner = EMPTY;
  state.winningCells = [];
  state.simultaneousWin = false;
  state.drawReason = null;
  state.lastMove = null;
  state.lastMover = null;
  state.moveCount = 0;
  state.selectedColumn = Math.floor(state.config.cols / 2);
  state.repetitionCounts = new Map();
  state.busy = false;
  state.aiThinking = false;
  state.dropAnimation = null;
  state.lastSearch = null;
  state.liveSearch = null;
  state.aiError = null;
  state.history = [];

  const initialKey = positionKey(
    state.board,
    state.currentPlayer,
    state.config.connect,
    state.config.chaosMode,
  );
  state.repetitionCounts.set(initialKey, 1);
  pushSnapshot();
  clearRound();
  setSettingsExpanded(!collapseSettings);
  renderAll();

  if (scrollToGame) {
    requestAnimationFrame(() => {
      elements.gamePanel.scrollIntoView({
        behavior: reducedMotion ? 'auto' : 'smooth',
        block: 'start',
      });
      elements.board.focus({ preventScroll: true });
    });
  }

  if (isAiGame() && state.currentPlayer === YELLOW) requestAiMove();
}

function applySettingsAndStartRound() {
  startRound(readSettingsForm(), {
    collapseSettings: true,
    scrollToGame: true,
    activateGameFirst: true,
  });
}

function restartRound() {
  startRound(state.config, { collapseSettings: elements.settingsBody.hidden });
}

function canHumanAct() {
  return state.status === 'playing'
    && !state.busy
    && !state.aiThinking
    && (!isAiGame() || state.currentPlayer === RED);
}

function findUndoIndex() {
  if (state.history.length < 2) return -1;
  if (!isAiGame()) return state.history.length - 2;

  for (let index = state.history.length - 2; index >= 0; index -= 1) {
    const snapshot = state.history[index];
    if (snapshot.status === 'playing' && snapshot.currentPlayer === RED) return index;
  }
  return -1;
}

function undoTurn() {
  if (state.busy) return;
  const targetIndex = findUndoIndex();
  if (targetIndex < 0) return;

  cancelAiSearch();
  closeResultDialog();
  clearBoardAnimations();
  state.version += 1;
  state.history = state.history.slice(0, targetIndex + 1);
  restoreSnapshot(state.history[targetIndex]);
  state.busy = false;
  state.aiThinking = false;
  state.aiError = null;
  saveJson(SCORES_KEY, state.scores);
  saveRound();
  renderAll();
}

function resetScores() {
  state.scores = { [RED]: 0, [YELLOW]: 0, draw: 0 };
  for (const snapshot of state.history) snapshot.scores = { ...state.scores };
  saveJson(SCORES_KEY, state.scores);
  saveRound();
  renderScores();
}

function renderAll() {
  renderLayout();
  renderActiveRulesSummary();
  renderScores();
  renderStatus();
  renderActions();
  renderBoard();
  renderEvaluation();
  renderGuidance();
}

function renderScores() {
  elements.redScore.textContent = String(state.scores[RED]);
  elements.yellowScore.textContent = String(state.scores[YELLOW]);
  elements.drawScore.textContent = String(state.scores.draw);
  elements.yellowScoreLabel.textContent = isAiGame() ? 'AI' : 'Yellow';
}

function statusMessage() {
  if (state.status === 'won') return `${playerName(state.winner)} wins!`;
  if (state.status === 'draw') {
    return state.drawReason === 'repetition' ? 'Draw by repetition' : 'Draw — board full';
  }
  if (state.aiError) return 'AI unavailable';
  if (state.aiThinking) return 'AI is thinking…';
  if (state.busy && state.lastMover) return `${playerName(state.lastMover)} is moving…`;

  return `${playerName(state.currentPlayer)} to move`;
}

function renderStatus() {
  const displayPlayer = state.status === 'won' ? state.winner : state.currentPlayer;
  elements.statusDisc.className = `turn-disc ${playerClass(displayPlayer || RED)}`;
  elements.statusText.textContent = statusMessage();
  // The reason lives in the collapsed AI details; "AI unavailable" alone
  // reads as a dead end, so the reason is repeated right under the status.
  elements.aiErrorText.hidden = !state.aiError;
  elements.aiErrorText.textContent = state.aiError ?? '';
  elements.thinkingIndicator.classList.toggle('is-idle', !state.aiThinking);
  if (state.aiThinking && state.liveSearch) {
    const { solver } = state.liveSearch;
    if (solver === 'perfect-strategy') {
      elements.thinkingProgress.textContent = 'Verified perfect move';
    } else if (solver === 'perfect-classic-policy') {
      elements.thinkingProgress.textContent = 'Verified perfect policy move';
    } else if (solver === 'perfect-chaos-complete') {
      elements.thinkingProgress.textContent = 'Complete Chaos certificate move';
    } else if (solver === 'perfect-book') {
      elements.thinkingProgress.textContent = 'Exact opening-book move';
    } else if (solver === 'bitboard-exact' || solver === 'classic-exact') {
      elements.thinkingProgress.textContent = state.liveSearch.nodes > 0
        ? `Exact solve · ${numberFormatter.format(state.liveSearch.nodes)} positions`
        : 'Exact solve to the end';
    } else if (solver === 'chaos-certified-prefix') {
      elements.thinkingProgress.textContent = 'Certified Chaos policy move';
    } else if (solver === 'terminal') {
      elements.thinkingProgress.textContent = 'Immediate result';
    } else if (solver === 'neural-loading' || solver === 'neural-searching') {
      elements.thinkingProgress.textContent = state.liveSearch.note ?? 'Neural opponent';
    } else {
      const label = solver === 'chaos-bounded-proof' || solver === 'chaos-search+bounded-proof'
        ? 'Bounded proof · ' : '';
      elements.thinkingProgress.textContent = `${label}Depth ${state.liveSearch.depth} · ${numberFormatter.format(state.liveSearch.nodes)} positions`;
    }
  } else {
    elements.thinkingProgress.textContent = '';
  }
  renderThinkingBar();

  const dimensions = boardDimensions(state.board);
  const repetition = currentRepetitionCount();
  const pieces = [`Move ${state.moveCount}`, `${dimensions.rows}×${dimensions.cols}`, `Connect ${state.config.connect}`];
  if (repetition >= 2 && state.status === 'playing') pieces.push(`position ${repetition}/3`);
  elements.moveInfo.textContent = pieces.join(' · ');
}

/**
 * A progress bar under the status while the AI searches, and a way out of
 * a long search: Move now plays the best move found so far.
 */
function renderThinkingBar() {
  const live = state.aiThinking ? state.liveSearch : null;
  const searching = Boolean(live) && live.solver !== 'neural-loading';
  elements.thinkingBarRow.hidden = !searching;
  if (!searching) return;
  const fraction = live.fraction
    ?? (live.maximumDepth > 0 ? Math.min(1, (live.depth ?? 0) / live.maximumDepth) : null);
  if (fraction === null || !Number.isFinite(fraction)) elements.thinkingBar.removeAttribute('value');
  else elements.thinkingBar.value = fraction;
  elements.moveNowButton.hidden = !(live.solver === 'neural-searching' || isLegalAiAction(live.action));
}

function moveNow() {
  if (!state.aiThinking || !state.aiRequest) return;
  const live = state.liveSearch;
  if (live?.solver === 'neural-searching') {
    state.moveNowRequested = true;             // the search stops at its next simulation
    return;
  }
  if (!live || !isLegalAiAction(live.action)) return;
  const summary = searchSummary({ ...live, solved: false });
  cancelAiSearch();
  state.lastSearch = summary;
  state.aiError = null;
  renderAiState();
  void performAction(live.action, 'ai');
}

function renderActions() {
  const humanCanMove = canHumanAct();
  const chaosAvailable = state.config.chaosMode;
  const undoIndex = findUndoIndex();

  elements.undoButton.disabled = undoIndex < 0 || state.busy;
  elements.transformToolbar.hidden = !chaosAvailable;
  for (const actionButton of elements.chaosActions) actionButton.hidden = !chaosAvailable;
  elements.chaosKeyboardHelp.hidden = !chaosAvailable;

  const disableChaos = !chaosAvailable || !humanCanMove;
  elements.flipButton.disabled = disableChaos;
  elements.rotateCcwButton.disabled = disableChaos;
  elements.rotateCwButton.disabled = disableChaos;
}

function renderGhostPreview() {
  const { cols } = boardDimensions(state.board);
  const visible = canHumanAct() && canDrop(state.board, state.selectedColumn);
  elements.columnControls.style.setProperty('--cols', String(cols));
  elements.columnControls.style.setProperty('--selected-column', String(state.selectedColumn));
  elements.ghostDisc.className = `ghost-disc ${playerClass(state.currentPlayer)}`;
  elements.ghostDisc.hidden = !visible;
}

function updateActiveDescendant() {
  const landingRow = getDropRow(state.board, state.selectedColumn);
  for (const cell of elements.board.querySelectorAll('.cell[aria-selected="true"]')) {
    cell.removeAttribute('aria-selected');
  }
  if (landingRow < 0) {
    elements.board.removeAttribute('aria-activedescendant');
    return;
  }
  const id = `cell-${landingRow}-${state.selectedColumn}`;
  const activeCell = document.getElementById(id);
  if (!activeCell) return;
  activeCell.setAttribute('aria-selected', 'true');
  elements.board.setAttribute('aria-activedescendant', id);
}

function renderBoard() {
  const { rows, cols } = boardDimensions(state.board);
  state.selectedColumn = Math.max(0, Math.min(cols - 1, state.selectedColumn));

  elements.boardFrame.classList.remove(...COLUMN_CLASSES);
  elements.boardFrame.classList.add(`cols-${cols}`);
  elements.boardFrame.style.setProperty('--cols', String(cols));
  elements.board.setAttribute('aria-rowcount', String(rows));
  elements.board.setAttribute('aria-colcount', String(cols));
  elements.board.setAttribute('aria-busy', String(state.busy || state.aiThinking));
  elements.board.setAttribute('aria-label', `${rows} by ${cols} Connect ${state.config.connect} board`);

  const winning = new Set(state.winningCells.map(([row, column]) => `${row},${column}`));
  const boardFragment = document.createDocumentFragment();

  for (let row = 0; row < rows; row += 1) {
    const rowElement = document.createElement('div');
    rowElement.className = 'board-row';
    rowElement.setAttribute('role', 'row');

    for (let column = 0; column < cols; column += 1) {
      const value = state.board[row][column];
      const cell = document.createElement('div');
      cell.id = `cell-${row}-${column}`;
      cell.className = 'cell';
      cell.dataset.row = String(row);
      cell.dataset.column = String(column);
      cell.setAttribute('role', 'gridcell');
      cell.setAttribute('aria-rowindex', String(row + 1));
      cell.setAttribute('aria-colindex', String(column + 1));

      if (value === RED) cell.classList.add('red');
      if (value === YELLOW) cell.classList.add('yellow');
      if (value === EMPTY && column === state.selectedColumn && canHumanAct()) {
        cell.classList.add('selected-column');
      }
      if (state.lastMove?.row === row && state.lastMove?.column === column) {
        cell.classList.add('last-move');
      }
      if (winning.has(`${row},${column}`)) cell.classList.add('winning');
      if (state.dropAnimation?.row === row && state.dropAnimation?.column === column) {
        cell.classList.add('dropping');
      }

      const cellDescription = value === EMPTY ? 'empty' : `${playerName(value)} piece`;
      cell.setAttribute('aria-label', `Row ${row + 1}, column ${column + 1}: ${cellDescription}`);
      rowElement.append(cell);
    }

    boardFragment.append(rowElement);
  }

  elements.board.replaceChildren(boardFragment);
  renderGhostPreview();
  updateActiveDescendant();
}

function renderBoardSelection() {
  const humanCanMove = canHumanAct();
  for (const cell of elements.board.querySelectorAll('.cell')) {
    const selected = Number(cell.dataset.column) === state.selectedColumn
      && !cell.classList.contains('red')
      && !cell.classList.contains('yellow')
      && humanCanMove;
    cell.classList.toggle('selected-column', selected);
  }
  renderGhostPreview();
  updateActiveDescendant();
}

function announceSelectedColumn() {
  const row = getDropRow(state.board, state.selectedColumn);
  elements.selectedColumnStatus.textContent = row < 0
    ? `Column ${state.selectedColumn + 1} is full.`
    : `Column ${state.selectedColumn + 1} selected. Press Enter or Space to drop.`;
}

function renderGuidance() {
  elements.touchHelp.hidden = !coarsePointer.matches;
  elements.touchHelp.classList.toggle('is-dismissed', state.touchHintDismissed);
}

function activeAnalysisSearch() {
  return state.aiThinking ? state.liveSearch : state.lastSearch;
}

function searchIsExact(search) {
  if (!search) return false;
  return search.solver === 'perfect-strategy'
    || search.solver === 'perfect-book'
    || search.solver === 'bitboard-exact'
    || Boolean(search.solved);
}

function setAnalysisMode(mode) {
  elements.evaluationPanel.dataset.analysisMode = mode;
  elements.evaluationBalance.hidden = mode !== 'heuristic';
  elements.exactResult.hidden = mode !== 'exact';
}

function exactResultCopy(search) {
  if (state.status === 'won') return state.winner === RED ? 'You won this position' : 'AI won this position';
  if (state.status === 'draw') return 'The position ended in a draw';
  if (!search) return 'The AI will choose only game-theoretically optimal moves';
  if (search.score > 0) return 'AI can force a win';
  if (search.score < 0) return 'You can force a win';
  return 'Best play leads to a draw';
}

function renderAiRecovery() {
  const visible = isAiGame() && Boolean(state.aiError);
  elements.aiRecovery.hidden = !visible;
  if (!visible) return;
  elements.retryAiButton.disabled = state.status !== 'playing'
    || state.currentPlayer !== YELLOW
    || state.aiThinking;
  elements.switchBrutalButton.hidden = state.config.opponent !== 'perfect'
    && state.config.opponent !== 'neural';
  elements.undoAiButton.disabled = findUndoIndex() < 0;
}

function renderEvaluation() {
  const visible = isAiGame();
  elements.evaluationPanel.hidden = !visible;
  elements.matchGrid.classList.toggle('single-column', !visible);
  if (!visible) return;

  const search = activeAnalysisSearch();
  if (state.aiError) {
    setAnalysisMode('error');
    elements.evaluationLabel.textContent = 'Analysis unavailable';
    elements.evaluationDescription.textContent = 'Retry or switch opponents from the status line.';
  } else if (state.config.opponent === 'perfect' || searchIsExact(search)
      || state.status !== 'playing') {
    const resultIsKnown = searchIsExact(search) || state.status !== 'playing';
    setAnalysisMode('exact');
    elements.evaluationLabel.textContent = state.status !== 'playing'
      ? 'Round result'
      : resultIsKnown ? 'Exact result' : 'Perfect play';
    elements.evaluationDescription.textContent = state.config.opponent === 'perfect'
      ? 'Game-theoretically verified'
      : search?.solver === 'chaos-exact-graph'
        ? 'Proved by retrograde analysis'
        : 'Proved by exact analysis';
    elements.exactBadge.textContent = resultIsKnown ? 'Proved' : 'Active';
    elements.exactResultText.textContent = exactResultCopy(search);
  } else {
    setAnalysisMode('heuristic');
    let redPercent;
    let label;
    if (state.status === 'won') {
      redPercent = state.winner === RED ? 100 : 0;
      label = state.winner === RED ? 'You won' : 'AI won';
    } else if (state.status === 'draw') {
      redPercent = 50;
      label = 'Draw';
    } else {
      const score = evaluateBoard(state.board, state.config.connect, YELLOW);
      const boundedScore = Math.max(-12_000, Math.min(12_000, score));
      const yellowShare = 1 / (1 + Math.exp(-boundedScore / 1_800));
      redPercent = Math.round((1 - yellowShare) * 100);
      if (score > 2_500) label = 'AI ahead';
      else if (score > 500) label = 'AI slight edge';
      else if (score < -2_500) label = 'You are ahead';
      else if (score < -500) label = 'You have a slight edge';
      else label = 'Even';
    }

    elements.evaluationLabel.textContent = label;
    const estimate = state.config.opponent === 'neural' ? 'Network estimate' : 'Heuristic position estimate';
    elements.evaluationDescription.textContent = estimate;
    elements.evaluationBalance.style.setProperty('--you-share', `${redPercent}%`);
    elements.evaluationBalance.setAttribute('aria-label', `${estimate}: ${label.toLowerCase()}`);
  }

  renderSearchInfo();
  renderAiRecovery();
}

function renderSearchInfo() {
  if (!isAiGame()) return;
  const search = state.aiThinking ? state.liveSearch : state.lastSearch;
  if (state.aiError) {
    elements.searchInfo.textContent = state.aiError;
  } else if (state.aiThinking && !search) {
    elements.searchInfo.textContent = state.config.opponent === 'neural'
      ? 'Preparing the neural opponent…'
      : 'Loading exact data in a background worker…';
  } else if (search) {
    const details = [];
    if (search.solver === 'perfect-strategy') {
      details.push('Perfect strategy', 'Game-theoretically exact');
      if (search.strategyEntryCount) {
        details.push(`${numberFormatter.format(search.strategyEntryCount)} verified decisions`);
      }
    } else if (search.solver === 'perfect-book') {
      details.push('Perfect book', 'Exact move');
      if (search.bookEntryCount) {
        details.push(`${numberFormatter.format(search.bookEntryCount)} solved openings`);
      }
    } else if (search.solver === 'chaos-exact-graph') {
      details.push('Exact Chaos retrograde', `${numberFormatter.format(search.nodes)} states`);
      if (search.depth > 0 && search.score !== 0) {
        details.push(`forced result within ${numberFormatter.format(search.depth)} plies`);
      } else {
        details.push('cycle-safe draw analysis');
      }
    } else if (search.solver === 'chaos-certified-prefix') {
      details.push(
        'Certified Chaos policy',
        `Policy layer ${numberFormatter.format(search.certifiedFromPieces ?? 0)}→${numberFormatter.format(search.certifiedThroughPieces ?? 8)} pieces`,
      );
      if (search.strategyEntryCount) {
        details.push(`${numberFormatter.format(search.strategyEntryCount)} verified decisions`);
      }
    } else if (search.solver === 'perfect-classic-policy') {
      details.push('Perfect classic policy', 'Game-theoretically exact');
      if (search.strategyEntryCount) {
        details.push(`${numberFormatter.format(search.strategyEntryCount)} verified decisions`);
      }
    } else if (search.solver === 'perfect-chaos-complete') {
      details.push('Complete Chaos certificate', 'Game-theoretically exact');
      if (search.strategyEntryCount) {
        details.push(`${numberFormatter.format(search.strategyEntryCount)} verified decisions`);
      }
    } else if (search.solver === 'classic-exact') {
      details.push('Exact classic solve', `${numberFormatter.format(search.depth)} cells left`);
      if (search.nodes > 0) details.push(`${numberFormatter.format(search.nodes)} positions`);
    } else if (search.solver === 'terminal') {
      details.push('Immediate result');
    } else if (search.solver === 'neural') {
      details.push(
        'Neural network',
        `${numberFormatter.format(search.nodes)} simulations`,
        search.backend === 'webgpu' ? 'on the GPU' : 'on the CPU',
        `${(search.elapsedMs / 1_000).toFixed(1)}s`,
      );
    } else {
      const seconds = search.elapsedMs / 1_000;
      const rate = search.elapsedMs > 0 ? Math.round(search.nodes / seconds) : 0;
      if (search.solver === 'bitboard-exact') details.push('Exact terminal solve');
      else if (search.solver === 'chaos-bounded-proof' || search.solver === 'chaos-search+bounded-proof') {
        details.push(search.solved ? 'Bounded Chaos proof, solved' : 'Bounded Chaos proof');
      } else if (search.solver === 'bitboard') details.push(search.solved ? 'Bitboard solved' : 'Bitboard');
      else if (search.solved) details.push('Solved');
      if (search.solver !== 'bitboard-exact') details.push(`Depth ${search.depth}`);
      details.push(
        `${numberFormatter.format(search.nodes)} positions`,
        `${seconds.toFixed(seconds >= 1 ? 1 : 2)}s`,
      );
      if (rate > 0) details.push(`${numberFormatter.format(rate)}/s`);
    }
    elements.searchInfo.textContent = details.join(' · ');
  } else {
    elements.searchInfo.textContent = 'The AI has not searched yet.';
  }
}

function clearBoardAnimations() {
  elements.boardFrame.classList.remove(...ANIMATION_CLASSES);
}

function animationPlan(action) {
  return TRANSFORM_ANIMATIONS[action.type] ?? null;
}

function pause(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, reducedMotion ? 0 : milliseconds));
}

async function performAction(action, source = 'human') {
  if (!action || state.status !== 'playing' || state.busy) return;
  if (source === 'human' && !canHumanAct()) return;
  if (action.type === ACTION_DROP && !canDrop(state.board, action.column)) return;
  if (action.type !== ACTION_DROP && !state.config.chaosMode) return;

  if (source === 'human' && !state.touchHintDismissed) {
    state.touchHintDismissed = true;
    renderGuidance();
  }

  const actor = state.currentPlayer;
  const result = applyAction(state.board, action, actor);
  if (!result) return;

  const roundVersion = state.version;
  state.busy = true;
  state.lastMover = actor;
  renderStatus();
  renderActions();

  const animation = animationPlan(action);
  if (animation) {
    clearBoardAnimations();
    elements.boardFrame.classList.add(animation.outClass);
    await pause(animation.outMs);
    if (roundVersion !== state.version) return;
  }

  state.board = result.board;
  state.moveCount += 1;
  state.lastMove = action.type === ACTION_DROP
    ? { row: result.row, column: result.column }
    : null;
  state.dropAnimation = action.type === ACTION_DROP
    ? { row: result.row, column: result.column }
    : null;
  renderAll();

  if (animation) {
    elements.boardFrame.classList.remove(animation.outClass);
    elements.boardFrame.classList.add(animation.inClass);
    await pause(animation.inMs);
    elements.boardFrame.classList.remove(animation.inClass);
  } else {
    await pause(360);
  }
  if (roundVersion !== state.version) return;
  state.dropAnimation = null;

  const outcome = resolveActionOutcome(
    state.board,
    state.config.connect,
    actor,
    action.type,
    action.type === ACTION_DROP ? { row: result.row, column: result.column } : null,
  );

  if (outcome.status === 'won') {
    state.status = 'won';
    state.winner = outcome.winner;
    state.winningCells = outcome.winningCells;
    state.simultaneousWin = outcome.simultaneousWin;
    state.scores[outcome.winner] += 1;
  } else if (outcome.status === 'draw') {
    state.status = 'draw';
    state.drawReason = 'full';
    state.scores.draw += 1;
  } else {
    state.currentPlayer = otherPlayer(actor);
    const key = positionKey(
      state.board,
      state.currentPlayer,
      state.config.connect,
      state.config.chaosMode,
    );
    const repetitions = (state.repetitionCounts.get(key) ?? 0) + 1;
    state.repetitionCounts.set(key, repetitions);

    if (repetitions >= 3) {
      state.status = 'draw';
      state.drawReason = 'repetition';
      state.scores.draw += 1;
    }
  }

  state.busy = false;
  if (state.status !== 'playing') disposeAiWorker();
  saveJson(SCORES_KEY, state.scores);
  pushSnapshot();
  saveRound();
  renderAll();

  if (state.status !== 'playing') {
    await pause(260);
    if (roundVersion === state.version) showResultDialog();
  } else if (isAiGame() && state.currentPlayer === YELLOW) {
    requestAiMove();
  }
}

function isLegalAiAction(action) {
  return legalActions(state.board, state.config.chaosMode).some((candidate) => (
    candidate.type === action?.type
      && (candidate.type !== ACTION_DROP || candidate.column === action.column)
  ));
}

function disposeAiWorker(worker = state.aiWorker) {
  if (!worker) return;
  worker.terminate();
  if (state.aiWorker === worker) state.aiWorker = null;
}

function cancelAiSearch() {
  state.aiRequestId += 1;
  state.aiRequest = null;
  disposeAiWorker();
  state.aiThinking = false;
  state.liveSearch = null;
}

function stopAiWithError(message) {
  state.aiRequest = null;
  disposeAiWorker();
  state.aiThinking = false;
  state.liveSearch = null;
  state.aiError = message || 'The AI could not make a verified legal move.';
  renderAll();
}

function searchSummary(result) {
  return {
    score: result.score ?? 0,
    depth: result.depth ?? 0,
    nodes: result.nodes ?? 0,
    elapsedMs: result.elapsedMs ?? 0,
    solved: Boolean(result.solved),
    solver: result.solver ?? 'general',
    bookEntryCount: result.bookEntryCount ?? null,
    strategyEntryCount: result.strategyEntryCount ?? null,
    certifiedFromPieces: result.certifiedFromPieces ?? null,
    certifiedThroughPieces: result.certifiedThroughPieces ?? null,
    backend: result.backend ?? null,
  };
}

function renderAiState() {
  renderStatus();
  renderEvaluation();
  renderGhostPreview();
  elements.board.setAttribute('aria-busy', String(state.busy || state.aiThinking));
}

function fallbackOrStop(request, message) {
  if (!request.perfectRequested
      && request.options?.difficulty === 'easy'
      && !request.fallbackStarted) {
    runFallback(request);
    return;
  }
  stopAiWithError(message);
}

function finishAiRequest(request, payload) {
  if (state.aiRequest !== request
      || request.id !== state.aiRequestId
      || request.roundVersion !== state.version) return;

  const result = payload?.result;
  if (!isLegalAiAction(result?.action)) {
    disposeAiWorker();
    fallbackOrStop(request, request.perfectRequested
      ? 'The verified perfect strategy returned no legal move.'
      : 'The selected AI returned no verified legal move. Retry or choose another level.');
    return;
  }

  state.aiRequest = null;
  state.aiThinking = false;
  state.liveSearch = null;
  state.aiError = null;
  state.lastSearch = searchSummary(result);
  void performAction(result.action, 'ai');
}

function runFallback(request) {
  if (request.fallbackStarted) return;
  request.fallbackStarted = true;
  if (request.perfectRequested) {
    stopAiWithError('The verified perfect strategy could not be loaded.');
    return;
  }

  setTimeout(() => {
    if (state.aiRequest !== request
        || request.id !== state.aiRequestId
        || request.roundVersion !== state.version) return;
    try {
      finishAiRequest(request, {
        result: chooseMove(request.position, {
          ...request.options,
          difficulty: 'easy',
          onIteration(progress) {
            state.liveSearch = progress;
            renderAiState();
          },
        }),
      });
    } catch {
      finishAiRequest(request, { result: null });
    }
  }, 0);
}

function handleAiWorkerMessage(worker, event) {
  if (state.aiWorker !== worker) return;
  const request = state.aiRequest;
  if (!request || event.data?.requestId !== request.id) return;

  if (event.data.kind === 'progress') {
    state.liveSearch = event.data.progress ?? null;
    renderAiState();
    return;
  }
  if (event.data.kind === 'error') {
    disposeAiWorker();
    fallbackOrStop(request, request.perfectRequested
      ? event.data.error || 'The verified perfect strategy failed.'
      : event.data.error || 'The selected AI worker failed. Retry the move.');
    return;
  }
  if (event.data.kind === 'result') {
    finishAiRequest(request, event.data);
    return;
  }
  handleAiWorkerError(worker, {
    message: 'The AI worker returned an unexpected response.',
  });
}

function handleAiWorkerError(worker, event) {
  if (state.aiWorker !== worker) return;
  const request = state.aiRequest;
  disposeAiWorker(worker);
  if (!request) return;
  fallbackOrStop(request, request.perfectRequested
    ? event.message || 'The verified perfect strategy worker failed.'
    : event.message || 'The selected AI worker failed. Retry the move.');
}

function ensureAiWorker() {
  if (state.aiWorker) return state.aiWorker;
  const worker = new Worker(new URL('./ai-worker.js', import.meta.url), { type: 'module' });
  worker.addEventListener('message', (event) => handleAiWorkerMessage(worker, event));
  worker.addEventListener('error', (event) => handleAiWorkerError(worker, event));
  worker.addEventListener('messageerror', () => handleAiWorkerError(worker, {
    message: 'The AI worker returned an unreadable message.',
  }));
  state.aiWorker = worker;
  return worker;
}

async function runNeuralMove(request) {
  const stale = () => state.aiRequest !== request
    || request.id !== state.aiRequestId
    || request.roundVersion !== state.version;
  try {
    const {
      DOWNLOAD_BYTES, cancelNeuralLoad, loadNeuralNetwork, neuralLoadState, recordSearch,
      simulationsFor,
    } = await import('./neural-runtime.js');
    const { bestAction, searchPosition } = await import('./neural-search.js');
    const { requestDownload, showDownloadProgress } = await import('./download-gate.js');
    let panel = null;
    let cancelled = false;
    if (neuralLoadState() !== 'ready') {
      if (neuralLoadState() === 'idle') {
        // Nobody should start a 73 MB download by picking an option in a
        // select box, so the page asks first and remembers a yes.
        const agreed = await requestDownload({
          id: 'neural-opponent',
          title: 'Neural opponent',
          description: 'The neural opponent is a trained network plus a search. Playing it needs a one-time download of the network and its runtime.',
          bytes: DOWNLOAD_BYTES.model + DOWNLOAD_BYTES.runtime,
        });
        if (stale()) return;
        if (!agreed) {
          stopAiWithError('The neural opponent needs a one-time download. Choose Download when asked, or pick another opponent.');
          return;
        }
      }
      // A download already in flight (from a request since undone) shows
      // its progress here as well.
      panel = showDownloadProgress({
        title: 'Neural opponent',
        note: 'Downloading the network and its runtime. This happens once; your browser keeps them.',
        onCancel: () => {
          cancelled = true;
          cancelNeuralLoad();
        },
      });
    }
    let network;
    try {
      network = await loadNeuralNetwork({
        onProgress: (progress) => {
          if (stale()) {
            panel?.close();
            panel = null;
            return;
          }
          if (panel && progress.stage === 'session') {
            panel.note(`Starting the network on ${progress.backend}. This can take a moment.`);
          } else if (panel) {
            panel.update(progress.loaded ?? 0, progress.total ?? 0, 'Downloaded');
          }
          state.liveSearch = {
            solver: 'neural-loading',
            note: progress.stage === 'session'
              ? `Starting the network on ${progress.backend}`
              : 'Downloading the network (once)',
          };
          renderAiState();
        },
      });
    } catch (error) {
      if (stale()) return;
      if (cancelled || error?.name === 'AbortError') {
        stopAiWithError('The neural download was cancelled. Retry to download it, or pick another opponent.');
        return;
      }
      throw error;
    } finally {
      panel?.close();
    }
    if (stale()) return;
    const simulations = simulationsFor(network);
    state.liveSearch = {
      solver: 'neural-searching',
      note: `Neural search · ${simulations} simulations on ${network.backend}`,
    };
    renderAiState();
    const started = performance.now();
    const result = await searchPosition(request.position, network.evaluate, {
      simulations,
      shouldStop: () => stale() || state.moveNowRequested,
      repeated: rootRepetition(request.position),
      onProgress: (done, total) => {
        if (stale()) return;
        state.liveSearch = { ...state.liveSearch, fraction: done / total };
        renderStatus();
      },
    });
    if (stale()) return;
    recordSearch(network, performance.now() - started, simulations);
    const action = bestAction(result);
    if (!action) {
      fallbackOrStop(request, 'The neural opponent found no legal move.');
      return;
    }
    finishAiRequest(request, {
      result: {
        action,
        score: result.value,
        depth: 0,
        nodes: simulations,
        elapsedMs: performance.now() - started,
        solver: 'neural',
        solved: false,
        backend: network.backend,
      },
    });
  } catch (error) {
    if (stale()) return;
    fallbackOrStop(request, `The neural opponent could not start: ${error.message}`);
  }
}

function requestAiMove() {
  if (!isAiGame()
      || state.currentPlayer !== YELLOW
      || state.status !== 'playing'
      || state.busy
      || state.aiThinking) return;

  const request = {
    id: state.aiRequestId + 1,
    roundVersion: state.version,
    position: {
      board: cloneBoard(state.board),
      currentPlayer: state.currentPlayer,
      connect: state.config.connect,
      chaosMode: state.config.chaosMode,
      startingPlayer: state.config.startingPlayer,
      repetitionCounts: [...state.repetitionCounts.entries()],
    },
    options: {
      difficulty: state.config.opponent,
      aiPlayer: YELLOW,
    },
    perfectRequested: state.config.opponent === 'perfect',
    fallbackStarted: false,
  };

  state.aiRequestId = request.id;
  state.aiRequest = request;
  state.aiThinking = true;
  state.aiError = null;
  state.liveSearch = null;
  state.moveNowRequested = false;
  renderAiState();

  if (state.config.opponent === 'neural') {
    // The network and its runtime are a large download, so they load on
    // first use rather than with the page, and the search runs here rather
    // than in the worker, which has no access to them.
    void runNeuralMove(request);
    return;
  }

  if (request.perfectRequested && state.config.chaosMode) {
    // The complete Chaos policies run to tens of megabytes; ask before the
    // worker fetches one, and show the download rather than a spinner.
    void gateExactTableThenPost(request);
    return;
  }
  postToWorker(request);
}

function postToWorker(request) {
  if (state.aiRequest !== request || request.id !== state.aiRequestId) return;
  try {
    ensureAiWorker().postMessage({
      requestId: request.id,
      position: request.position,
      options: request.options,
    });
  } catch {
    disposeAiWorker();
    fallbackOrStop(request, request.perfectRequested
      ? 'The verified perfect strategy worker could not start.'
      : 'The selected AI worker could not start. Retry the move.');
  }
}

/** How often the position to search has already occurred this round (0, 1 or 2). */
function rootRepetition(position) {
  const counts = new Map(position.repetitionCounts ?? []);
  const key = positionKey(position.board, position.currentPlayer, position.connect, position.chaosMode);
  return Math.max(0, Math.min(2, (counts.get(key) ?? 1) - 1));
}

const LARGE_TABLE_BYTES = 8_000_000;
const TABLE_DOWNLOAD_TIMEOUT_MS = 600_000;

async function gateExactTableThenPost(request) {
  const stale = () => state.aiRequest !== request || request.id !== state.aiRequestId;
  try {
    const {
      findPerfectChaosCompletePolicy, loadPerfectChaosCompleteManifest, perfectChaosCompleteRole,
    } = await import('./perfect-chaos-complete.js');
    const manifestUrl = new URL('../data/perfect-chaos-complete/manifest.json', import.meta.url);
    const manifest = await loadPerfectChaosCompleteManifest(manifestUrl);
    const { rows, cols } = boardDimensions(request.position.board);
    const role = perfectChaosCompleteRole(request.position.startingPlayer, YELLOW);
    const entry = role === null ? null : findPerfectChaosCompletePolicy(
      manifest, rows, cols, request.position.connect, role,
    );
    if (stale()) return;
    if (entry && Number(entry.bytes) > LARGE_TABLE_BYTES && !loadedExactTables.has(entry.file)) {
      const { fetchWithProgress, requestDownload, showDownloadProgress } = await import('./download-gate.js');
      const agreed = await requestDownload({
        id: `exact-${entry.file}`,
        title: `Perfect ${rows}×${cols} Chaos`,
        description: 'Perfect play on this board reads a complete solved table. It is downloaded once and kept by your browser.',
        bytes: Number(entry.bytes),
      });
      if (stale()) return;
      if (!agreed) {
        stopAiWithError('Perfect play on this board needs a one-time table download. Choose Download when asked, or pick another opponent.');
        return;
      }
      const controller = new AbortController();
      let timedOut = false;
      const deadline = setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, TABLE_DOWNLOAD_TIMEOUT_MS);
      const panel = showDownloadProgress({
        title: `Perfect ${rows}×${cols} Chaos`,
        note: 'Downloading the solved table.',
        onCancel: () => controller.abort(),
      });
      try {
        // Warms the browser cache; the worker's own fetch then completes at once.
        await fetchWithProgress(new URL(entry.file, manifestUrl).href,
          (loaded, total) => panel.update(loaded, total, 'Downloaded'),
          { signal: controller.signal, expectedBytes: Number(entry.bytes) });
        loadedExactTables.add(entry.file);
      } catch (error) {
        if (stale()) return;
        if (timedOut) {
          stopAiWithError('The solved table did not finish downloading within ten minutes. Retry, or pick another opponent.');
        } else if (error?.name === 'AbortError') {
          stopAiWithError('The table download was cancelled. Retry to download it, or pick another opponent.');
        } else {
          stopAiWithError(`The solved table could not be downloaded: ${error.message}`);
        }
        return;
      } finally {
        clearTimeout(deadline);
        panel.close();
      }
      if (stale()) return;
    }
  } catch {
    // The worker reports its own failures; a gate that cannot resolve the
    // table simply steps aside.
  }
  postToWorker(request);
}

const loadedExactTables = new Set();

function chooseColumn(column, announce = false) {
  const { cols } = boardDimensions(state.board);
  const nextColumn = Math.max(0, Math.min(cols - 1, column));
  if (nextColumn !== state.selectedColumn) {
    state.selectedColumn = nextColumn;
    renderBoardSelection();
  }
  if (announce) announceSelectedColumn();
}

function moveSelectedColumn(delta) {
  chooseColumn(state.selectedColumn + delta, true);
}

function columnFromPointer(event, element) {
  const { cols } = boardDimensions(state.board);
  const bounds = element.getBoundingClientRect();
  if (bounds.width <= 0) return state.selectedColumn;
  const fraction = Math.max(0, Math.min(0.999999, (event.clientX - bounds.left) / bounds.width));
  return Math.floor(fraction * cols);
}

function dropSelectedColumn() {
  if (!canHumanAct()) return;
  void performAction({ type: ACTION_DROP, column: state.selectedColumn });
}

function showResultDialog() {
  if (state.status === 'playing') return;

  elements.dialogDisc.className = 'dialog-disc';
  if (state.status === 'won') {
    elements.dialogDisc.classList.add(playerClass(state.winner));
    elements.dialogTitle.textContent = `${playerName(state.winner)} wins`;
    if (state.simultaneousWin) {
      elements.dialogMessage.textContent = `${playerName(state.lastMover)} created winning lines for both sides. In chaos mode, the transforming player loses that tie.`;
    } else {
      elements.dialogMessage.textContent = `${state.config.connect} connected pieces end the round.`;
    }
  } else {
    elements.dialogDisc.classList.add('draw');
    elements.dialogTitle.textContent = 'Draw';
    elements.dialogMessage.textContent = state.drawReason === 'repetition'
      ? 'The same position, with the same player to move, occurred three times.'
      : 'Every space is occupied and nobody connected enough pieces.';
  }

  if (!elements.resultDialog.open) elements.resultDialog.showModal();
}

function closeResultDialog() {
  if (elements.resultDialog.open) elements.resultDialog.close();
}

function openRuleEditor() {
  closeResultDialog();
  setSettingsExpanded(true);
  elements.setupPanel.scrollIntoView({
    behavior: reducedMotion ? 'auto' : 'smooth',
    block: 'start',
  });
  requestAnimationFrame(() => elements.rowsInput.focus({ preventScroll: true }));
}

function retryAiMove() {
  if (state.status !== 'playing' || state.currentPlayer !== YELLOW || state.aiThinking) return;
  state.aiError = null;
  renderAll();
  requestAiMove();
}

function switchToBrutal() {
  if (state.config.opponent !== 'perfect' && state.config.opponent !== 'neural') return;
  state.config = normalizeConfig({ ...state.config, opponent: 'brutal' });
  populateSettingsForm(state.config);
  saveJson(SETTINGS_KEY, state.config);
  state.aiError = null;
  state.lastSearch = null;
  saveRound();
  renderAll();
  if (state.currentPlayer === YELLOW && state.status === 'playing') requestAiMove();
}

function isTypingTarget(target) {
  return target instanceof HTMLInputElement
    || target instanceof HTMLSelectElement
    || target instanceof HTMLTextAreaElement
    || target instanceof HTMLButtonElement;
}

function handleBoardKeydown(event) {
  if (event.key === 'ArrowLeft') {
    moveSelectedColumn(-1);
    event.preventDefault();
  } else if (event.key === 'ArrowRight') {
    moveSelectedColumn(1);
    event.preventDefault();
  } else if (event.key === 'Home') {
    chooseColumn(0, true);
    event.preventDefault();
  } else if (event.key === 'End') {
    chooseColumn(boardDimensions(state.board).cols - 1, true);
    event.preventDefault();
  } else if (event.key === 'Enter' || event.key === ' ') {
    dropSelectedColumn();
    event.preventDefault();
  }
}

function handleGlobalKeydown(event) {
  // Plain letters only: Ctrl+U, Ctrl+F, Ctrl+R and Alt+N belong to the browser.
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  if (isTypingTarget(event.target)
      || elements.resultDialog.open
      || elements.rulesDialog.open
      || elements.downloadDialog.open) return;
  const key = event.key.toLowerCase();

  if (key === 'n') {
    restartRound();
    event.preventDefault();
  } else if (key === 'u') {
    undoTurn();
    event.preventDefault();
  } else if (key === 'f' && state.config.chaosMode) {
    void performAction({ type: ACTION_FLIP });
    event.preventDefault();
  } else if (key === 'r' && state.config.chaosMode) {
    void performAction({ type: event.shiftKey ? ACTION_ROTATE_CCW : ACTION_ROTATE_CW });
    event.preventDefault();
  }
}

elements.settingsForm.addEventListener('submit', (event) => {
  event.preventDefault();
  applySettingsAndStartRound();
});
const updateRuleForm = () => {
  updateConnectLimit();
  updatePerfectAvailability();
};
elements.rowsInput.addEventListener('input', updateRuleForm);
elements.colsInput.addEventListener('input', updateRuleForm);
elements.connectInput.addEventListener('input', updatePerfectAvailability);
elements.chaosInput.addEventListener('change', updatePerfectAvailability);
elements.opponentInput.addEventListener('change', updateOpponentLabels);
elements.settingsToggle.addEventListener('click', () => {
  setSettingsExpanded(elements.settingsBody.hidden);
});
elements.resetScoreButton.addEventListener('click', resetScores);
elements.restartButton.addEventListener('click', () => restartRound());
elements.undoButton.addEventListener('click', undoTurn);
elements.flipButton.addEventListener('click', () => void performAction({ type: ACTION_FLIP }));
elements.rotateCcwButton.addEventListener('click', () => void performAction({ type: ACTION_ROTATE_CCW }));
elements.rotateCwButton.addEventListener('click', () => void performAction({ type: ACTION_ROTATE_CW }));
elements.reviewBoardButton.addEventListener('click', closeResultDialog);
elements.changeRulesButton.addEventListener('click', openRuleEditor);
elements.playAgainButton.addEventListener('click', () => restartRound());
elements.retryAiButton.addEventListener('click', retryAiMove);
elements.moveNowButton.addEventListener('click', moveNow);
elements.switchBrutalButton.addEventListener('click', switchToBrutal);
elements.undoAiButton.addEventListener('click', undoTurn);
elements.rulesButton.addEventListener('click', () => {
  if (!elements.rulesDialog.open) elements.rulesDialog.showModal();
});
elements.closeRulesButton.addEventListener('click', () => elements.rulesDialog.close());
elements.rulesDoneButton.addEventListener('click', () => elements.rulesDialog.close());
elements.rulesDialog.addEventListener('click', (event) => {
  if (event.target !== elements.rulesDialog) return;
  const bounds = elements.rulesDialog.getBoundingClientRect();
  const onBackdrop = event.clientX < bounds.left || event.clientX > bounds.right
    || event.clientY < bounds.top || event.clientY > bounds.bottom;
  if (onBackdrop) elements.rulesDialog.close();
});
elements.board.addEventListener('keydown', handleBoardKeydown);
elements.board.addEventListener('focus', announceSelectedColumn);
elements.board.addEventListener('pointermove', (event) => {
  const cell = event.target.closest?.('.cell');
  if (cell) chooseColumn(Number(cell.dataset.column));
});
elements.board.addEventListener('click', (event) => {
  const cell = event.target.closest?.('.cell');
  if (!cell) return;
  chooseColumn(Number(cell.dataset.column));
  dropSelectedColumn();
});
elements.columnControls.addEventListener('pointermove', (event) => {
  chooseColumn(columnFromPointer(event, elements.columnControls));
});
elements.columnControls.addEventListener('click', (event) => {
  chooseColumn(columnFromPointer(event, elements.columnControls));
  dropSelectedColumn();
});
document.addEventListener('keydown', handleGlobalKeydown);

const savedRound = loadJson(ROUND_KEY, null);
startRound(state.config, { collapseSettings: state.gameFirstLayout });
if (savedRound) restoreSavedRound(savedRound);
