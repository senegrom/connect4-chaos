import { chooseMove, evaluateBoard } from './ai.js';
import {
  ACTION_DROP,
  ACTION_FLIP,
  ACTION_ROTATE_CCW,
  ACTION_ROTATE_CW,
  EMPTY,
  RED,
  YELLOW,
  actionLabel,
  applyAction,
  boardDimensions,
  canDrop,
  cloneBoard,
  createBoard,
  legalActions,
  normalizeConfig,
  otherPlayer,
  positionKey,
  resolveActionOutcome,
} from './engine.js';

const SETTINGS_KEY = 'connect4-chaos.settings.v1';
const SCORES_KEY = 'connect4-chaos.scores.v1';
const DIFFICULTY_LABELS = Object.freeze({
  human: 'Human',
  easy: 'Easy AI',
  medium: 'Medium AI',
  hard: 'Hard AI',
  brutal: 'Brutal AI',
});
const DIFFICULTY_HINTS = Object.freeze({
  human: 'Two people share this device.',
  easy: 'Instant moves with basic wins and blocks.',
  medium: 'Checks the exact opening book, then searches 10 plies.',
  hard: 'Uses exact book moves, 14-ply search, and late-game solving.',
  brutal: 'Uses exact book moves, 16-ply search, and the largest exact endgame.',
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
  settingsBody: document.querySelector('#settingsBody'),
  settingsToggle: document.querySelector('#settingsToggle'),
  activeRulesSummary: document.querySelector('#activeRulesSummary'),
  settingsForm: document.querySelector('#settingsForm'),
  rowsInput: document.querySelector('#rowsInput'),
  colsInput: document.querySelector('#colsInput'),
  connectInput: document.querySelector('#connectInput'),
  opponentInput: document.querySelector('#opponentInput'),
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
  evaluationPercent: document.querySelector('#evaluationPercent'),
  evaluationMeter: document.querySelector('#evaluationMeter'),
  searchInfo: document.querySelector('#searchInfo'),
  statusDisc: document.querySelector('#statusDisc'),
  statusText: document.querySelector('#statusText'),
  moveInfo: document.querySelector('#moveInfo'),
  gamePanel: document.querySelector('#gamePanel'),
  thinkingIndicator: document.querySelector('#thinkingIndicator'),
  thinkingProgress: document.querySelector('#thinkingProgress'),
  restartButton: document.querySelector('#restartButton'),
  undoButton: document.querySelector('#undoButton'),
  flipButton: document.querySelector('#flipButton'),
  rotateCcwButton: document.querySelector('#rotateCcwButton'),
  rotateCwButton: document.querySelector('#rotateCwButton'),
  chaosActions: [...document.querySelectorAll('.chaos-action')],
  chaosKeyboardHelp: document.querySelector('#chaosKeyboardHelp'),
  boardFrame: document.querySelector('#boardFrame'),
  columnControls: document.querySelector('#columnControls'),
  board: document.querySelector('#gameBoard'),
  resultDialog: document.querySelector('#resultDialog'),
  dialogDisc: document.querySelector('#dialogDisc'),
  dialogTitle: document.querySelector('#dialogTitle'),
  dialogMessage: document.querySelector('#dialogMessage'),
  reviewBoardButton: document.querySelector('#reviewBoardButton'),
  playAgainButton: document.querySelector('#playAgainButton'),
  rulesButton: document.querySelector('#rulesButton'),
  rulesDialog: document.querySelector('#rulesDialog'),
  closeRulesButton: document.querySelector('#closeRulesButton'),
  rulesDoneButton: document.querySelector('#rulesDoneButton'),
};

const reducedMotion = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
const compactViewport = globalThis.matchMedia?.('(max-width: 39rem)') ?? { matches: false };
const numberFormatter = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });

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
  lastAction: '',
  moveCount: 0,
  selectedColumn: 0,
  repetitionCounts: new Map(),
  scores: normalizeScores(loadJson(SCORES_KEY, {})),
  history: [],
  busy: false,
  aiThinking: false,
  aiWorker: null,
  aiRequestId: 0,
  version: 0,
  dropAnimation: null,
  lastSearch: null,
  liveSearch: null,
};

function loadJson(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
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
  updateOpponentLabels();
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

function activeRulesText(config = state.config) {
  const opponent = DIFFICULTY_LABELS[config.opponent] ?? DIFFICULTY_LABELS.medium;
  const chaos = config.chaosMode ? ' · Chaos' : '';
  return `${config.rows}×${config.cols} · Connect ${config.connect} · ${opponent}${chaos}`;
}

function renderActiveRulesSummary() {
  elements.activeRulesSummary.textContent = activeRulesText();
}

function setSettingsExpanded(expanded, focusToggle = false) {
  elements.settingsBody.hidden = !expanded;
  elements.settingsToggle.setAttribute('aria-expanded', String(expanded));
  elements.settingsToggle.textContent = expanded ? 'Hide settings' : 'Change rules';
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
    lastAction: state.lastAction,
    moveCount: state.moveCount,
    selectedColumn: state.selectedColumn,
    repetitionCounts: [...state.repetitionCounts.entries()],
    scores: { ...state.scores },
    lastSearch: state.lastSearch
      ? {
        ...state.lastSearch,
        action: state.lastSearch.action ? { ...state.lastSearch.action } : null,
        principalVariation: state.lastSearch.principalVariation?.map((action) => ({ ...action })) ?? [],
      }
      : null,
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
  state.lastAction = snapshot.lastAction;
  state.moveCount = snapshot.moveCount;
  state.selectedColumn = snapshot.selectedColumn;
  state.repetitionCounts = new Map(snapshot.repetitionCounts);
  state.scores = { ...snapshot.scores };
  state.lastSearch = snapshot.lastSearch
    ? {
      ...snapshot.lastSearch,
      action: snapshot.lastSearch.action ? { ...snapshot.lastSearch.action } : null,
      principalVariation: snapshot.lastSearch.principalVariation?.map((action) => ({ ...action })) ?? [],
    }
    : null;
  state.liveSearch = null;
  state.dropAnimation = null;
}

function pushSnapshot() {
  state.history.push(makeSnapshot());
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
    collapseSettings = compactViewport.matches,
    scrollToGame = false,
  } = options;

  cancelAiSearch();
  closeResultDialog();
  clearBoardAnimations();

  state.config = normalizeConfig(config);
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
  state.lastAction = '';
  state.moveCount = 0;
  state.selectedColumn = Math.floor(state.config.cols / 2);
  state.repetitionCounts = new Map();
  state.busy = false;
  state.aiThinking = false;
  state.dropAnimation = null;
  state.lastSearch = null;
  state.liveSearch = null;
  state.history = [];

  const initialKey = positionKey(
    state.board,
    state.currentPlayer,
    state.config.connect,
    state.config.chaosMode,
  );
  state.repetitionCounts.set(initialKey, 1);
  pushSnapshot();
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
    collapseSettings: compactViewport.matches,
    scrollToGame: compactViewport.matches,
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
  saveJson(SCORES_KEY, state.scores);
  renderAll();
}

function resetScores() {
  state.scores = { [RED]: 0, [YELLOW]: 0, draw: 0 };
  for (const snapshot of state.history) snapshot.scores = { ...state.scores };
  saveJson(SCORES_KEY, state.scores);
  renderScores();
}

function renderAll() {
  renderActiveRulesSummary();
  renderScores();
  renderStatus();
  renderActions();
  renderBoard();
  renderEvaluation();
}

function renderScores() {
  elements.redScore.textContent = String(state.scores[RED]);
  elements.yellowScore.textContent = String(state.scores[YELLOW]);
  elements.drawScore.textContent = String(state.scores.draw);
  elements.yellowScoreLabel.textContent = isAiGame() ? 'AI' : 'Yellow';
}

function statusMessage() {
  if (state.status === 'won') {
    if (state.simultaneousWin) return `${playerName(state.winner)} wins — the transformer made both lines`;
    return `${playerName(state.winner)} wins!`;
  }
  if (state.status === 'draw') {
    return state.drawReason === 'repetition' ? 'Draw by threefold repetition' : 'Draw — the board is full';
  }
  if (state.aiThinking) return 'AI is thinking…';
  if (state.busy && state.lastMover) return `${playerName(state.lastMover)} is moving…`;

  return `${playerName(state.currentPlayer)} to move`;
}

function renderStatus() {
  const displayPlayer = state.status === 'won' ? state.winner : state.currentPlayer;
  elements.statusDisc.className = `turn-disc ${playerClass(displayPlayer || RED)}`;
  elements.statusText.textContent = statusMessage();
  elements.thinkingIndicator.hidden = false;
  elements.thinkingIndicator.classList.toggle('is-idle', !state.aiThinking);
  elements.thinkingIndicator.setAttribute('aria-hidden', String(!state.aiThinking));
  if (state.aiThinking && state.liveSearch) {
    elements.thinkingProgress.textContent = state.liveSearch.solver === 'perfect-book'
      ? 'Exact opening-book move'
      : `Depth ${state.liveSearch.depth} · ${numberFormatter.format(state.liveSearch.nodes)} positions`;
  } else {
    elements.thinkingProgress.textContent = '';
  }

  const dimensions = boardDimensions(state.board);
  const repetition = currentRepetitionCount();
  const pieces = [`Move ${state.moveCount}`, `${dimensions.rows}×${dimensions.cols}`, `Connect ${state.config.connect}`];
  if (repetition >= 2 && state.status === 'playing') pieces.push(`position ${repetition}/3`);
  elements.moveInfo.textContent = pieces.join(' · ');
}

function renderActions() {
  const humanCanMove = canHumanAct();
  const chaosAvailable = state.config.chaosMode;
  const undoIndex = findUndoIndex();

  elements.undoButton.disabled = undoIndex < 0 || state.busy;
  for (const actionButton of elements.chaosActions) actionButton.hidden = !chaosAvailable;
  elements.chaosKeyboardHelp.hidden = !chaosAvailable;

  const disableChaos = !chaosAvailable || !humanCanMove;
  elements.flipButton.disabled = disableChaos;
  elements.rotateCcwButton.disabled = disableChaos;
  elements.rotateCwButton.disabled = disableChaos;
}

function renderBoard() {
  const { rows, cols } = boardDimensions(state.board);
  state.selectedColumn = Math.max(0, Math.min(cols - 1, state.selectedColumn));

  elements.boardFrame.classList.remove(...COLUMN_CLASSES);
  elements.boardFrame.classList.add(`cols-${cols}`);
  elements.board.setAttribute('aria-rowcount', String(rows));
  elements.board.setAttribute('aria-colcount', String(cols));
  elements.board.setAttribute('aria-busy', String(state.busy || state.aiThinking));
  elements.board.setAttribute(
    'aria-label',
    `${rows} by ${cols} Connect ${state.config.connect} board. Use left and right arrows to choose a column, then Enter or Space to drop.`,
  );

  const controlsFragment = document.createDocumentFragment();
  for (let column = 0; column < cols; column += 1) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'column-button';
    button.dataset.column = String(column);
    button.textContent = `Column ${column + 1}`;
    button.setAttribute('aria-label', `Drop in column ${column + 1}`);
    button.setAttribute('aria-pressed', String(column === state.selectedColumn));
    button.disabled = !canHumanAct() || !canDrop(state.board, column);
    if (column === state.selectedColumn) {
      button.classList.add('selected', playerClass(state.currentPlayer));
    }
    controlsFragment.append(button);
  }
  elements.columnControls.replaceChildren(controlsFragment);

  const winning = new Set(state.winningCells.map(([row, column]) => `${row},${column}`));
  const boardFragment = document.createDocumentFragment();

  for (let row = 0; row < rows; row += 1) {
    const rowElement = document.createElement('div');
    rowElement.className = 'board-row';
    rowElement.setAttribute('role', 'row');

    for (let column = 0; column < cols; column += 1) {
      const value = state.board[row][column];
      const cell = document.createElement('div');
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
}

function renderBoardSelection() {
  const humanCanMove = canHumanAct();
  for (const button of elements.columnControls.querySelectorAll('.column-button')) {
    const column = Number(button.dataset.column);
    const selected = column === state.selectedColumn;
    button.classList.toggle('selected', selected);
    button.classList.toggle('red', selected && state.currentPlayer === RED);
    button.classList.toggle('yellow', selected && state.currentPlayer === YELLOW);
    button.setAttribute('aria-pressed', String(selected));
    button.disabled = !humanCanMove || !canDrop(state.board, column);
  }

  for (const cell of elements.board.querySelectorAll('.cell')) {
    const selected = Number(cell.dataset.column) === state.selectedColumn
      && !cell.classList.contains('red')
      && !cell.classList.contains('yellow')
      && humanCanMove;
    cell.classList.toggle('selected-column', selected);
  }
}

function renderEvaluation() {
  const visible = isAiGame();
  elements.evaluationPanel.hidden = !visible;
  elements.matchGrid.classList.toggle('single-column', !visible);
  if (!visible) return;

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

  elements.evaluationMeter.value = redPercent;
  elements.evaluationMeter.textContent = label;
  elements.evaluationLabel.textContent = label;
  elements.evaluationPercent.textContent = 'Heuristic position estimate · not a win probability';

  const search = state.liveSearch ?? state.lastSearch;
  if (state.aiThinking && !search) {
    elements.searchInfo.textContent = 'Starting the search in a background worker…';
  } else if (search) {
    const details = [];
    if (search.solver === 'perfect-book') {
      details.push('Perfect book', 'Exact move');
      if (search.bookEntryCount) {
        details.push(`${numberFormatter.format(search.bookEntryCount)} solved openings`);
      }
    } else {
      const seconds = search.elapsedMs / 1_000;
      const rate = search.elapsedMs > 0 ? Math.round(search.nodes / seconds) : 0;
      if (search.solver === 'bitboard') details.push(search.solved ? 'Bitboard solved' : 'Bitboard');
      else if (search.solved) details.push('Solved');
      details.push(
        `Depth ${search.depth}`,
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

  const actor = state.currentPlayer;
  const result = applyAction(state.board, action, actor);
  if (!result) return;

  const roundVersion = state.version;
  state.busy = true;
  state.lastMover = actor;
  state.lastAction = `${playerName(actor)} ${actionLabel(action)}`;
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
  saveJson(SCORES_KEY, state.scores);
  pushSnapshot();
  renderAll();

  if (state.status !== 'playing') {
    await pause(260);
    if (roundVersion === state.version) showResultDialog();
  } else if (isAiGame() && state.currentPlayer === YELLOW) {
    requestAiMove();
  }
}

function fallbackAction() {
  return legalActions(state.board, state.config.chaosMode)
    .find((action) => action.type === ACTION_DROP)
    ?? legalActions(state.board, state.config.chaosMode)[0]
    ?? null;
}

function cancelAiSearch() {
  state.aiRequestId += 1;
  if (state.aiWorker) state.aiWorker.terminate();
  state.aiWorker = null;
  state.aiThinking = false;
  state.liveSearch = null;
}

function requestAiMove() {
  if (!isAiGame() || state.currentPlayer !== YELLOW || state.status !== 'playing' || state.busy) return;

  cancelAiSearch();
  const requestId = state.aiRequestId;
  const roundVersion = state.version;
  const position = {
    board: cloneBoard(state.board),
    currentPlayer: state.currentPlayer,
    connect: state.config.connect,
    chaosMode: state.config.chaosMode,
    repetitionCounts: [...state.repetitionCounts.entries()],
  };
  const options = {
    difficulty: state.config.opponent,
    aiPlayer: YELLOW,
  };

  state.aiThinking = true;
  state.liveSearch = null;
  renderAll();

  const finish = (payload) => {
    if (requestId !== state.aiRequestId || roundVersion !== state.version) return;
    if (state.aiWorker) state.aiWorker.terminate();
    state.aiWorker = null;
    state.aiThinking = false;
    state.liveSearch = null;

    const result = payload?.result;
    const action = result?.action ?? fallbackAction();
    if (result) {
      state.lastSearch = {
        action: result.action ? { ...result.action } : null,
        depth: result.depth ?? 0,
        nodes: result.nodes ?? 0,
        elapsedMs: result.elapsedMs ?? 0,
        score: result.score ?? 0,
        tableHits: result.tableHits ?? 0,
        cutoffs: result.cutoffs ?? 0,
        tableResets: result.tableResets ?? 0,
        tableStores: result.tableStores ?? 0,
        tableCollisions: result.tableCollisions ?? 0,
        solved: Boolean(result.solved),
        solver: result.solver ?? 'general',
        bookPly: result.bookPly ?? null,
        bookMaxPly: result.bookMaxPly ?? null,
        bookEntryCount: result.bookEntryCount ?? null,
        principalVariation: Array.isArray(result.principalVariation)
          ? result.principalVariation.map((action) => ({ ...action }))
          : [],
      };
    }
    renderAll();
    if (action) void performAction(action, 'ai');
  };

  const runFallback = () => {
    setTimeout(() => {
      if (requestId !== state.aiRequestId || roundVersion !== state.version) return;
      try {
        finish({
          result: chooseMove(position, {
            ...options,
            difficulty: 'easy',
            onIteration(progress) {
              state.liveSearch = progress;
              renderStatus();
              renderEvaluation();
            },
          }),
        });
      } catch {
        finish({ result: null });
      }
    }, 0);
  };

  try {
    const worker = new Worker(new URL('./ai-worker.js', import.meta.url), { type: 'module' });
    state.aiWorker = worker;
    worker.addEventListener('message', (event) => {
      if (event.data?.requestId !== requestId) return;
      if (event.data.kind === 'progress') {
        state.liveSearch = event.data.progress ?? null;
        renderStatus();
        renderEvaluation();
        return;
      }
      if (event.data.kind === 'error') {
        worker.terminate();
        if (state.aiWorker === worker) state.aiWorker = null;
        state.liveSearch = null;
        runFallback();
        return;
      }
      finish(event.data);
    });
    worker.addEventListener('error', () => {
      if (state.aiWorker === worker) {
        worker.terminate();
        state.aiWorker = null;
        runFallback();
      }
    }, { once: true });
    worker.postMessage({ requestId, position, options });
  } catch {
    runFallback();
  }
}

function chooseColumn(column) {
  const { cols } = boardDimensions(state.board);
  const nextColumn = Math.max(0, Math.min(cols - 1, column));
  if (nextColumn === state.selectedColumn) return;
  state.selectedColumn = nextColumn;
  renderBoardSelection();
}

function moveSelectedColumn(delta) {
  chooseColumn(state.selectedColumn + delta);
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
    chooseColumn(0);
    event.preventDefault();
  } else if (event.key === 'End') {
    chooseColumn(boardDimensions(state.board).cols - 1);
    event.preventDefault();
  } else if (event.key === 'Enter' || event.key === ' ') {
    dropSelectedColumn();
    event.preventDefault();
  }
}

function handleGlobalKeydown(event) {
  if (isTypingTarget(event.target) || elements.resultDialog.open || elements.rulesDialog.open) return;
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
elements.rowsInput.addEventListener('input', updateConnectLimit);
elements.colsInput.addEventListener('input', updateConnectLimit);
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
elements.playAgainButton.addEventListener('click', () => restartRound());
elements.rulesButton.addEventListener('click', () => {
  if (!elements.rulesDialog.open) elements.rulesDialog.showModal();
});
elements.closeRulesButton.addEventListener('click', () => elements.rulesDialog.close());
elements.rulesDoneButton.addEventListener('click', () => elements.rulesDialog.close());
elements.rulesDialog.addEventListener('click', (event) => {
  if (event.target === elements.rulesDialog) elements.rulesDialog.close();
});
elements.board.addEventListener('keydown', handleBoardKeydown);
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
  const button = event.target.closest?.('.column-button');
  if (button) chooseColumn(Number(button.dataset.column));
});
elements.columnControls.addEventListener('click', (event) => {
  const button = event.target.closest?.('.column-button');
  if (!button) return;
  chooseColumn(Number(button.dataset.column));
  dropSelectedColumn();
});
document.addEventListener('keydown', handleGlobalKeydown);

startRound(state.config, { collapseSettings: compactViewport.matches });
