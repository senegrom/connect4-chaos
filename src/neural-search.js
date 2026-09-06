// PUCT search for the browser, matching neural/gpu_mcts.py.
//
// The network alone answers from the current position; this looks ahead,
// and the lookahead is where most of the strength is. On solved boards the
// policy head alone misplays about 3.5% of chaos positions while the same
// network with 128 simulations misplays 0.5%.
//
// Values are always for the player to move at that node, so an edge's value
// is the negation of the value of the position it leads to. Untried actions
// start from the network's per-action head rather than from nothing, which
// is worth more than the simulation it would take to find out.

import {
  ACTION_DROP, ACTION_FLIP, ACTION_ROTATE_CW, ACTION_ROTATE_CCW,
  applyAction, legalActions, otherPlayer, positionKey, resolveActionOutcome,
} from './engine.js';
import { FLIP, ROTATE_CW, ROTATE_CCW } from './neural-planes.js';
import { throwIfAborted } from './async-control.js';

const C_PUCT = 1.5;
const OUTCOME_SCORE = [-1, 0, 1];       // loss, draw, win

/** The network's action index for an engine action. */
export function actionIndex(action) {
  if (action.type === ACTION_DROP) return action.column;
  if (action.type === ACTION_FLIP) return FLIP;
  if (action.type === ACTION_ROTATE_CW) return ROTATE_CW;
  if (action.type === ACTION_ROTATE_CCW) return ROTATE_CCW;
  throw new RangeError(`Unknown action type: ${action.type}`);
}

function softmaxOverLegal(logits, actions) {
  let best = -Infinity;
  for (const action of actions) best = Math.max(best, logits[actionIndex(action)]);
  let total = 0;
  const weights = actions.map((action) => {
    const weight = Math.exp(logits[actionIndex(action)] - best);
    total += weight;
    return weight;
  });
  return weights.map((weight) => weight / (total || 1));
}

function expectedOutcome(distribution) {
  let total = 0;
  let sum = 0;
  const maximum = Math.max(...distribution);
  for (let outcome = 0; outcome < 3; outcome += 1) {
    const weight = Math.exp(distribution[outcome] - maximum);
    total += weight;
    sum += weight * OUTCOME_SCORE[outcome];
  }
  return total > 0 ? sum / total : 0;
}

/** Applies one action, returning the position it leads to and its outcome. */
function step(board, connect, chaosMode, mover, action) {
  // applyAction returns a new board and reports where a drop landed; it
  // does not modify the one it is given.
  const applied = applyAction(board, action, mover);
  if (!applied) return null;                       // the column was full
  const lastDrop = action.type === ACTION_DROP
    ? { row: applied.row, column: applied.column }
    : null;
  const outcome = resolveActionOutcome(applied.board, connect, mover, action.type, lastDrop);
  let terminal = null;
  if (outcome.status === 'won') terminal = outcome.winner === mover ? 1 : -1;
  else if (outcome.status === 'draw') terminal = 0;
  return { board: applied.board, terminal, chaosMode };
}

class Node {
  constructor(board, mover, priors, actions, values, value) {
    this.board = board;
    this.mover = mover;
    this.actions = actions;
    this.prior = priors;
    this.untried = values;              // per-action estimate from the network
    this.visits = new Float64Array(actions.length);
    this.valueSum = new Float64Array(actions.length);
    this.children = new Array(actions.length).fill(null);
    this.terminal = new Array(actions.length).fill(undefined);
    this.value = value;
  }

  select() {
    let total = 0;
    for (let i = 0; i < this.visits.length; i += 1) total += this.visits[i];
    const explore = Math.sqrt(Math.max(1, total));
    let best = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < this.actions.length; i += 1) {
      const seen = this.visits[i];
      const q = seen > 0 ? this.valueSum[i] / seen : this.untried[i];
      const score = q + C_PUCT * this.prior[i] * explore / (1 + seen);
      if (score > bestScore) {
        best = i;
        bestScore = score;
      }
    }
    return best;
  }
}

/**
 * Runs `simulations` from `position` and returns the visit counts.
 *
 * `evaluate(board, mover, actions, connect, chaosMode, repeated)` resolves to
 * `{ policy: Float32Array(13), value: Float32Array(3), q: Float32Array(39) }`
 * with raw logits, exactly as the exported network produces them.
 */
export async function searchPosition(position, evaluate, options = {}) {
  const simulations = options.simulations ?? 128;
  if (!Number.isInteger(simulations) || simulations < 1) {
    throw new RangeError('simulations must be a positive integer');
  }
  const signal = options.signal;
  throwIfAborted(signal);
  // `shouldStop()` ends the search early, so a move the page no longer
  // wants (undone, restarted) stops burning the evaluation budget.
  const shouldStop = options.shouldStop ?? (() => false);
  const onProgress = options.onProgress ?? null;
  const { connect, chaosMode } = position;
  const rootKey = positionKey(position.board, position.currentPlayer, connect, chaosMode);
  const history = new Map(position.repetitionCounts ?? []);
  if (!history.has(rootKey)) history.set(rootKey, 1 + (options.repeated ?? 0));
  let evaluations = 0;
  const evaluateNode = async (...args) => {
    throwIfAborted(signal);
    evaluations += 1;
    const result = await evaluate(...args);
    throwIfAborted(signal);
    return result;
  };
  const empty = () => ({ actions: [], visits: [], policy: [], value: 0,
    completedSimulations: 0, evaluations });
  if (history.get(rootKey) >= 3) return empty();
  const root = await expand(position.board, position.currentPlayer, connect, chaosMode, evaluateNode,
    Math.max(0, history.get(rootKey) - 1));
  if (!root || root.actions.length === 0) return empty();

  for (let simulation = 0; simulation < simulations; simulation += 1) {
    // Cached terminal edges may not await an evaluator. Yield explicitly so
    // stop/input events also run for those batches and synchronous test backends.
    if (simulation > 0 && simulation % 8 === 0) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    throwIfAborted(signal);
    if (simulation > 0 && shouldStop()) break;
    const counts = new Map(history);
    const path = [];
    let node = root;
    let value = null;
    for (let depth = 0; depth < 64; depth += 1) {
      const index = node.select();
      path.push([node, index]);
      if (node.terminal[index] !== undefined && node.terminal[index] !== null) {
        value = node.terminal[index];
        break;
      }
      const child = node.children[index];
      if (child) {
        counts.set(child.key, (counts.get(child.key) ?? 0) + 1);
        node = child;
        continue;
      }
      const outcome = step(node.board, connect, chaosMode, node.mover, node.actions[index]);
      if (!outcome) {                     // not actually playable
        node.terminal[index] = 0;
        value = 0;
        break;
      }
      if (outcome.terminal !== null) {
        node.terminal[index] = outcome.terminal;
        value = outcome.terminal;
        break;
      }
      const nextPlayer = otherPlayer(node.mover);
      const key = positionKey(outcome.board, nextPlayer, connect, chaosMode);
      const repetitions = (counts.get(key) ?? 0) + 1;
      // Nodes are not shared across paths: this edge always has the same
      // history, so a repetition draw can be cached just like a board-full draw.
      if (repetitions >= 3) {
        node.terminal[index] = 0;
        value = 0;
        break;
      }
      counts.set(key, repetitions);
      const next = await expand(outcome.board, nextPlayer, connect, chaosMode,
        evaluateNode, repetitions - 1);
      if (next) next.key = key;
      node.children[index] = next;
      // The child's value is for its own mover, so this edge sees its negation.
      value = next ? -next.value : 0;
      break;
    }
    // A depth cutoff is a leaf estimate, not an invented terminal draw.
    if (value === null) value = -node.value;
    for (let depth = path.length - 1; depth >= 0; depth -= 1) {
      const [owner, index] = path[depth];
      const sign = (path.length - 1 - depth) % 2 === 0 ? 1 : -1;
      owner.visits[index] += 1;
      owner.valueSum[index] += sign * value;
    }
    if (onProgress && (simulation % 8 === 7 || simulation === simulations - 1)) {
      onProgress(simulation + 1, simulations);
    }
  }

  const visits = Array.from(root.visits);
  const total = visits.reduce((sum, count) => sum + count, 0);
  let valueSum = 0;
  for (let i = 0; i < visits.length; i += 1) valueSum += root.valueSum[i];
  return {
    actions: root.actions,
    visits,
    completedSimulations: total,
    evaluations,
    policy: visits.map((count) => (total > 0 ? count / total : 1 / visits.length)),
    value: total > 0 ? valueSum / total : root.value,
  };
}

/** Corrupt/incompatible inference must enter recovery, not silently pick column 1.
 * Negative infinity is legitimate masking, including one-hot WDL/Q logits. */
function validateOutput(output, actions) {
  for (const [head, length] of [['policy', 13], ['value', 3], ['q', 39]]) {
    const values = output?.[head];
    if ((!Array.isArray(values) && !ArrayBuffer.isView(values)) || values.length !== length
        || !Array.from(values).every((value) => Number.isFinite(value) || value === -Infinity)) {
      throw new Error(`Neural evaluator returned invalid ${head} output logits.`);
    }
  }
  const { policy, value, q } = output;
  if (!actions.some((action) => Number.isFinite(policy[actionIndex(action)]))) {
    throw new Error('Neural evaluator masked every legal policy output.');
  }
  if (!Array.from(value).some(Number.isFinite)) {
    throw new Error('Neural evaluator returned all-masked value logits.');
  }
  for (const action of actions) {
    const index = actionIndex(action) * 3;
    if (![q[index], q[index + 1], q[index + 2]].some(Number.isFinite)) {
      throw new Error('Neural evaluator returned all-masked Q logits for a legal action.');
    }
  }
}

async function expand(board, mover, connect, chaosMode, evaluate, repeated = 0) {
  const actions = legalActions(board, chaosMode);
  if (actions.length === 0) return null;
  const output = await evaluate(board, mover, actions, connect, chaosMode, repeated);
  validateOutput(output, actions);
  const { policy, value, q } = output;
  const priors = softmaxOverLegal(policy, actions);
  const untried = actions.map((action) => {
    const at = actionIndex(action) * 3;
    return expectedOutcome([q[at], q[at + 1], q[at + 2]]);
  });
  return new Node(board, mover, priors, actions, untried, expectedOutcome(value));
}

/** The most-visited action, which is what the search recommends. */
export function bestAction(result) {
  let best = 0;
  for (let i = 1; i < result.visits.length; i += 1) {
    if (result.visits[i] > result.visits[best]) best = i;
  }
  return result.actions[best] ?? null;
}
