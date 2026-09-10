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
// A single position leaves the GPU almost idle: measured in the browser, a
// batch of eight costs about as much as one position (18.1 ms for one,
// 20.2 ms for eight), so the search collects that many leaves per network
// call. Each edge on a collected path carries a virtual loss until its
// result arrives, which keeps the batch from being eight copies of the same
// line - the standard cost of parallel PUCT, and cheap next to eight
// separate calls.
export const SEARCH_BATCH = 8;
const VIRTUAL_LOSS = 1;

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
    // Edges whose child is being evaluated in the current batch. Selection
    // must not queue the same leaf twice, and the edge cannot be descended
    // through until its node exists.
    this.pending = new Uint8Array(actions.length);
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
 * Runs `simulations` from `position` and returns visits, mean action values
 * and discovered terminal outcomes, all values for the root player.
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
  // `shouldStop(completedSimulations)` lets the page reduce the budget or stop
  // work for a move it no longer wants (undone, restarted).
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
  // `evaluateMany(items)` evaluates a whole batch in one network call and
  // resolves to one output per item, in order. Without it the search still
  // works, one leaf at a time, which is what the tests and any older backend
  // provide.
  const evaluateMany = options.evaluateMany ?? null;
  // A backend can change during this search; read its current batch limit
  // before collecting more leaves after a GPU-to-CPU fallback.
  const batchSize = () => {
    const requested = typeof options.batchSize === 'function' ? options.batchSize() : options.batchSize;
    return Math.max(1, requested ?? (evaluateMany ? SEARCH_BATCH : 1));
  };
  const evaluateLeaves = async (items) => {
    throwIfAborted(signal);
    const requests = items.map(({ board, mover, actions, repeated }) =>
      ({ board, mover, actions, connect, chaosMode, repeated }));
    const outputs = evaluateMany
      ? await evaluateMany(requests)
      : await Promise.all(requests.map((request) => evaluate(
        request.board, request.mover, request.actions, connect, chaosMode, request.repeated)));
    throwIfAborted(signal);
    if (!Array.isArray(outputs) || outputs.length !== items.length) {
      throw new Error('Neural evaluator returned the wrong number of outputs.');
    }
    return outputs;
  };
  const empty = () => ({ actions: [], visits: [], actionValues: [], terminalValues: [], policy: [], value: 0,
    completedSimulations: 0, evaluations });
  if (history.get(rootKey) >= 3) return empty();
  const root = await expand(position.board, position.currentPlayer, connect, chaosMode, evaluateNode,
    Math.max(0, history.get(rootKey) - 1));
  if (!root || root.actions.length === 0) return empty();

  // Descends to one leaf, marking the edges it passes with a virtual loss so
  // the next descent in the same batch prefers a different branch. Returns
  // the value it found, a leaf to evaluate, or `blocked` when it reached an
  // edge already queued in this batch.
  const descend = () => {
    const counts = new Map(history);
    const path = [];
    let node = root;
    for (let depth = 0; depth < 64; depth += 1) {
      const index = node.select();
      path.push([node, index]);
      node.visits[index] += VIRTUAL_LOSS;
      node.valueSum[index] -= VIRTUAL_LOSS;
      if (node.terminal[index] !== undefined && node.terminal[index] !== null) {
        return { path, value: node.terminal[index] };
      }
      if (node.pending[index]) return { path, blocked: true };
      const child = node.children[index];
      if (child) {
        counts.set(child.key, (counts.get(child.key) ?? 0) + 1);
        node = child;
        continue;
      }
      const outcome = step(node.board, connect, chaosMode, node.mover, node.actions[index]);
      if (!outcome) return { path, value: 0 };            // not actually playable
      if (outcome.terminal !== null) {
        node.terminal[index] = outcome.terminal;
        return { path, value: outcome.terminal };
      }
      const nextPlayer = otherPlayer(node.mover);
      const key = positionKey(outcome.board, nextPlayer, connect, chaosMode);
      const repetitions = (counts.get(key) ?? 0) + 1;
      // Nodes are not shared across paths: this edge always has the same
      // history, so a repetition draw can be cached just like a board-full draw.
      if (repetitions >= 3) {
        node.terminal[index] = 0;
        return { path, value: 0 };
      }
      const actions = legalActions(outcome.board, chaosMode);
      if (actions.length === 0) return { path, value: 0 };
      node.pending[index] = 1;
      return { path, leaf: { owner: node, index, key, actions, repeated: repetitions - 1,
                             board: outcome.board, mover: nextPlayer } };
    }
    // A depth cutoff is a leaf estimate, not an invented terminal draw.
    return { path, value: -node.value };
  };

  const release = (path) => {
    for (const [owner, index] of path) {
      owner.visits[index] -= VIRTUAL_LOSS;
      owner.valueSum[index] += VIRTUAL_LOSS;
    }
  };

  const backpropagate = (path, value) => {
    for (let depth = path.length - 1; depth >= 0; depth -= 1) {
      const [owner, index] = path[depth];
      owner.visits[index] -= VIRTUAL_LOSS;
      owner.valueSum[index] += VIRTUAL_LOSS;
      const sign = (path.length - 1 - depth) % 2 === 0 ? 1 : -1;
      owner.visits[index] += 1;
      owner.valueSum[index] += sign * value;
    }
  };

  let completed = 0;
  while (completed < simulations) {
    throwIfAborted(signal);
    if (completed > 0 && shouldStop(completed)) break;
    const target = Math.min(batchSize(), simulations - completed);
    const settled = [];                       // [path, value] pairs ready to back up
    const leaves = [];
    const blocked = [];
    // Every descent either settles, produces a leaf, or is blocked; the
    // attempt cap stops a tree that is entirely terminal or entirely queued
    // from spinning.
    for (let attempt = 0; settled.length + leaves.length < target
                          && attempt < target * 4; attempt += 1) {
      const found = descend();
      if (found.blocked) blocked.push(found.path);
      else if (found.leaf) leaves.push(found);
      else settled.push(found);
    }
    // Awaiting the network is only a microtask on a synchronous backend,
    // which never lets a timer run. Yield a macrotask once per round - the
    // cadence the search had per eight simulations before it batched - so
    // stop and input events still interrupt it.
    if (completed > 0) await new Promise((resolve) => setTimeout(resolve, 0));
    if (leaves.length > 0) {
      evaluations += leaves.length;
      const outputs = await evaluateLeaves(leaves.map((entry) => entry.leaf));
      throwIfAborted(signal);
      for (let i = 0; i < leaves.length; i += 1) {
        const { owner, index, key, actions, board, mover } = leaves[i].leaf;
        owner.pending[index] = 0;
        const next = makeNode(board, mover, actions, outputs[i]);
        next.key = key;
        owner.children[index] = next;
        // The child's value is for its own mover, so this edge sees its negation.
        settled.push({ path: leaves[i].path, value: -next.value });
      }
    }
    for (const path of blocked) release(path);
    for (const { path, value } of settled) backpropagate(path, value);
    if (settled.length === 0) break;         // nothing left that can be explored
    completed += settled.length;
    if (onProgress) onProgress(Math.min(completed, simulations), simulations);
  }

  const visits = Array.from(root.visits);
  const total = visits.reduce((sum, count) => sum + count, 0);
  let valueSum = 0;
  for (let i = 0; i < visits.length; i += 1) valueSum += root.valueSum[i];
  return {
    actions: root.actions,
    visits,
    actionValues: visits.map((count, index) => (count > 0 ? root.valueSum[index] / count : null)),
    // Keep rule-confirmed outcomes separate from even a confident estimate.
    terminalValues: root.terminal.map((value) => value ?? null),
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

/** Builds a node from one network output. Separate from the evaluation so a
 * whole batch of leaves can be evaluated in a single call. */
function makeNode(board, mover, actions, output) {
  validateOutput(output, actions);
  const { policy, value, q } = output;
  const priors = softmaxOverLegal(policy, actions);
  const untried = actions.map((action) => {
    const at = actionIndex(action) * 3;
    return expectedOutcome([q[at], q[at + 1], q[at + 2]]);
  });
  return new Node(board, mover, priors, actions, untried, expectedOutcome(value));
}

async function expand(board, mover, connect, chaosMode, evaluate, repeated = 0) {
  const actions = legalActions(board, chaosMode);
  if (actions.length === 0) return null;
  const output = await evaluate(board, mover, actions, connect, chaosMode, repeated);
  return makeNode(board, mover, actions, output);
}

/** Take a discovered immediate win; otherwise prefer visits, then mean value. */
export function bestAction(result) {
  let best = 0;
  for (let i = 1; i < result.visits.length; i += 1) {
    const wins = result.terminalValues?.[i] === 1;
    const bestWins = result.terminalValues?.[best] === 1;
    if (wins !== bestWins) {
      if (wins) best = i;
      continue;
    }
    if (result.visits[i] > result.visits[best]
        || (result.visits[i] === result.visits[best]
          && (result.actionValues?.[i] ?? -Infinity) > (result.actionValues?.[best] ?? -Infinity))) {
      best = i;
    }
  }
  return result.actions[best] ?? null;
}
