// PUCT search for the browser, matching neural/gpu_mcts.py.
//
// The network alone answers from the current position; this looks ahead,
// and the lookahead is where most of the strength is. On solved boards the
// first network the site shipped (generation 65, 2026-09-03) misplayed
// about 3.5% of chaos positions on its policy head alone and 0.5% with 128
// simulations.
//
// Values are always for the player to move at that node, so an edge's value
// is the negation of the value of the position it leads to. Untried actions
// start from the network's per-action head rather than from nothing, which
// is worth more than the simulation it would take to find out.
//
// A value is a win/draw/loss split collapsed to win minus loss. The search
// keeps the splits as well, only to report them: the page shows how likely
// each result is, which a single number cannot say. They never steer it.

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

// Splits read [win, draw, loss], the other way round from the network's
// logits (OUTCOME_SCORE), because that is the order the page names them in.
const WIN = Float64Array.of(1, 0, 0);
const DRAW = Float64Array.of(0, 1, 0);
const LOSS = Float64Array.of(0, 0, 1);

/** A position the rules decided, as a split for the player to move in it.
 * `terminal` is the result for the player whose move ended the game. */
function decided(terminal) {
  if (terminal > 0) return LOSS;
  return terminal < 0 ? WIN : DRAW;
}

/** Writes the softmax of the three logits at `logits[from]` into `into` at
 * `at`, as [win, draw, loss]. Separate from expectedOutcome, so the values
 * the search runs on stay exactly what they were. */
function writeSplit(logits, from, into, at) {
  const loss = logits[from];
  const draw = logits[from + 1];
  const win = logits[from + 2];
  const maximum = Math.max(loss, draw, win);
  const lossWeight = Math.exp(loss - maximum);
  const drawWeight = Math.exp(draw - maximum);
  const winWeight = Math.exp(win - maximum);
  // validateOutput leaves a finite logit, so the largest weighs exactly 1.
  const total = lossWeight + drawWeight + winWeight;
  into[at] = winWeight / total;
  into[at + 1] = drawWeight / total;
  into[at + 2] = lossWeight / total;
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
    // The splits behind `value` and `untried`, for this node's mover:
    // [win, draw, loss] from the value head, and three per action from the
    // per-action head. makeNode fills them.
    this.outcome = new Float64Array(3);
    this.untriedOutcomes = new Float64Array(actions.length * 3);
    // Only the root sums splits, beside valueSum: nothing else reports them.
    this.outcomeSum = null;
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
 * Beside them come splits, each [win, draw, loss] for the root player:
 * `outcome` for the position as the search sees it, `actionOutcomes` for
 * each action - the per-action head's until a simulation tries it - and
 * `networkOutcome`, the value head's for the position before any search.
 * `actionNetworkOutcomes` holds, for each action, the value head's for the
 * position it leads to, or null where the network never evaluated that
 * position: no simulation tried the action, or the rules decided where it
 * leads (a win, a loss or a draw, by repetition too). For a position already
 * drawn the network is not asked: `outcome` is a certain draw,
 * `actionOutcomes` and `actionNetworkOutcomes` are empty and
 * `networkOutcome` is null. `onProgress(completed, simulations, outcome,
 * networkOutcome)` passes the search's `outcome` so far and the root's
 * `networkOutcome`.
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
  if (!history.has(rootKey)) history.set(rootKey, 1);
  let evaluations = 0;
  const evaluateNode = async (...args) => {
    throwIfAborted(signal);
    evaluations += 1;
    const result = await evaluate(...args);
    throwIfAborted(signal);
    return result;
  };
  // `evaluateMany(items)` evaluates a whole batch in one network call and
  // resolves to one output per item, in order. Without it the search
  // evaluates one leaf at a time, as it does on WebAssembly, where batching
  // measured no faster.
  const evaluateMany = options.evaluateMany ?? null;
  // A backend can change during this search; read its current batch limit
  // before collecting more leaves after a GPU-to-CPU fallback.
  const batchSize = () => {
    const requested = typeof options.batchSize === 'function' ? options.batchSize() : options.batchSize;
    return Math.max(1, requested ?? (evaluateMany ? SEARCH_BATCH : 1));
  };
  const evaluateLeaves = async (items) => {
    throwIfAborted(signal);
    // A batch carries only what the network reads: each leaf's legal actions
    // stay here rather than being cloned into the worker for nothing.
    const outputs = evaluateMany
      ? await evaluateMany(items.map(({ board, mover, repeated }) => ({ board, mover, connect, chaosMode, repeated })))
      : await Promise.all(items.map(({ board, mover, actions, repeated }) => evaluate(
        board, mover, actions, connect, chaosMode, repeated)));
    throwIfAborted(signal);
    if (!Array.isArray(outputs) || outputs.length !== items.length) {
      throw new Error('Neural evaluator returned the wrong number of outputs.');
    }
    return outputs;
  };
  // A position already drawn, by repetition or a full board: the network is
  // not asked, so it has no split of its own to report.
  const empty = () => ({ actions: [], visits: [], actionValues: [], terminalValues: [], policy: [], value: 0,
    completedSimulations: 0, evaluations, outcome: Array.from(DRAW), actionOutcomes: [], networkOutcome: null,
    actionNetworkOutcomes: [] });
  if (history.get(rootKey) >= 3) return empty();
  const root = await expand(position.board, position.currentPlayer, connect, chaosMode, evaluateNode,
    Math.max(0, history.get(rootKey) - 1));
  if (!root || root.actions.length === 0) return empty();
  root.outcomeSum = new Float64Array(root.actions.length * 3);

  // Descends to one leaf, marking the edges it passes with a virtual loss so
  // the next descent in the same batch prefers a different branch. Returns
  // the value it found, a leaf to evaluate, or `blocked` when it reached an
  // edge already queued in this batch. Beside a value comes `split`, the
  // position the path ends in as seen by the player to move there.
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
        return { path, value: node.terminal[index], split: decided(node.terminal[index]) };
      }
      if (node.pending[index]) return { path, blocked: true };
      const child = node.children[index];
      if (child) {
        counts.set(child.key, (counts.get(child.key) ?? 0) + 1);
        node = child;
        continue;
      }
      const outcome = step(node.board, connect, chaosMode, node.mover, node.actions[index]);
      if (!outcome) return { path, value: 0, split: DRAW };   // not actually playable
      if (outcome.terminal !== null) {
        node.terminal[index] = outcome.terminal;
        return { path, value: outcome.terminal, split: decided(outcome.terminal) };
      }
      const nextPlayer = otherPlayer(node.mover);
      const key = positionKey(outcome.board, nextPlayer, connect, chaosMode);
      const repetitions = (counts.get(key) ?? 0) + 1;
      // Nodes are not shared across paths: this edge always has the same
      // history, so a repetition draw can be cached just like a board-full draw.
      if (repetitions >= 3) {
        node.terminal[index] = 0;
        return { path, value: 0, split: DRAW };
      }
      const actions = legalActions(outcome.board, chaosMode);
      if (actions.length === 0) return { path, value: 0, split: DRAW };
      node.pending[index] = 1;
      return { path, leaf: { owner: node, index, key, actions, repeated: repetitions - 1,
                             board: outcome.board, mover: nextPlayer } };
    }
    // A depth cutoff is a leaf estimate, not an invented terminal draw.
    return { path, value: -node.value, split: node.outcome };
  };

  const release = (path) => {
    for (const [owner, index] of path) {
      owner.visits[index] -= VIRTUAL_LOSS;
      owner.valueSum[index] += VIRTUAL_LOSS;
    }
  };

  const backpropagate = (path, value, split) => {
    for (let depth = path.length - 1; depth >= 0; depth -= 1) {
      const [owner, index] = path[depth];
      owner.visits[index] -= VIRTUAL_LOSS;
      owner.valueSum[index] += VIRTUAL_LOSS;
      const sign = (path.length - 1 - depth) % 2 === 0 ? 1 : -1;
      owner.visits[index] += 1;
      owner.valueSum[index] += sign * value;
    }
    // The position the path ends in is path.length plies from the root, and
    // win and loss trade places at every ply, as the value's sign flips. The
    // root's mover moves at the even ones.
    const at = path[0][1] * 3;
    const turned = path.length % 2 === 1;
    root.outcomeSum[at] += turned ? split[2] : split[0];
    root.outcomeSum[at + 1] += split[1];
    root.outcomeSum[at + 2] += turned ? split[0] : split[2];
  };

  // The root's split so far: every simulation's, or the network's own
  // before any. O(actions), as the page asks for it after every batch.
  const rootOutcome = () => {
    let total = 0;
    const outcome = [0, 0, 0];
    for (let i = 0; i < root.actions.length; i += 1) {
      total += root.visits[i];
      for (let result = 0; result < 3; result += 1) outcome[result] += root.outcomeSum[i * 3 + result];
    }
    if (total === 0) return Array.from(root.outcome);
    for (let result = 0; result < 3; result += 1) outcome[result] /= total;
    return outcome;
  };

  let completed = 0;
  while (completed < simulations) {
    throwIfAborted(signal);
    if (completed > 0 && shouldStop(completed)) break;
    const target = Math.min(batchSize(), simulations - completed);
    const settled = [];                       // { path, value, split } entries ready to back up
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
        settled.push({ path: leaves[i].path, value: -next.value, split: next.outcome });
      }
    }
    for (const path of blocked) release(path);
    for (const { path, value, split } of settled) backpropagate(path, value, split);
    if (settled.length === 0) break;         // nothing left that can be explored
    completed += settled.length;
    // The page keeps what it is given, so it gets a copy of the root's value
    // head, never the node's own array: one small array per batch.
    if (onProgress) onProgress(Math.min(completed, simulations), simulations, rootOutcome(), Array.from(root.outcome));
  }

  const visits = Array.from(root.visits);
  const total = visits.reduce((sum, count) => sum + count, 0);
  let valueSum = 0;
  for (let i = 0; i < visits.length; i += 1) valueSum += root.valueSum[i];
  const shares = (from, at, count = 1) => [from[at] / count, from[at + 1] / count, from[at + 2] / count];
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
    outcome: rootOutcome(),
    actionOutcomes: visits.map((count, index) => (count > 0
      ? shares(root.outcomeSum, index * 3, count) : shares(root.untriedOutcomes, index * 3))),
    networkOutcome: Array.from(root.outcome),
    // Only an evaluated position has a child node. Its split is for its own
    // mover, the root player's opponent, so its win is the root's loss.
    actionNetworkOutcomes: root.children.map((child) => (child
      ? [child.outcome[2], child.outcome[1], child.outcome[0]] : null)),
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
  const node = new Node(board, mover, priors, actions, untried, expectedOutcome(value));
  writeSplit(value, 0, node.outcome, 0);
  for (let i = 0; i < actions.length; i += 1) writeSplit(q, actionIndex(actions[i]) * 3, node.untriedOutcomes, i * 3);
  return node;
}

async function expand(board, mover, connect, chaosMode, evaluate, repeated = 0) {
  const actions = legalActions(board, chaosMode);
  if (actions.length === 0) return null;
  const output = await evaluate(board, mover, actions, connect, chaosMode, repeated);
  return makeNode(board, mover, actions, output);
}

/** Take a discovered immediate win; otherwise maximize policy over non-losses.
 * A loss is eligible only when every action is a rule-confirmed loss.
 * Unknown/unvisited moves remain eligible, even with a pessimistic estimate. */
export function bestAction(result) {
  const hasNonLoss = result.actions.some((_, i) => result.terminalValues?.[i] !== -1);
  const { policy } = result;
  let best = -1;
  for (let i = 0; i < result.actions.length; i += 1) {
    if (hasNonLoss && result.terminalValues?.[i] === -1) continue;
    if (best === -1) { best = i; continue; }
    const wins = result.terminalValues?.[i] === 1;
    const bestWins = result.terminalValues?.[best] === 1;
    if (wins !== bestWins) {
      if (wins) best = i;
      continue;
    }
    if (policy[i] > policy[best]
        || (policy[i] === policy[best]
          && (result.actionValues?.[i] ?? -Infinity) > (result.actionValues?.[best] ?? -Infinity))) {
      best = i;
    }
  }
  return result.actions[best] ?? null;
}
