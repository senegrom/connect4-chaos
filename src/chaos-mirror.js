import { ACTION_DROP, ACTION_ROTATE_CCW, ACTION_ROTATE_CW } from './engine.js';

/** An action as it reads on the board's mirror image: a drop moves to the
 * mirrored column, the two rotations trade places, and a flip stays a flip.
 * Every table stores one of a position and its mirror, so each lookup of a
 * mirrored position maps its action back through this. */
export function mirrorChaosAction(action, columns) {
  if (!action) return null;
  if (action.type === ACTION_DROP) return { type: ACTION_DROP, column: columns - 1 - action.column };
  if (action.type === ACTION_ROTATE_CW) return { type: ACTION_ROTATE_CCW };
  if (action.type === ACTION_ROTATE_CCW) return { type: ACTION_ROTATE_CW };
  return { type: action.type };
}

/** A packed Chaos state - mover and opponent bit masks, rows + 1 bits per
 * column - seen in the mirror: each column's group trades places with its
 * mirror column's. */
export function mirrorPackedState(state) {
  const stride = state.rows + 1;
  const groupMask = (1n << BigInt(stride)) - 1n;
  const flip = (bits) => {
    let mirrored = 0n;
    for (let column = 0; column < state.columns; column += 1) {
      const group = (bits >> BigInt(column * stride)) & groupMask;
      mirrored |= group << BigInt((state.columns - 1 - column) * stride);
    }
    return mirrored;
  };
  return { ...state, mover: flip(state.mover), opponent: flip(state.opponent) };
}

/** The order the Chaos tables keep packed states in, and pick the smaller of
 * a position and its mirror by: shape, then mover bits, then opponent bits. */
export function comparePackedStates(first, second) {
  if (first.rows !== second.rows) return first.rows - second.rows;
  if (first.columns !== second.columns) return first.columns - second.columns;
  if (first.mover !== second.mover) return first.mover < second.mover ? -1 : 1;
  if (first.opponent !== second.opponent) return first.opponent < second.opponent ? -1 : 1;
  return 0;
}
