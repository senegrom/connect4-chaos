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
