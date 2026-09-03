// Board -> network input, matching neural/chaos_game.py exactly.
//
// The network sees a fixed 10x10 canvas with the board left- and
// bottom-aligned on it, so one set of weights serves every size. Planes are
// mover-relative: plane 0 always holds the pieces of the player to move.
// Row 0 is the bottom row, as in the engine and the solver.

export const CANVAS = 10;
export const PLANES = 7;
export const ACTIONS = 13;
export const DROP_ACTIONS = 10;
export const FLIP = 10;
export const ROTATE_CW = 11;
export const ROTATE_CCW = 12;

export const ACTION_NAMES = Object.freeze([
  ...Array.from({ length: DROP_ACTIONS }, (_, column) => `drop${column}`),
  'flip', 'rotate_cw', 'rotate_ccw',
]);

/**
 * Writes one position into `out` at `offset`, as PLANES x CANVAS x CANVAS.
 *
 * `cells(row, column)` returns 0 for empty, 1 for the player to move and 2
 * for the opponent. Rows count from the bottom.
 */
export function writePlanes(out, offset, rows, cols, connect, chaosMode, cells,
  repeatedOnce = false, repeatedTwice = false) {
  const area = CANVAS * CANVAS;
  out.fill(0, offset, offset + PLANES * area);
  for (let column = 0; column < cols; column += 1) {
    for (let row = 0; row < rows; row += 1) {
      const cell = cells(row, column);
      const at = row * CANVAS + column;
      if (cell === 1) out[offset + at] = 1;
      else if (cell === 2) out[offset + area + at] = 1;
      out[offset + 2 * area + at] = 1;                    // on-board mask
    }
  }
  const constants = [connect / 10, chaosMode ? 1 : 0, repeatedOnce ? 1 : 0, repeatedTwice ? 1 : 0];
  for (let plane = 0; plane < constants.length; plane += 1) {
    const value = constants[plane];
    if (value === 0) continue;
    const start = offset + (3 + plane) * area;
    out.fill(value, start, start + area);
  }
}

/** A freshly allocated input tensor for `count` positions. */
export function planeBuffer(count) {
  return new Float32Array(count * PLANES * CANVAS * CANVAS);
}
