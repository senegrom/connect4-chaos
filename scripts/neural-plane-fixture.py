"""Writes a fixture the browser encoder is checked against.

The engine numbers rows from the top of the board and the network from the
bottom, so the two encodings differ by a flip. That is exactly the kind of
mistake that produces a player which is subtly, silently wrong, so the
planes are generated here from the Python side and compared in a test.

Besides one random position per board, the fixture holds positions with the
repetition planes set, and a non-square Chaos board after a rotation, which
turns its rows into columns.

Usage: python scripts/neural-plane-fixture.py tests/fixtures/neural-planes.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural.chaos_game import empty_state, successors, to_planes   # noqa: E402

CONFIGS = [(4, 4, 3, True), (6, 7, 4, False), (5, 6, 4, True), (4, 5, 4, False)]


def grid_from_state(state):
    """The position as the engine holds it: row 0 at the top, 1 = mover."""
    grid = [[0] * state.columns for _ in range(state.rows)]
    for column in range(state.columns):
        for row in range(state.rows):
            bit = 1 << (column * state.stride + row)
            value = 1 if state.mover & bit else (2 if state.opponent & bit else 0)
            grid[state.rows - 1 - row][column] = value          # bottom row last
    return grid


def case(state, connect, chaos, repeated=0):
    """One fixture entry. The dimensions are the state's: a rotation
    transposes a board that is not square, so the configured ones would
    describe the wrong board."""
    return {
        "rows": state.rows,
        "columns": state.columns,
        "connect": connect,
        "chaos": chaos,
        "repeated": repeated,
        "grid": grid_from_state(state),
        "planes": to_planes(state, connect, chaos, repeated >= 1, repeated >= 2),
    }


def walk(rng, state, connect, chaos, plies):
    for _ply in range(plies):
        playable = [edge for edge in successors(state, connect, chaos=chaos) if edge.child is not None]
        if not playable:
            break
        state = rng.choice(playable).child
    return state


def main() -> None:
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/neural-planes.json")
    rng = random.Random(20260903)
    cases = []
    states = []
    for rows, columns, connect, chaos in CONFIGS:
        state = walk(rng, empty_state(rows, columns), connect, chaos, rng.randint(2, 9))
        states.append(state)
        cases.append(case(state, connect, chaos))
    # A position seen once and one seen twice before.
    cases.append(case(states[1], CONFIGS[1][2], CONFIGS[1][3], repeated=1))
    cases.append(case(states[0], CONFIGS[0][2], CONFIGS[0][3], repeated=2))
    # Five by six turned on its side: six rows of five, pieces and all.
    dropped = walk(rng, empty_state(5, 6), 4, False, 5)
    rotated = next(edge.child for edge in successors(dropped, 4, chaos=True)
                   if edge.action == "rotate_cw" and edge.child is not None)
    assert (rotated.rows, rotated.columns) == (6, 5), (rotated.rows, rotated.columns)
    cases.append(case(rotated, 4, True))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    print(f"wrote {out_path} with {len(cases)} positions")


if __name__ == "__main__":
    main()
