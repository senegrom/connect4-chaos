"""Writes a fixture the browser encoder is checked against.

The engine numbers rows from the top of the board and the network from the
bottom, so the two encodings differ by a flip. That is exactly the kind of
mistake that produces a player which is subtly, silently wrong, so the
planes are generated here from the Python side and compared in a test.

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


def grid_from_state(state, rows, columns):
    """The position as the engine holds it: row 0 at the top, 1 = mover."""
    grid = [[0] * columns for _ in range(rows)]
    for column in range(columns):
        for row in range(rows):
            bit = 1 << (column * state.stride + row)
            value = 1 if state.mover & bit else (2 if state.opponent & bit else 0)
            grid[rows - 1 - row][column] = value          # bottom row last
    return grid


def main() -> None:
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/neural-planes.json")
    rng = random.Random(20260903)
    cases = []
    for rows, columns, connect, chaos in CONFIGS:
        state = empty_state(rows, columns)
        for ply in range(rng.randint(2, 9)):
            edges = successors(state, connect, chaos=chaos)
            playable = [edge for edge in edges if edge.child is not None]
            if not playable:
                break
            state = rng.choice(playable).child
        cases.append({
            "rows": rows,
            "columns": columns,
            "connect": connect,
            "chaos": chaos,
            "grid": grid_from_state(state, rows, columns),
            "planes": to_planes(state, connect, chaos),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    print(f"wrote {out_path} with {len(cases)} positions")


if __name__ == "__main__":
    main()
