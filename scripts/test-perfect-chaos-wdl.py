#!/usr/bin/env python3
"""Regression tests for exact Perfect Chaos W/D/L propagation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

FORMAT = "connect4-chaos-closed-wdl-graph-v1"
OBJECTIVE = "maximize-win-then-draw-then-loss"


def graph(nodes: list[dict], roots: list[int] | None = None) -> dict:
    return {
        "format": FORMAT,
        "objective": OBJECTIVE,
        "role": "red",
        "roots": roots or [0],
        "nodes": nodes,
    }


def solve(script: Path, directory: Path, name: str, value: dict) -> dict:
    source = directory / f"{name}.json"
    output = directory / f"{name}.solution.json"
    source.write_text(json.dumps(value, indent=2) + "\n")
    result = subprocess.run(
        [sys.executable, str(script), "solve", "--input", str(source), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"W/D/L solver rejected {name}:\n{result.stderr}")
    stdout = json.loads(result.stdout)
    saved = json.loads(output.read_text())
    if stdout != saved:
        raise AssertionError(f"W/D/L solver output mismatch for {name}")
    return saved


def require_failure(script: Path, directory: Path, name: str, value: dict, marker: str) -> None:
    source = directory / f"{name}.json"
    source.write_text(json.dumps(value, indent=2) + "\n")
    result = subprocess.run(
        [sys.executable, str(script), "solve", "--input", str(source)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        raise AssertionError(f"W/D/L solver accepted invalid {name}: {result.stdout}")
    if marker not in result.stderr:
        raise AssertionError(f"missing {marker!r} for {name}: {result.stderr}")


def policy_edge(solution: dict, node: int) -> int:
    return next(record["edge"] for record in solution["policy"] if record["node"] == node)


def main() -> int:
    script = Path(__file__).with_name("perfect-chaos-wdl.py")
    with tempfile.TemporaryDirectory(prefix="perfect-chaos-wdl-") as temporary:
        root = Path(temporary)

        # The first move is safe but only draws; the second wins. A safety-only
        # selector may choose edge 0, while the exact objective must choose 1.
        solution = solve(script, root, "safe-draw-versus-win", graph([
            {
                "aiTurn": True,
                "edges": [
                    {"terminal": "draw", "action": {"type": "flip"}},
                    {"terminal": "win", "action": {"type": "drop", "column": 3}},
                ],
            },
        ]))
        if solution["rootValues"] != ["win"] or policy_edge(solution, 0) != 1:
            raise AssertionError("exact W/D/L failed to prefer a win over a safe draw")

        solution = solve(script, root, "opponent-forces-loss", graph([
            {
                "aiTurn": False,
                "edges": [
                    {"terminal": "draw", "action": {"type": "flip"}},
                    {"terminal": "loss", "action": {"type": "drop", "column": 0}},
                ],
            },
        ]))
        if solution["rootValues"] != ["loss"]:
            raise AssertionError("opponent minimisation did not choose an available AI loss")

        solution = solve(script, root, "closed-cycle-draw", graph([
            {
                "aiTurn": True,
                "edges": [
                    {"next": 1, "action": {"type": "flip"}},
                    {"terminal": "loss", "action": {"type": "drop", "column": 0}},
                ],
            },
            {
                "aiTurn": False,
                "edges": [
                    {"next": 0, "action": {"type": "flip"}},
                ],
            },
        ]))
        if solution["rootValues"] != ["draw"] or policy_edge(solution, 0) != 0:
            raise AssertionError("closed repetition cycle was not preserved as the optimal draw")
        if not solution["drawRegionClosedVerified"]:
            raise AssertionError("draw-region closure was not recorded")

        solution = solve(script, root, "ranked-forced-win", graph([
            {"aiTurn": True, "edges": [{"next": 1, "action": {"type": "drop", "column": 3}}]},
            {
                "aiTurn": False,
                "edges": [
                    {"terminal": "win", "action": {"type": "flip"}},
                    {"next": 2, "action": {"type": "rotateCW"}},
                ],
            },
            {"aiTurn": True, "edges": [{"terminal": "win", "action": {"type": "drop", "column": 2}}]},
        ]))
        if solution["rootValues"] != ["win"] or solution["ranks"][0] != 3:
            raise AssertionError("ranked forced-win propagation is incorrect")

        solution = solve(script, root, "exact-oracle-handoff", graph([
            {
                "aiTurn": True,
                "edges": [
                    {"oracle": "loss", "action": {"type": "drop", "column": 0}},
                    {"oracle": "draw", "action": {"type": "rotateCCW"}},
                ],
            },
        ]))
        if solution["rootValues"] != ["draw"] or policy_edge(solution, 0) != 1:
            raise AssertionError("exact frontier handoff did not preserve optimal draw selection")

        require_failure(
            script,
            root,
            "unresolved-frontier",
            graph([{"aiTurn": True, "edges": [{"frontier": 4, "action": {"type": "flip"}}]}]),
            "exactly one of next, terminal, or oracle",
        )
        require_failure(
            script,
            root,
            "bad-objective",
            {**graph([{"aiTurn": True, "edges": [{"terminal": "draw", "action": {"type": "flip"}}]}]),
             "objective": "avoid-loss-only"},
            "wrong optimisation objective",
        )
        require_failure(
            script,
            root,
            "missing-action",
            graph([{"aiTurn": True, "edges": [{"terminal": "win"}]}]),
            ".action is required",
        )

    print("perfect-chaos-wdl: all regression cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
