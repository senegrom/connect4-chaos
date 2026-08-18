#!/usr/bin/env python3
"""Focused fail-closed tests for perfect-chaos-main-supervisor.py."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("perfect-chaos-main-supervisor.py")
SPEC = importlib.util.spec_from_file_location("perfect_chaos_main_supervisor", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def state(role: str, existing: int, cumulative: int) -> dict[str, object]:
    return {
        "role": role,
        "sourceRun": 12345 if role == "red" else 23456,
        "sourceSha": "a" * 40 if role == "red" else "b" * 40,
        "sourceArtifact": f"perfect-chaos-{role}-18-round",
        "existingRejections": existing,
        "cumulativeRejections": cumulative,
        "prepareShards": 64,
        "prepareWorkers": 4,
        "shardCount": 256,
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", newline="\n")


def expect_failure(function, pattern: str) -> None:
    try:
        function()
    except RuntimeError as error:
        if pattern not in str(error):
            raise AssertionError(f"Expected {pattern!r} in {error!r}") from error
    else:
        raise AssertionError(f"Expected RuntimeError containing {pattern!r}")


def two_role_root() -> tuple[tempfile.TemporaryDirectory[str], Path]:
    directory = tempfile.TemporaryDirectory()
    root = Path(directory.name)
    write_json(root / MODULE.STATE_ROOT / "red.json", state("red", 0, 100))
    write_json(root / MODULE.STATE_ROOT / "yellow.json", state("yellow", 0, 200))
    return directory, root


def test_oldest_unresolved_role_is_selected() -> None:
    directory, root = two_role_root()
    with directory:
        result = MODULE.inspect_states(root, commit_times={"red": 20, "yellow": 10})
        assert result["selected"]["role"] == "yellow"
        assert result["activeRoles"] == []
        assert result["states"]["red"]["cumulativeRejections"] == 100
        assert result["states"]["yellow"]["cumulativeRejections"] == 200
        assert result["states"]["red"]["active"] is False
        assert result["states"]["yellow"]["active"] is False


def test_ties_are_deterministic() -> None:
    directory, root = two_role_root()
    with directory:
        result = MODULE.inspect_states(root, commit_times={"red": 10, "yellow": 10})
        assert result["selected"]["role"] == "red"


def test_active_oldest_role_does_not_block_the_other_role() -> None:
    directory, root = two_role_root()
    with directory:
        result = MODULE.inspect_states(
            root,
            commit_times={"red": 5, "yellow": 10},
            active_roles=["red"],
        )
        assert result["activeRoles"] == ["red"]
        assert result["states"]["red"]["active"] is True
        assert result["states"]["yellow"]["active"] is False
        assert result["selected"]["role"] == "yellow"


def test_all_unresolved_roles_active_selects_nothing() -> None:
    directory, root = two_role_root()
    with directory:
        result = MODULE.inspect_states(
            root,
            commit_times={"red": 5, "yellow": 10},
            active_roles=["yellow", "red", "red"],
        )
        assert result["activeRoles"] == ["red", "yellow"]
        assert result["selected"] is None
        assert all(record["active"] for record in result["states"].values())


def test_unknown_active_role_fails_closed() -> None:
    expect_failure(
        lambda: MODULE.normalize_active_roles(["green"]),
        "Unknown active Perfect Chaos roles",
    )


def test_verified_closure_candidate_is_not_relaunched() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_json(root / MODULE.STATE_ROOT / "red.json", state("red", 50, 100))
        write_json(root / MODULE.STATE_ROOT / "yellow.json", state("yellow", 100, 200))
        write_json(
            root / MODULE.STATE_ROOT / "closure-candidates" / "red-100.json",
            {
                "format": "connect4-chaos-auto-advance-decision-v1",
                "role": "red",
                "fromPieces": 16,
                "targetPieces": 18,
                "closedCandidate": True,
                "newRejectedRoots": 0,
                "cumulativeRejectedRoots": 100,
            },
        )
        result = MODULE.inspect_states(
            root,
            commit_times={"red": 1, "yellow": 2},
            active_roles=["red"],
        )
        assert result["states"]["red"]["closedCandidate"] is True
        assert result["states"]["red"]["active"] is True
        assert result["selected"]["role"] == "yellow"


def test_malformed_state_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        invalid = state("red", 100, 100)
        write_json(root / MODULE.STATE_ROOT / "red.json", invalid)
        expect_failure(
            lambda: MODULE.inspect_states(root, commit_times={"red": 1}),
            "must exceed existingRejections",
        )


def test_malformed_candidate_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_json(root / MODULE.STATE_ROOT / "red.json", state("red", 0, 100))
        write_json(
            root / MODULE.STATE_ROOT / "closure-candidates" / "red-100.json",
            {
                "format": "connect4-chaos-auto-advance-decision-v1",
                "role": "red",
                "fromPieces": 16,
                "targetPieces": 18,
                "closedCandidate": False,
                "newRejectedRoots": 1,
                "cumulativeRejectedRoots": 100,
            },
        )
        expect_failure(
            lambda: MODULE.inspect_states(root, commit_times={"red": 1}),
            "closedCandidate must be True",
        )


def main() -> None:
    tests = [
        test_oldest_unresolved_role_is_selected,
        test_ties_are_deterministic,
        test_active_oldest_role_does_not_block_the_other_role,
        test_all_unresolved_roles_active_selects_nothing,
        test_unknown_active_role_fails_closed,
        test_verified_closure_candidate_is_not_relaunched,
        test_malformed_state_fails_closed,
        test_malformed_candidate_fails_closed,
    ]
    for test in tests:
        test()
    print({"tests": len(tests), "status": "pass"})


if __name__ == "__main__":
    main()
