#!/usr/bin/env python3
"""Focused fail-closed tests for perfect-chaos-main-supervisor.py."""

from __future__ import annotations

import hashlib
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


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recovery_audit(
    role: str,
    action: str,
    expected_digest: str,
    *,
    handled: bool = True,
) -> dict[str, object]:
    return {
        "format": MODULE.RECOVERY_FORMAT,
        "action": action,
        "handled": handled,
        "runId": 34567 if role == "red" else 45678,
        "runAttempt": 3,
        "runSha": "c" * 40 if role == "red" else "d" * 40,
        "role": role,
        "statePath": (MODULE.STATE_ROOT / f"{role}.json").as_posix(),
        "expectedStateSha256": expected_digest,
        "oldProfile": {
            "prepareShards": 256,
            "prepareWorkers": 1,
        },
        "rerunRequested": False,
    }


def write_recovery_audit(
    root: Path,
    role: str,
    action: str,
    expected_digest: str,
    *,
    handled: bool = True,
) -> Path:
    path = root / MODULE.AUDIT_ROOT / f"perfect-chaos-main-recovery-{role}.json"
    write_json(path, recovery_audit(role, action, expected_digest, handled=handled))
    return path


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
        assert result["states"]["red"]["recoveryBlocked"] is False


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


def test_matching_exhausted_recovery_blocks_only_that_role() -> None:
    directory, root = two_role_root()
    with directory:
        red_path = root / MODULE.STATE_ROOT / "red.json"
        write_recovery_audit(root, "red", "exhausted", file_sha256(red_path))
        result = MODULE.inspect_states(
            root,
            commit_times={"red": 5, "yellow": 10},
        )
        red = result["states"]["red"]
        assert red["recoveryBlocked"] is True
        assert red["recoveryAction"] == "exhausted"
        assert red["recoveryExpectedStateSha256"] == red["stateSha256"]
        assert result["selected"]["role"] == "yellow"


def test_matching_non_prepare_exhaustion_blocks_role() -> None:
    directory, root = two_role_root()
    with directory:
        yellow_path = root / MODULE.STATE_ROOT / "yellow.json"
        write_recovery_audit(
            root,
            "yellow",
            "non-prepare-exhausted",
            file_sha256(yellow_path),
        )
        result = MODULE.inspect_states(
            root,
            commit_times={"red": 10, "yellow": 5},
        )
        assert result["states"]["yellow"]["recoveryBlocked"] is True
        assert result["selected"]["role"] == "red"


def test_state_change_invalidates_terminal_recovery_block() -> None:
    directory, root = two_role_root()
    with directory:
        write_recovery_audit(root, "red", "exhausted", "0" * 64)
        result = MODULE.inspect_states(
            root,
            commit_times={"red": 5, "yellow": 10},
        )
        assert result["states"]["red"]["recoveryBlocked"] is False
        assert result["states"]["red"]["recoveryAction"] == "exhausted"
        assert result["selected"]["role"] == "red"


def test_nonterminal_recovery_audit_does_not_block() -> None:
    directory, root = two_role_root()
    with directory:
        red_path = root / MODULE.STATE_ROOT / "red.json"
        write_recovery_audit(root, "red", "adapted", file_sha256(red_path))
        result = MODULE.inspect_states(
            root,
            commit_times={"red": 5, "yellow": 10},
        )
        assert result["states"]["red"]["recoveryBlocked"] is False
        assert result["selected"]["role"] == "red"


def test_malformed_recovery_audit_fails_closed() -> None:
    directory, root = two_role_root()
    with directory:
        audit = recovery_audit("red", "exhausted", "not-a-digest")
        write_json(
            root / MODULE.AUDIT_ROOT / "perfect-chaos-main-recovery-red.json",
            audit,
        )
        expect_failure(
            lambda: MODULE.inspect_states(
                root,
                commit_times={"red": 5, "yellow": 10},
            ),
            "expectedStateSha256",
        )


def test_unhandled_terminal_recovery_fails_closed() -> None:
    directory, root = two_role_root()
    with directory:
        red_path = root / MODULE.STATE_ROOT / "red.json"
        write_recovery_audit(
            root,
            "red",
            "exhausted",
            file_sha256(red_path),
            handled=False,
        )
        expect_failure(
            lambda: MODULE.inspect_states(
                root,
                commit_times={"red": 5, "yellow": 10},
            ),
            "terminal recovery action must be handled",
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
        test_matching_exhausted_recovery_blocks_only_that_role,
        test_matching_non_prepare_exhaustion_blocks_role,
        test_state_change_invalidates_terminal_recovery_block,
        test_nonterminal_recovery_audit_does_not_block,
        test_malformed_recovery_audit_fails_closed,
        test_unhandled_terminal_recovery_fails_closed,
        test_verified_closure_candidate_is_not_relaunched,
        test_malformed_state_fails_closed,
        test_malformed_candidate_fails_closed,
    ]
    for test in tests:
        test()
    print({"tests": len(tests), "status": "pass"})


if __name__ == "__main__":
    main()
