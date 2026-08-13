#!/usr/bin/env python3
"""Regression tests for the Perfect Chaos safety/optimality claim boundary."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def invoke(
    script: Path,
    safety: Path,
    claim: str,
    optimality: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(script),
        "--claim", claim,
        "--safety-manifest", str(safety),
    ]
    if optimality is not None:
        command.extend(["--optimality-manifest", str(optimality)])
    return subprocess.run(command, capture_output=True, text=True, check=False)


def require_success(result: subprocess.CompletedProcess[str]) -> dict:
    if result.returncode != 0:
        raise AssertionError(f"claim gate rejected valid evidence:\n{result.stderr}")
    return json.loads(result.stdout)


def require_failure(result: subprocess.CompletedProcess[str], marker: str) -> None:
    if result.returncode == 0:
        raise AssertionError(f"claim gate accepted invalid evidence:\n{result.stdout}")
    if marker not in result.stderr:
        raise AssertionError(f"missing failure marker {marker!r}:\n{result.stderr}")


def optimality_fixture(directory: Path, safety_hash: str) -> Path:
    policy = directory / "optimal-policy.bin"
    values = directory / "wdl-values.bin"
    policy.write_bytes(b"synthetic-optimal-policy")
    values.write_bytes(b"synthetic-exact-wdl-values")
    manifest = {
        "format": "connect4-chaos-perfect-optimality-manifest-v1",
        "theorem": (
            "exact-wdl-minimax-with-ranked-winning-progress-and-literal-threefold-repetition"
        ),
        "objective": "maximize-win-then-draw-then-loss",
        "board": {"rows": 6, "columns": 7, "connect": 4, "chaosMode": True},
        "safetyManifestSha256": safety_hash,
        "coverage": {
            "fromEmptyBoard": True,
            "allReachableAiDecisionsValued": True,
            "allLegalOpponentActionsCovered": True,
            "frontierHandoffComplete": True,
            "literalThreefoldVerified": True,
        },
        "independence": {
            "implementations": ["native-cpp", "independent-reference"],
            "agreement": True,
        },
        "roles": {
            role: {
                "rootValue": value,
                "policyComplete": True,
                "allChosenActionsOptimal": True,
                "rankedWinningProgressVerified": True,
                "drawRegionClosedVerified": True,
                "adversarialClosureComplete": True,
            }
            for role, value in (("red", "win"), ("yellow", "draw"))
        },
        "artifacts": [
            {"path": policy.name, "bytes": policy.stat().st_size, "sha256": digest(policy)},
            {"path": values.name, "bytes": values.stat().st_size, "sha256": digest(values)},
        ],
    }
    path = directory / "optimality.json"
    write_json(path, manifest)
    return path


def main() -> int:
    repository = Path(__file__).resolve().parent.parent
    script = repository / "scripts" / "perfect-chaos-claim-gate.py"
    safety = repository / "data" / "perfect-chaos-prefix" / "manifest.json"

    safety_report = require_success(invoke(script, safety, "safety"))
    if safety_report["perfectClaimAllowed"] is not False:
        raise AssertionError("safety-only evidence unexpectedly authorises Perfect Chaos")
    if safety_report["safety"]["objective"] != "non-losing-safety":
        raise AssertionError("safety certificate objective was misclassified")
    if safety_report["allowedLabel"] != "Non-losing certified":
        raise AssertionError("safety certificate received a misleading label")

    require_failure(
        invoke(script, safety, "perfect"),
        "cannot authorise the Perfect Chaos label",
    )

    with tempfile.TemporaryDirectory(prefix="perfect-chaos-claim-") as temporary:
        root = Path(temporary)
        optimality = optimality_fixture(root, digest(safety))
        perfect_report = require_success(invoke(script, safety, "perfect", optimality))
        if perfect_report["perfectClaimAllowed"] is not True:
            raise AssertionError("complete synthetic optimality evidence was not accepted")
        if perfect_report["allowedLabel"] != "Perfect Chaos":
            raise AssertionError("valid optimality evidence received the wrong label")

        value = json.loads(optimality.read_text())
        value["roles"]["red"]["allChosenActionsOptimal"] = False
        invalid = root / "non-optimal-action.json"
        write_json(invalid, value)
        require_failure(invoke(script, safety, "perfect", invalid), "allChosenActionsOptimal")

        value = json.loads(optimality.read_text())
        value["safetyManifestSha256"] = "0" * 64
        invalid = root / "wrong-safety.json"
        write_json(invalid, value)
        require_failure(invoke(script, safety, "perfect", invalid), "safetyManifestSha256")

        value = json.loads(optimality.read_text())
        value["independence"]["implementations"] = ["one-solver"]
        invalid = root / "one-solver.json"
        write_json(invalid, value)
        require_failure(invoke(script, safety, "perfect", invalid), "two distinct")

        value = json.loads(optimality.read_text())
        value["coverage"]["literalThreefoldVerified"] = False
        invalid = root / "wrong-repetition.json"
        write_json(invalid, value)
        require_failure(invoke(script, safety, "perfect", invalid), "literalThreefoldVerified")

        value = json.loads(optimality.read_text())
        (root / "optimal-policy.bin").write_bytes(b"tampered")
        invalid = root / "tampered-artifact.json"
        write_json(invalid, value)
        require_failure(invoke(script, safety, "perfect", invalid), "Artifact identity mismatch")

    print("perfect-chaos-claim-gate: all regression cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
