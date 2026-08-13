#!/usr/bin/env python3
"""Fail-closed claim gate for 6x7 Perfect Chaos releases.

A layered prefix safety certificate proves that its selected policy cannot lose
before a certified frontier. It does not prove that the selected move maximises
win > draw > loss. This gate prevents a safety-only artifact from authorising a
"Perfect" product claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

BOARD = {"rows": 6, "columns": 7, "connect": 4, "chaosMode": True}
SAFETY_FORMAT = "connect4-chaos-layered-prefix-manifest-v1"
SAFETY_THEOREM = "finite-safety-game-with-quotient-cycles-lifting-to-threefold-draws"
OPTIMALITY_FORMAT = "connect4-chaos-perfect-optimality-manifest-v1"
OPTIMALITY_THEOREM = (
    "exact-wdl-minimax-with-ranked-winning-progress-and-literal-threefold-repetition"
)
OPTIMALITY_OBJECTIVE = "maximize-win-then-draw-then-loss"
REPORT_FORMAT = "connect4-chaos-claim-gate-report-v1"
ROLES = ("red", "yellow")
ROOT_VALUES = {"win", "draw", "loss"}


def load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular, non-symlink JSON file.")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object.")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_exact(value: Any, expected: Any, field: str) -> None:
    if value != expected:
        raise RuntimeError(f"{field} mismatch: expected {expected!r}, received {value!r}.")


def require_true(value: Any, field: str) -> None:
    if value is not True:
        raise RuntimeError(f"{field} must be exactly true.")


def require_safe_relative_path(base: Path, raw: Any, field: str) -> Path:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise RuntimeError(f"{field} must be a non-empty POSIX relative path.")
    relative = Path(raw)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError(f"{field} escapes or ambiguously addresses its manifest directory.")
    target = base.joinpath(relative)
    resolved_base = base.resolve()
    resolved_target = target.resolve()
    if resolved_target != resolved_base and resolved_base not in resolved_target.parents:
        raise RuntimeError(f"{field} escapes its manifest directory.")
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"{field} does not name a regular, non-symlink artifact.")
    return target


def verify_artifact(base: Path, record: Any, field: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise RuntimeError(f"{field} must be an artifact object.")
    target = require_safe_relative_path(base, record.get("path"), f"{field}.path")
    size = record.get("bytes")
    digest = record.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise RuntimeError(f"{field}.bytes must be a non-negative integer.")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"{field}.sha256 must be a 64-character digest.")
    actual_size = target.stat().st_size
    actual_digest = sha256(target)
    if actual_size != size or actual_digest != digest:
        raise RuntimeError(f"Artifact identity mismatch for {target}.")
    return {"path": str(target), "bytes": actual_size, "sha256": actual_digest}


def verify_safety_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path, "safety manifest")
    require_exact(manifest.get("format"), SAFETY_FORMAT, "safety.format")
    require_exact(manifest.get("theorem"), SAFETY_THEOREM, "safety.theorem")
    require_exact(manifest.get("board"), BOARD, "safety.board")

    boundaries = manifest.get("boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        raise RuntimeError("safety.boundaries must be a non-empty list.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in boundaries):
        raise RuntimeError("safety.boundaries must contain integers only.")
    if boundaries[0] != 8 or any(
        current != previous + 2 for previous, current in zip(boundaries, boundaries[1:])
    ):
        raise RuntimeError("safety.boundaries must be contiguous even layers beginning at 8.")

    roles = manifest.get("roles")
    artifacts = manifest.get("artifacts")
    if not isinstance(roles, dict) or not isinstance(artifacts, dict):
        raise RuntimeError("safety manifest must contain role and artifact maps.")

    verified_artifacts: dict[str, list[dict[str, Any]]] = {}
    manifest_base = path.parent
    for role in ROLES:
        role_record = roles.get(role)
        if not isinstance(role_record, dict):
            raise RuntimeError(f"safety.roles.{role} is missing.")
        replay = role_record.get("replay")
        if not isinstance(replay, dict) or replay.get("role") != role:
            raise RuntimeError(f"safety.roles.{role}.replay is incompatible.")
        segments = replay.get("segments")
        if not isinstance(segments, list) or len(segments) != len(boundaries):
            raise RuntimeError(f"safety.roles.{role}.replay has the wrong segment count.")
        expected_from = 0
        for index, (segment, boundary) in enumerate(zip(segments, boundaries)):
            if not isinstance(segment, dict):
                raise RuntimeError(f"safety.roles.{role}.replay.segments[{index}] is invalid.")
            require_exact(
                segment.get("fromPieces"), expected_from,
                f"safety.roles.{role}.replay.segments[{index}].fromPieces",
            )
            require_exact(
                segment.get("frontierPieces"), boundary,
                f"safety.roles.{role}.replay.segments[{index}].frontierPieces",
            )
            expected_from = boundary

        records = artifacts.get(role)
        if not isinstance(records, list) or not records:
            raise RuntimeError(f"safety.artifacts.{role} must be non-empty.")
        seen_paths: set[str] = set()
        verified_artifacts[role] = []
        role_base = manifest_base / role
        for index, record in enumerate(records):
            if isinstance(record, dict) and record.get("path") in seen_paths:
                raise RuntimeError(f"safety.artifacts.{role} contains a duplicate path.")
            if isinstance(record, dict) and isinstance(record.get("path"), str):
                seen_paths.add(record["path"])
            verified_artifacts[role].append(
                verify_artifact(role_base, record, f"safety.artifacts.{role}[{index}]")
            )

    return {
        "manifestSha256": sha256(path),
        "frontierPieces": boundaries[-1],
        "objective": "non-losing-safety",
        "optimality": "unproved",
        "artifacts": verified_artifacts,
    }


def verify_optimality_manifest(path: Path, safety_hash: str) -> dict[str, Any]:
    manifest = load_json(path, "optimality manifest")
    require_exact(manifest.get("format"), OPTIMALITY_FORMAT, "optimality.format")
    require_exact(manifest.get("theorem"), OPTIMALITY_THEOREM, "optimality.theorem")
    require_exact(manifest.get("objective"), OPTIMALITY_OBJECTIVE, "optimality.objective")
    require_exact(manifest.get("board"), BOARD, "optimality.board")
    require_exact(
        manifest.get("safetyManifestSha256"), safety_hash,
        "optimality.safetyManifestSha256",
    )

    coverage = manifest.get("coverage")
    if not isinstance(coverage, dict):
        raise RuntimeError("optimality.coverage is missing.")
    for field in (
        "fromEmptyBoard",
        "allReachableAiDecisionsValued",
        "allLegalOpponentActionsCovered",
        "frontierHandoffComplete",
        "literalThreefoldVerified",
    ):
        require_true(coverage.get(field), f"optimality.coverage.{field}")

    independence = manifest.get("independence")
    if not isinstance(independence, dict):
        raise RuntimeError("optimality.independence is missing.")
    implementations = independence.get("implementations")
    if (
        not isinstance(implementations, list)
        or len(implementations) < 2
        or len(set(implementations)) != len(implementations)
        or any(not isinstance(value, str) or not value for value in implementations)
    ):
        raise RuntimeError("optimality requires at least two distinct named implementations.")
    require_true(independence.get("agreement"), "optimality.independence.agreement")

    roles = manifest.get("roles")
    if not isinstance(roles, dict):
        raise RuntimeError("optimality.roles is missing.")
    root_values: dict[str, str] = {}
    for role in ROLES:
        record = roles.get(role)
        if not isinstance(record, dict):
            raise RuntimeError(f"optimality.roles.{role} is missing.")
        root_value = record.get("rootValue")
        if root_value not in ROOT_VALUES:
            raise RuntimeError(f"optimality.roles.{role}.rootValue is invalid.")
        root_values[role] = root_value
        for field in (
            "policyComplete",
            "allChosenActionsOptimal",
            "rankedWinningProgressVerified",
            "drawRegionClosedVerified",
            "adversarialClosureComplete",
        ):
            require_true(record.get(field), f"optimality.roles.{role}.{field}")

    artifact_records = manifest.get("artifacts")
    if not isinstance(artifact_records, list) or not artifact_records:
        raise RuntimeError("optimality.artifacts must be non-empty.")
    verified_artifacts: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, record in enumerate(artifact_records):
        if isinstance(record, dict) and record.get("path") in seen_paths:
            raise RuntimeError("optimality.artifacts contains a duplicate path.")
        if isinstance(record, dict) and isinstance(record.get("path"), str):
            seen_paths.add(record["path"])
        verified_artifacts.append(
            verify_artifact(path.parent, record, f"optimality.artifacts[{index}]")
        )

    return {
        "manifestSha256": sha256(path),
        "objective": OPTIMALITY_OBJECTIVE,
        "rootValues": root_values,
        "independentImplementations": implementations,
        "artifacts": verified_artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", required=True, choices=("safety", "perfect"))
    parser.add_argument("--safety-manifest", required=True, type=Path)
    parser.add_argument("--optimality-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    safety = verify_safety_manifest(args.safety_manifest)
    optimality = None
    if args.claim == "perfect":
        if args.optimality_manifest is None:
            raise RuntimeError(
                "A non-losing safety certificate cannot authorise the Perfect Chaos label; "
                "an exact W/D/L optimality manifest is required."
            )
        optimality = verify_optimality_manifest(
            args.optimality_manifest,
            safety["manifestSha256"],
        )
    elif args.optimality_manifest is not None:
        optimality = verify_optimality_manifest(
            args.optimality_manifest,
            safety["manifestSha256"],
        )

    report = {
        "format": REPORT_FORMAT,
        "requestedClaim": args.claim,
        "safety": safety,
        "optimality": optimality,
        "perfectClaimAllowed": args.claim == "perfect" and optimality is not None,
        "allowedLabel": "Perfect Chaos" if optimality is not None else "Non-losing certified",
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
