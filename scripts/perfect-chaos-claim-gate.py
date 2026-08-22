#!/usr/bin/env python3
"""Fail-closed claim gate for 6x7 Perfect Chaos releases.

A layered prefix safety certificate proves that its selected policy cannot lose
before a certified frontier. It does not prove that the selected move maximises
win > draw > loss. A Perfect claim additionally requires two independently
implemented verifier reports, their source identities, and every proof artifact
to agree exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

BOARD = {"rows": 6, "columns": 7, "connect": 4, "chaosMode": True}
SAFETY_FORMAT = "connect4-chaos-layered-prefix-manifest-v1"
SAFETY_THEOREM = "finite-safety-game-with-quotient-cycles-lifting-to-threefold-draws"
OPTIMALITY_FORMAT = "connect4-chaos-perfect-optimality-manifest-v2"
OPTIMALITY_THEOREM = (
    "exact-wdl-minimax-with-ranked-winning-progress-and-literal-threefold-repetition"
)
OPTIMALITY_OBJECTIVE = "maximize-win-then-draw-then-loss"
VERIFIER_REPORT_FORMAT = "connect4-chaos-perfect-optimality-verifier-report-v1"
REPORT_FORMAT = "connect4-chaos-claim-gate-report-v2"
ROLES = ("red", "yellow")
ROOT_VALUES = {"win", "draw", "loss"}
COVERAGE_FIELDS = (
    "fromEmptyBoard",
    "allReachableAiDecisionsValued",
    "allLegalOpponentActionsCovered",
    "frontierHandoffComplete",
    "literalThreefoldVerified",
)
ROLE_FLAGS = (
    "policyComplete",
    "allChosenActionsOptimal",
    "rankedWinningProgressVerified",
    "drawRegionClosedVerified",
    "adversarialClosureComplete",
)
PROOF_KINDS = ("graph", "values", "policy", "closure")
DIGEST = re.compile(r"[0-9a-f]{64}")


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


def require_keys(value: dict[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected.difference(value))
    unknown = sorted(set(value).difference(expected))
    if missing or unknown:
        raise RuntimeError(f"{field} has missing={missing!r}, unknown={unknown!r}.")


def require_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or DIGEST.fullmatch(value) is None:
        raise RuntimeError(f"{field} must be a lowercase SHA-256 digest.")
    return value


def safe_relative(value: Any, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError(f"{field} must be a non-empty POSIX relative path.")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != value
    ):
        raise RuntimeError(f"{field} escapes or ambiguously addresses its manifest directory.")
    return relative


def require_safe_relative_path(base: Path, raw: Any, field: str) -> tuple[str, Path]:
    relative = safe_relative(raw, field)
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"{field} may not traverse a symlink.")
    resolved_base = base.resolve()
    resolved_target = current.resolve()
    if resolved_target != resolved_base and resolved_base not in resolved_target.parents:
        raise RuntimeError(f"{field} escapes its manifest directory.")
    if not current.is_file():
        raise RuntimeError(f"{field} does not name a regular, non-symlink artifact.")
    return relative.as_posix(), current


def verify_artifact(base: Path, record: Any, field: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise RuntimeError(f"{field} must be an artifact object.")
    require_keys(record, {"path", "bytes", "sha256"}, field)
    relative, target = require_safe_relative_path(base, record["path"], f"{field}.path")
    size = record["bytes"]
    digest = require_digest(record["sha256"], f"{field}.sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise RuntimeError(f"{field}.bytes must be a non-negative integer.")
    actual_size = target.stat().st_size
    actual_digest = sha256(target)
    if actual_size != size or actual_digest != digest:
        raise RuntimeError(f"Artifact identity mismatch for {target}.")
    return {"path": relative, "bytes": actual_size, "sha256": actual_digest}


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
            verified = verify_artifact(
                role_base, record, f"safety.artifacts.{role}[{index}]"
            )
            if verified["path"] in seen_paths:
                raise RuntimeError(f"safety.artifacts.{role} contains a duplicate path.")
            seen_paths.add(verified["path"])
            verified_artifacts[role].append(verified)

    return {
        "manifestSha256": sha256(path),
        "frontierPieces": boundaries[-1],
        "objective": "non-losing-safety",
        "optimality": "unproved",
        "artifacts": verified_artifacts,
    }


def validate_coverage(value: Any, field: str) -> dict[str, bool]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{field} is missing.")
    require_keys(value, set(COVERAGE_FIELDS), field)
    for name in COVERAGE_FIELDS:
        require_true(value[name], f"{field}.{name}")
    return {name: True for name in COVERAGE_FIELDS}


def validate_role_claim(
    value: Any,
    field: str,
    artifact_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{field} is missing.")
    expected = {"rootValue", "proofArtifacts", *ROLE_FLAGS}
    require_keys(value, expected, field)
    root_value = value["rootValue"]
    if root_value not in ROOT_VALUES:
        raise RuntimeError(f"{field}.rootValue is invalid.")
    for name in ROLE_FLAGS:
        require_true(value[name], f"{field}.{name}")
    proof = value["proofArtifacts"]
    if not isinstance(proof, dict):
        raise RuntimeError(f"{field}.proofArtifacts must be an object.")
    require_keys(proof, set(PROOF_KINDS), f"{field}.proofArtifacts")
    selected: dict[str, dict[str, Any]] = {}
    for kind in PROOF_KINDS:
        relative = safe_relative(proof[kind], f"{field}.proofArtifacts.{kind}").as_posix()
        artifact = artifact_map.get(relative)
        if artifact is None:
            raise RuntimeError(f"{field}.proofArtifacts.{kind} is not a verified artifact.")
        selected[kind] = artifact
    if len({record["path"] for record in selected.values()}) != len(PROOF_KINDS):
        raise RuntimeError(f"{field}.proofArtifacts must identify distinct files.")
    return {
        "rootValue": root_value,
        **{name: True for name in ROLE_FLAGS},
        "proofArtifacts": {kind: selected[kind]["path"] for kind in PROOF_KINDS},
        "proofArtifactSha256": {kind: selected[kind]["sha256"] for kind in PROOF_KINDS},
    }


def validate_report_role(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{field} is missing.")
    expected = {"rootValue", "proofArtifactSha256", *ROLE_FLAGS}
    require_keys(value, expected, field)
    root_value = value["rootValue"]
    if root_value not in ROOT_VALUES:
        raise RuntimeError(f"{field}.rootValue is invalid.")
    for name in ROLE_FLAGS:
        require_true(value[name], f"{field}.{name}")
    hashes = value["proofArtifactSha256"]
    if not isinstance(hashes, dict):
        raise RuntimeError(f"{field}.proofArtifactSha256 must be an object.")
    require_keys(hashes, set(PROOF_KINDS), f"{field}.proofArtifactSha256")
    return {
        "rootValue": root_value,
        **{name: True for name in ROLE_FLAGS},
        "proofArtifactSha256": {
            kind: require_digest(hashes[kind], f"{field}.proofArtifactSha256.{kind}")
            for kind in PROOF_KINDS
        },
    }


def verify_verifier_report(
    path: Path,
    implementation: str,
    source_digest: str,
    safety_hash: str,
    coverage: dict[str, bool],
    role_claims: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    report = load_json(path, f"{implementation} verifier report")
    require_keys(
        report,
        {
            "format", "implementation", "implementationSourceSha256",
            "objective", "board", "safetyManifestSha256", "coverage", "roles",
        },
        f"{implementation} verifier report",
    )
    require_exact(report["format"], VERIFIER_REPORT_FORMAT, f"{implementation}.format")
    require_exact(report["implementation"], implementation, f"{implementation}.implementation")
    require_exact(
        require_digest(report["implementationSourceSha256"], f"{implementation}.implementationSourceSha256"),
        source_digest,
        f"{implementation}.implementationSourceSha256",
    )
    require_exact(report["objective"], OPTIMALITY_OBJECTIVE, f"{implementation}.objective")
    require_exact(report["board"], BOARD, f"{implementation}.board")
    require_exact(
        report["safetyManifestSha256"], safety_hash,
        f"{implementation}.safetyManifestSha256",
    )
    report_coverage = validate_coverage(report["coverage"], f"{implementation}.coverage")
    require_exact(report_coverage, coverage, f"{implementation}.coverage")
    reports = report["roles"]
    if not isinstance(reports, dict):
        raise RuntimeError(f"{implementation}.roles is missing.")
    require_keys(reports, set(ROLES), f"{implementation}.roles")
    verified_roles: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        verified = validate_report_role(reports[role], f"{implementation}.roles.{role}")
        expected = {
            key: value
            for key, value in role_claims[role].items()
            if key != "proofArtifacts"
        }
        require_exact(verified, expected, f"{implementation}.roles.{role}")
        verified_roles[role] = verified
    return {
        "implementation": implementation,
        "implementationSourceSha256": source_digest,
        "reportSha256": sha256(path),
        "coverage": report_coverage,
        "roles": verified_roles,
    }


def verify_optimality_manifest(path: Path, safety_hash: str) -> dict[str, Any]:
    manifest = load_json(path, "optimality manifest")
    require_keys(
        manifest,
        {
            "format", "theorem", "objective", "board", "safetyManifestSha256",
            "coverage", "roles", "artifacts", "independence",
        },
        "optimality",
    )
    require_exact(manifest["format"], OPTIMALITY_FORMAT, "optimality.format")
    require_exact(manifest["theorem"], OPTIMALITY_THEOREM, "optimality.theorem")
    require_exact(manifest["objective"], OPTIMALITY_OBJECTIVE, "optimality.objective")
    require_exact(manifest["board"], BOARD, "optimality.board")
    require_exact(manifest["safetyManifestSha256"], safety_hash, "optimality.safetyManifestSha256")
    coverage = validate_coverage(manifest["coverage"], "optimality.coverage")

    records = manifest["artifacts"]
    if not isinstance(records, list) or not records:
        raise RuntimeError("optimality.artifacts must be non-empty.")
    artifact_map: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        verified = verify_artifact(path.parent, record, f"optimality.artifacts[{index}]")
        if verified["path"] in artifact_map:
            raise RuntimeError("optimality.artifacts contains a duplicate path.")
        artifact_map[verified["path"]] = verified

    roles = manifest["roles"]
    if not isinstance(roles, dict):
        raise RuntimeError("optimality.roles is missing.")
    require_keys(roles, set(ROLES), "optimality.roles")
    role_claims = {
        role: validate_role_claim(roles[role], f"optimality.roles.{role}", artifact_map)
        for role in ROLES
    }
    proof_paths = {
        role_claims[role]["proofArtifacts"][kind]
        for role in ROLES
        for kind in PROOF_KINDS
    }
    if len(proof_paths) != len(ROLES) * len(PROOF_KINDS):
        raise RuntimeError("Optimality role proof artifacts must be distinct across roles.")

    independence = manifest["independence"]
    if not isinstance(independence, dict):
        raise RuntimeError("optimality.independence is missing.")
    require_keys(independence, {"implementations"}, "optimality.independence")
    implementations = independence["implementations"]
    if not isinstance(implementations, list) or len(implementations) < 2:
        raise RuntimeError("optimality requires at least two independent verifier reports.")

    seen_names: set[str] = set()
    source_paths: set[str] = set()
    report_paths: set[str] = set()
    source_hashes: set[str] = set()
    verified_reports: list[dict[str, Any]] = []
    for index, record in enumerate(implementations):
        field = f"optimality.independence.implementations[{index}]"
        if not isinstance(record, dict):
            raise RuntimeError(f"{field} must be an object.")
        require_keys(record, {"name", "source", "report"}, field)
        name = record["name"]
        if not isinstance(name, str) or not name or name in seen_names:
            raise RuntimeError("Verifier implementation names must be non-empty and distinct.")
        seen_names.add(name)
        source = safe_relative(record["source"], f"{field}.source").as_posix()
        report_path = safe_relative(record["report"], f"{field}.report").as_posix()
        if source not in artifact_map or report_path not in artifact_map:
            raise RuntimeError(f"{field} source and report must be verified artifacts.")
        if source in source_paths or report_path in report_paths:
            raise RuntimeError("Verifier source and report paths must be distinct.")
        source_paths.add(source)
        report_paths.add(report_path)
        source_hashes.add(artifact_map[source]["sha256"])
        _, report_file = require_safe_relative_path(path.parent, report_path, f"{field}.report")
        verified_reports.append(
            verify_verifier_report(
                report_file,
                name,
                artifact_map[source]["sha256"],
                safety_hash,
                coverage,
                role_claims,
            )
        )

    if len(source_hashes) < 2:
        raise RuntimeError("Independent verifiers must have distinct source-code hashes.")
    if proof_paths & source_paths or proof_paths & report_paths or source_paths & report_paths:
        raise RuntimeError("Proof, verifier-source, and verifier-report artifacts must be disjoint.")
    referenced = proof_paths | source_paths | report_paths
    unreferenced = sorted(set(artifact_map).difference(referenced))
    if unreferenced:
        raise RuntimeError(f"optimality.artifacts contains unreferenced file(s): {unreferenced!r}.")

    return {
        "manifestSha256": sha256(path),
        "objective": OPTIMALITY_OBJECTIVE,
        "rootValues": {role: role_claims[role]["rootValue"] for role in ROLES},
        "independentImplementations": [report["implementation"] for report in verified_reports],
        "verifierReports": verified_reports,
        "artifacts": [artifact_map[path] for path in sorted(artifact_map)],
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
        optimality = verify_optimality_manifest(args.optimality_manifest, safety["manifestSha256"])
    elif args.optimality_manifest is not None:
        optimality = verify_optimality_manifest(args.optimality_manifest, safety["manifestSha256"])

    perfect_allowed = args.claim == "perfect" and optimality is not None
    report = {
        "format": REPORT_FORMAT,
        "requestedClaim": args.claim,
        "safety": safety,
        "optimality": optimality,
        "perfectClaimAllowed": perfect_allowed,
        "allowedLabel": "Perfect Chaos" if perfect_allowed else "Non-losing certified",
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
