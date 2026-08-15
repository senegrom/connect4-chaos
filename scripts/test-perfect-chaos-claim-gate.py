#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BOARD = {"rows": 6, "columns": 7, "connect": 4, "chaosMode": True}
THEOREM = "exact-wdl-minimax-with-ranked-winning-progress-and-literal-threefold-repetition"
OBJECTIVE = "maximize-win-then-draw-then-loss"
COVERAGE = {
    "fromEmptyBoard": True,
    "allReachableAiDecisionsValued": True,
    "allLegalOpponentActionsCovered": True,
    "frontierHandoffComplete": True,
    "literalThreefoldVerified": True,
}
FLAGS = {
    "policyComplete": True,
    "allChosenActionsOptimal": True,
    "rankedWinningProgressVerified": True,
    "drawRegionClosedVerified": True,
    "adversarialClosureComplete": True,
}
KINDS = ("graph", "values", "policy", "closure")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def artifact(root: Path, relative: str) -> dict:
    path = root / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)}


def invoke(script: Path, safety: Path, claim: str, optimality: Path | None = None):
    command = [sys.executable, str(script), "--claim", claim, "--safety-manifest", str(safety)]
    if optimality is not None:
        command.extend(["--optimality-manifest", str(optimality)])
    return subprocess.run(command, capture_output=True, text=True, check=False)


def require_success(result) -> dict:
    if result.returncode != 0:
        raise AssertionError(f"claim gate rejected valid evidence:\n{result.stderr}")
    return json.loads(result.stdout)


def require_failure(result, marker: str) -> None:
    if result.returncode == 0:
        raise AssertionError(f"claim gate accepted invalid evidence:\n{result.stdout}")
    if marker not in result.stderr:
        raise AssertionError(f"missing failure marker {marker!r}:\n{result.stderr}")


def safety_fixture(root: Path) -> Path:
    safety = root / "safety"
    for role in ("red", "yellow"):
        role_dir = safety / role
        role_dir.mkdir(parents=True, exist_ok=True)
        (role_dir / "0-8.policy.bin").write_bytes(f"{role}-policy".encode())
        (role_dir / "0-8.frontier.bin").write_bytes(f"{role}-frontier".encode())
    value = {
        "format": "connect4-chaos-layered-prefix-manifest-v1",
        "theorem": "finite-safety-game-with-quotient-cycles-lifting-to-threefold-draws",
        "board": BOARD,
        "boundaries": [8],
        "roles": {
            role: {"replay": {"role": role, "segments": [{"fromPieces": 0, "frontierPieces": 8}]}}
            for role in ("red", "yellow")
        },
        "artifacts": {
            role: [
                artifact(safety / role, "0-8.policy.bin"),
                artifact(safety / role, "0-8.frontier.bin"),
            ]
            for role in ("red", "yellow")
        },
    }
    path = safety / "manifest.json"
    write_json(path, value)
    return path


def optimality_fixture(root: Path, safety_hash: str, *, same_sources: bool = False) -> Path:
    proof = root / "optimality"
    proof.mkdir(parents=True, exist_ok=True)
    role_claims = {}
    for role, root_value in (("red", "win"), ("yellow", "draw")):
        selected = {}
        for kind in KINDS:
            relative = f"proof/{role}-{kind}.bin"
            path = proof / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{role}-{kind}-evidence".encode())
            selected[kind] = relative
        role_claims[role] = {
            "rootValue": root_value,
            **FLAGS,
            "proofArtifacts": selected,
        }

    sources = {
        "python-wdl": "verifiers/python-wdl.py",
        "reference-js": "verifiers/reference-wdl.mjs",
    }
    (proof / sources["python-wdl"]).parent.mkdir(parents=True, exist_ok=True)
    (proof / sources["python-wdl"]).write_text("# independent Python verifier\n")
    (proof / sources["reference-js"]).write_text(
        "# independent Python verifier\n" if same_sources else "// independent JavaScript verifier\n"
    )

    report_paths = {}
    for name, source in sources.items():
        report_relative = f"reports/{name}.json"
        report_paths[name] = report_relative
        roles = {}
        for role in ("red", "yellow"):
            roles[role] = {
                "rootValue": role_claims[role]["rootValue"],
                **FLAGS,
                "proofArtifactSha256": {
                    kind: digest(proof / role_claims[role]["proofArtifacts"][kind])
                    for kind in KINDS
                },
            }
        write_json(proof / report_relative, {
            "format": "connect4-chaos-perfect-optimality-verifier-report-v1",
            "implementation": name,
            "implementationSourceSha256": digest(proof / source),
            "objective": OBJECTIVE,
            "board": BOARD,
            "safetyManifestSha256": safety_hash,
            "coverage": COVERAGE,
            "roles": roles,
        })

    all_paths = [
        role_claims[role]["proofArtifacts"][kind]
        for role in ("red", "yellow")
        for kind in KINDS
    ] + list(sources.values()) + list(report_paths.values())
    manifest = {
        "format": "connect4-chaos-perfect-optimality-manifest-v2",
        "theorem": THEOREM,
        "objective": OBJECTIVE,
        "board": BOARD,
        "safetyManifestSha256": safety_hash,
        "coverage": COVERAGE,
        "roles": role_claims,
        "artifacts": [artifact(proof, relative) for relative in all_paths],
        "independence": {
            "implementations": [
                {"name": name, "source": source, "report": report_paths[name]}
                for name, source in sources.items()
            ]
        },
    }
    path = proof / "optimality.json"
    write_json(path, manifest)
    return path


def refresh_artifact(manifest_path: Path, relative: str) -> None:
    value = json.loads(manifest_path.read_text())
    record = next(item for item in value["artifacts"] if item["path"] == relative)
    path = manifest_path.parent / relative
    record["bytes"] = path.stat().st_size
    record["sha256"] = digest(path)
    write_json(manifest_path, value)


def main() -> int:
    repository = Path(__file__).resolve().parent.parent
    script = Path(__file__).with_name("perfect-chaos-claim-gate.py")
    actual_safety = repository / "data" / "perfect-chaos-prefix" / "manifest.json"
    if actual_safety.exists():
        actual_report = require_success(invoke(script, actual_safety, "safety"))
        assert actual_report["perfectClaimAllowed"] is False
        assert actual_report["allowedLabel"] == "Non-losing certified"

    with tempfile.TemporaryDirectory(prefix="claim-gate-v2-") as temporary:
        root = Path(temporary)
        safety = safety_fixture(root)
        safety_hash = digest(safety)

        safety_report = require_success(invoke(script, safety, "safety"))
        assert safety_report["perfectClaimAllowed"] is False
        assert safety_report["allowedLabel"] == "Non-losing certified"
        require_failure(invoke(script, safety, "perfect"), "cannot authorise")

        valid = optimality_fixture(root / "valid", safety_hash)
        perfect = require_success(invoke(script, safety, "perfect", valid))
        assert perfect["perfectClaimAllowed"] is True
        assert perfect["allowedLabel"] == "Perfect Chaos"
        assert perfect["optimality"]["independentImplementations"] == [
            "python-wdl", "reference-js"
        ]

        safety_with_optimality = require_success(invoke(script, safety, "safety", valid))
        assert safety_with_optimality["perfectClaimAllowed"] is False
        assert safety_with_optimality["allowedLabel"] == "Non-losing certified"

        one = optimality_fixture(root / "one", safety_hash)
        value = json.loads(one.read_text())
        removed = value["independence"]["implementations"].pop()
        value["artifacts"] = [
            item for item in value["artifacts"]
            if item["path"] not in {removed["source"], removed["report"]}
        ]
        write_json(one, value)
        require_failure(invoke(script, safety, "perfect", one), "at least two")

        same = optimality_fixture(root / "same", safety_hash, same_sources=True)
        require_failure(invoke(script, safety, "perfect", same), "distinct source-code hashes")

        mismatch = optimality_fixture(root / "mismatch", safety_hash)
        report_path = mismatch.parent / "reports/reference-js.json"
        report = json.loads(report_path.read_text())
        report["roles"]["red"]["rootValue"] = "loss"
        write_json(report_path, report)
        refresh_artifact(mismatch, "reports/reference-js.json")
        require_failure(invoke(script, safety, "perfect", mismatch), "reference-js.roles.red mismatch")

        wrong_safety = optimality_fixture(root / "wrong-safety", safety_hash)
        value = json.loads(wrong_safety.read_text())
        value["safetyManifestSha256"] = "0" * 64
        write_json(wrong_safety, value)
        require_failure(invoke(script, safety, "perfect", wrong_safety), "safetyManifestSha256")

        tampered = optimality_fixture(root / "tampered", safety_hash)
        (tampered.parent / "proof/red-policy.bin").write_bytes(b"tampered")
        require_failure(invoke(script, safety, "perfect", tampered), "Artifact identity mismatch")

        extra = optimality_fixture(root / "extra", safety_hash)
        extra_file = extra.parent / "unreferenced.bin"
        extra_file.write_bytes(b"extra")
        value = json.loads(extra.read_text())
        value["artifacts"].append(artifact(extra.parent, "unreferenced.bin"))
        write_json(extra, value)
        require_failure(invoke(script, safety, "perfect", extra), "unreferenced file")

        false_flag = optimality_fixture(root / "false-flag", safety_hash)
        value = json.loads(false_flag.read_text())
        value["roles"]["red"]["allChosenActionsOptimal"] = False
        write_json(false_flag, value)
        require_failure(invoke(script, safety, "perfect", false_flag), "allChosenActionsOptimal")

    print("perfect-chaos-claim-gate-v2: all regression cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
