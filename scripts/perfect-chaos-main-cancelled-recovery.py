#!/usr/bin/env python3
"""Adapt one exact Perfect Chaos preparation cancelled at the runner boundary.

The ordinary recovery module accepts completed failures. GitHub reports a job
that reaches the hosted-runner wall-clock ceiling as ``cancelled`` instead.
This wrapper admits only that narrow shape: the workflow and its unique exact
preparation job must both be completed/cancelled. It then delegates all role,
state, identity, schema, and resource-profile decisions to the existing pure
recovery implementation after normalising only the two conclusion labels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, NoReturn

MODULE_PATH = Path(__file__).with_name("perfect-chaos-main-recovery.py")
SPEC = importlib.util.spec_from_file_location("perfect_chaos_main_recovery", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)

FORMAT = "connect4-chaos-main-cancelled-preparation-recovery-v1"


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def cancelled_prepare_job(jobs_payload: dict[str, Any]) -> dict[str, Any]:
    jobs = jobs_payload.get("jobs")
    if not isinstance(jobs, list):
        fail("Jobs payload must contain a jobs array")
    matches = [
        job
        for job in jobs
        if isinstance(job, dict) and job.get("name") == RECOVERY.PREPARE_JOB
    ]
    if len(matches) != 1:
        fail(
            f"A cancelled recovery requires exactly one {RECOVERY.PREPARE_JOB!r} "
            f"job; found {len(matches)}"
        )
    prepare = matches[0]
    if prepare.get("status") != "completed" or prepare.get("conclusion") != "cancelled":
        fail(
            "Cancelled recovery requires the exact preparation job itself to be "
            f"completed/cancelled; status={prepare.get('status')!r}, "
            f"conclusion={prepare.get('conclusion')!r}"
        )
    return prepare


def decide_cancelled_recovery(
    run: dict[str, Any],
    jobs_payload: dict[str, Any],
    current_state: dict[str, Any],
    current_bytes: bytes,
    run_state_bytes: bytes,
    *,
    changed_paths: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if run.get("status") != "completed" or run.get("conclusion") != "cancelled":
        fail("Cancelled preparation recovery requires a completed cancelled workflow run")
    prepare = cancelled_prepare_job(jobs_payload)

    normalised_run = dict(run)
    normalised_run["conclusion"] = "failure"
    normalised_jobs = {
        **jobs_payload,
        "jobs": [dict(job) if isinstance(job, dict) else job for job in jobs_payload["jobs"]],
    }
    for job in normalised_jobs["jobs"]:
        if isinstance(job, dict) and job.get("name") == RECOVERY.PREPARE_JOB:
            job["conclusion"] = "failure"

    decision, updated = RECOVERY.decide_recovery(
        normalised_run,
        normalised_jobs,
        current_state,
        current_bytes,
        run_state_bytes,
        changed_paths=changed_paths,
    )
    if decision.get("action") == "not-prepare-failure":
        fail("Cancelled preparation normalisation did not produce a preparation decision")

    decision = {
        **decision,
        "format": FORMAT,
        "runConclusion": "cancelled",
        "prepareConclusion": prepare["conclusion"],
        "normalisedConclusion": "failure",
    }
    return decision, updated


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--state-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    root = arguments.root.resolve()
    if root.is_symlink() or not root.is_dir():
        fail(f"Repository root must be a regular directory: {root}")

    run = RECOVERY.load_object(arguments.run, "workflow run")
    jobs = RECOVERY.load_object(arguments.jobs, "workflow jobs")
    head_sha = run.get("head_sha")
    if not isinstance(head_sha, str) or RECOVERY.SHA_RE.fullmatch(head_sha) is None:
        fail("Workflow run has an invalid head SHA")

    changed_paths = (
        RECOVERY.changed_paths_for_push(root, head_sha)
        if run.get("event") == "push"
        else None
    )
    role = RECOVERY.role_from_run(run, changed_paths)
    relative = RECOVERY.STATE_ROOT / f"{role}.json"
    current_path = root / relative
    if current_path.is_symlink() or not current_path.is_file():
        fail(f"Current role state must be a regular file: {current_path}")
    current_bytes = current_path.read_bytes()
    current_state = RECOVERY.load_object(current_path, "current role state")
    run_bytes = RECOVERY.state_at_run(root, head_sha, relative)

    decision, updated = decide_cancelled_recovery(
        run,
        jobs,
        current_state,
        current_bytes,
        run_bytes,
        changed_paths=changed_paths,
    )

    arguments.decision.parent.mkdir(parents=True, exist_ok=True)
    arguments.decision.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        newline="\n",
    )
    arguments.state_output.parent.mkdir(parents=True, exist_ok=True)
    if updated is None:
        arguments.state_output.unlink(missing_ok=True)
    else:
        arguments.state_output.write_text(
            json.dumps(updated, indent=2) + "\n",
            newline="\n",
        )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
