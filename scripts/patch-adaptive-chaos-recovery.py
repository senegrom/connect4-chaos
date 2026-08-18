#!/usr/bin/env python3
"""Install the reviewed adaptive-preparation integration with strict anchors."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one replacement target, found {count}")
    path.write_text(text.replace(old, new, 1), newline="\n")


supervisor = ROOT / ".github/workflows/supervise-perfect-chaos-main.yml"
replace_once(
    supervisor,
    "    if: github.event_name != 'workflow_run' || github.event.workflow_run.head_branch == 'main'\n",
    """    if: >-
      github.event_name != 'workflow_run'
      || (
        github.event.workflow_run.head_branch == 'main'
        && (
          github.event.workflow_run.name != 'Continue Perfect Chaos 18-piece refinement on main'
          || github.event.workflow_run.conclusion != 'failure'
        )
      )
""",
    "failed-continuation supervisor exclusion",
)
replace_once(
    supervisor,
    """          python3 scripts/test-perfect-chaos-main-supervisor.py
          python3 scripts/test-perfect-chaos-main-active-runs.py
""",
    """          python3 scripts/test-perfect-chaos-main-supervisor.py
          python3 scripts/test-perfect-chaos-main-active-runs.py
          python3 scripts/test-perfect-chaos-main-recovery.py
""",
    "supervisor recovery tests",
)

ci = ROOT / ".github/workflows/ci.yml"
replace_once(
    ci,
    """          python3 scripts/test-perfect-chaos-main-supervisor.py
          python3 scripts/test-perfect-chaos-main-active-runs.py

      - name: Browser smoke
""",
    """          python3 scripts/test-perfect-chaos-main-supervisor.py
          python3 scripts/test-perfect-chaos-main-active-runs.py
          python3 scripts/test-perfect-chaos-main-recovery.py

      - name: Browser smoke
""",
    "ordinary CI recovery tests",
)

recovery = ROOT / "scripts/perfect-chaos-main-recovery.py"
replace_once(
    recovery,
    '    require_int(state["prepareShards"], "prepareShards", 1, 512)\n',
    '    require_int(state["prepareShards"], "prepareShards", 1, 256)\n',
    "recovery shard schema limit",
)
replace_once(
    recovery,
    """def next_profile(shards: int, workers: int) -> tuple[int, int] | None:
    if shards < 512:
        return min(512, max(shards + 1, shards * 2)), workers
    if workers > 1:
        return shards, max(1, workers // 2)
    return None
""",
    """def next_profile(shards: int, workers: int) -> tuple[int, int] | None:
    if shards < 256:
        return min(256, max(shards + 1, shards * 2)), workers
    if workers > 1:
        return shards, max(1, workers // 2)
    return None
""",
    "recovery profile ladder",
)
replace_once(
    recovery,
    '                "reason": "Preparation already uses 512 shards and one worker.",\n',
    '                "reason": "Preparation already uses 256 shards and one worker.",\n',
    "recovery exhaustion message",
)

recovery_tests = ROOT / "scripts/test-perfect-chaos-main-recovery.py"
replace_once(
    recovery_tests,
    """def test_shards_are_capped_before_workers_drop() -> None:
    decision, updated = decide("red", shards=400, workers=8)
    assert decision["newProfile"] == {"prepareShards": 512, "prepareWorkers": 8}
    assert updated is not None and updated["prepareShards"] == 512
""",
    """def test_shards_are_capped_before_workers_drop() -> None:
    decision, updated = decide("red", shards=200, workers=8)
    assert decision["newProfile"] == {"prepareShards": 256, "prepareWorkers": 8}
    assert updated is not None and updated["prepareShards"] == 256
""",
    "recovery shard-cap test",
)
replace_once(
    recovery_tests,
    """def test_workers_drop_only_after_maximum_sharding() -> None:
    decision, updated = decide("red", shards=512, workers=8)
    assert decision["newProfile"] == {"prepareShards": 512, "prepareWorkers": 4}
    assert updated is not None and updated["prepareWorkers"] == 4
""",
    """def test_workers_drop_only_after_maximum_sharding() -> None:
    decision, updated = decide("red", shards=256, workers=8)
    assert decision["newProfile"] == {"prepareShards": 256, "prepareWorkers": 4}
    assert updated is not None and updated["prepareWorkers"] == 4
""",
    "recovery worker-reduction test",
)
replace_once(
    recovery_tests,
    """def test_exhausted_profile_blocks_identical_retry() -> None:
    decision, updated = decide("yellow", shards=512, workers=1)
""",
    """def test_exhausted_profile_blocks_identical_retry() -> None:
    decision, updated = decide("yellow", shards=256, workers=1)
""",
    "recovery exhaustion test",
)

print({
    "supervisor": str(supervisor.relative_to(ROOT)),
    "ci": str(ci.relative_to(ROOT)),
    "recovery": str(recovery.relative_to(ROOT)),
    "recoveryTests": str(recovery_tests.relative_to(ROOT)),
    "status": "patched",
})
