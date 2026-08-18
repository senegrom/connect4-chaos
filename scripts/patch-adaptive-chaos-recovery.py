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

print({
    "supervisor": str(supervisor.relative_to(ROOT)),
    "ci": str(ci.relative_to(ROOT)),
    "status": "patched",
})
