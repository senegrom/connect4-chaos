#!/usr/bin/env python3
"""Install symmetric loss-rank checks in the exact Perfect Chaos W/D/L solver."""

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement, found {count}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1))


def main() -> None:
    solver = Path("scripts/perfect-chaos-wdl.py")
    tests = Path("scripts/test-perfect-chaos-wdl.py")
    crosscheck = Path("scripts/perfect-chaos-wdl-crosscheck.mjs")

    replace_once(
        solver,
        '''    all_chosen_actions_optimal = True\n    ranked_winning_progress_verified = True\n    draw_region_closed_verified = True\n''',
        '''    all_chosen_actions_optimal = True\n    ranked_winning_progress_verified = True\n    ranked_losing_delay_verified = True\n    draw_region_closed_verified = True\n''',
    )
    replace_once(
        solver,
        '''        elif values[index] == VALUES["draw"]:\n            if node.ai_turn:\n                if not any(value == VALUES["draw"] for value in edge_values):\n                    draw_region_closed_verified = False\n            elif min(edge_values) != VALUES["draw"]:\n                draw_region_closed_verified = False\n\n    if not all_chosen_actions_optimal:\n''',
        '''        elif values[index] == VALUES["loss"]:\n            losing_edge_ranks = [\n                edge_rank(edge) for edge in node.edges\n                if edge_value(edge) == VALUES["loss"]\n            ]\n            if node.ai_turn:\n                selected_edge = node.edges[best_edges[index]]\n                selected_rank = edge_rank(selected_edge)\n                if (\n                    len(losing_edge_ranks) != len(node.edges)\n                    or edge_value(selected_edge) != VALUES["loss"]\n                    or selected_rank != max(losing_edge_ranks)\n                    or ranks[index] != selected_rank + 1\n                ):\n                    ranked_losing_delay_verified = False\n            elif (\n                not losing_edge_ranks\n                or ranks[index] != min(losing_edge_ranks) + 1\n            ):\n                ranked_losing_delay_verified = False\n        elif values[index] == VALUES["draw"]:\n            if node.ai_turn:\n                if not any(value == VALUES["draw"] for value in edge_values):\n                    draw_region_closed_verified = False\n            elif min(edge_values) != VALUES["draw"]:\n                draw_region_closed_verified = False\n\n    if not all_chosen_actions_optimal:\n''',
    )
    replace_once(
        solver,
        '''    if not ranked_winning_progress_verified:\n        raise RuntimeError("Winning ranks do not make strict finite progress.")\n    if not draw_region_closed_verified:\n''',
        '''    if not ranked_winning_progress_verified:\n        raise RuntimeError("Winning ranks do not make strict finite progress.")\n    if not ranked_losing_delay_verified:\n        raise RuntimeError("Losing ranks do not preserve the longest exact delay.")\n    if not draw_region_closed_verified:\n''',
    )
    replace_once(
        solver,
        '''        "rankedWinningProgressVerified": ranked_winning_progress_verified,\n        "drawRegionClosedVerified": draw_region_closed_verified,\n''',
        '''        "rankedWinningProgressVerified": ranked_winning_progress_verified,\n        "rankedLosingDelayVerified": ranked_losing_delay_verified,\n        "drawRegionClosedVerified": draw_region_closed_verified,\n''',
    )

    replace_once(
        tests,
        '''        solution = solve(script, root, "exact-oracle-handoff", graph([\n''',
        '''        solution = solve(script, root, "delayed-forced-loss", graph([\n            {\n                "aiTurn": True,\n                "edges": [\n                    {"terminal": "loss", "action": {"type": "drop", "column": 0}},\n                    {"next": 1, "action": {"type": "drop", "column": 1}},\n                ],\n            },\n            {\n                "aiTurn": False,\n                "edges": [{"next": 2, "action": {"type": "flip"}}],\n            },\n            {\n                "aiTurn": True,\n                "edges": [{"terminal": "loss", "action": {"type": "drop", "column": 2}}],\n            },\n        ]))\n        if (\n            solution["rootValues"] != ["loss"]\n            or solution["ranks"][0] != 3\n            or policy_edge(solution, 0) != 1\n            or not solution["rankedLosingDelayVerified"]\n        ):\n            raise AssertionError("exact W/D/L did not choose the longest unavoidable-loss delay")\n\n        solution = solve(script, root, "exact-oracle-handoff", graph([\n''',
    )

    replace_once(
        crosscheck,
        '''    if (!solved.allChosenActionsOptimal\n        || !solved.rankedWinningProgressVerified\n        || !solved.drawRegionClosedVerified) {\n''',
        '''    if (!solved.allChosenActionsOptimal\n        || !solved.rankedWinningProgressVerified\n        || !solved.rankedLosingDelayVerified\n        || !solved.drawRegionClosedVerified) {\n''',
    )


if __name__ == "__main__":
    main()
