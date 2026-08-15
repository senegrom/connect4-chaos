#!/usr/bin/env python3
"""Exact fixed-role W/D/L retrograde solver for closed Perfect Chaos graphs.

The existing layered prefix solver proves non-loss. This module adds the
missing objective layer: at AI nodes choose win > draw > loss, while opponent
nodes choose loss < draw < win from the AI's perspective. Unresolved closed
cycles become draws, matching the automatic threefold rule for a fresh root.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GRAPH_FORMAT = "connect4-chaos-closed-wdl-graph-v1"
SOLUTION_FORMAT = "connect4-chaos-exact-wdl-solution-v1"
OBJECTIVE = "maximize-win-then-draw-then-loss"
VALUES = {"loss": -1, "draw": 0, "win": 1}
VALUE_NAMES = {-1: "loss", 0: "draw", 1: "win"}
UNKNOWN = 2


@dataclass(frozen=True)
class Edge:
    next_node: int | None
    value: int | None
    action: Any
    source: str


@dataclass(frozen=True)
class Node:
    ai_turn: bool
    edges: tuple[Edge, ...]


def require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{field} must be an object.")
    return value


def require_index(value: Any, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < maximum:
        raise RuntimeError(f"{field} must be an integer in [0, {maximum}).")
    return value


def parse_action(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        raise RuntimeError(f"{field} is required.")
    action = require_object(value, field)
    action_type = action.get("type")
    if action_type not in {"drop", "flip", "rotateCW", "rotateCCW"}:
        raise RuntimeError(f"{field}.type is invalid.")
    expected = {"type", "column"} if action_type == "drop" else {"type"}
    if set(action) != expected:
        raise RuntimeError(f"{field} has unexpected or missing fields.")
    if action_type == "drop":
        column = action["column"]
        if isinstance(column, bool) or not isinstance(column, int) or column < 0:
            raise RuntimeError(f"{field}.column must be a non-negative integer.")
    return dict(action)


def parse_graph(path: Path) -> tuple[dict[str, Any], list[Node], list[int]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("The W/D/L graph must be a regular, non-symlink JSON file.")
    try:
        graph = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not parse the W/D/L graph: {error}") from error
    graph = require_object(graph, "graph")
    if graph.get("format") != GRAPH_FORMAT:
        raise RuntimeError("Unsupported W/D/L graph format.")
    if graph.get("objective") != OBJECTIVE:
        raise RuntimeError("The W/D/L graph has the wrong optimisation objective.")
    role = graph.get("role")
    if role not in {"red", "yellow"}:
        raise RuntimeError("graph.role must be red or yellow.")

    raw_nodes = graph.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise RuntimeError("graph.nodes must be a non-empty list.")
    node_count = len(raw_nodes)
    nodes: list[Node] = []
    for node_index, raw_node in enumerate(raw_nodes):
        record = require_object(raw_node, f"graph.nodes[{node_index}]")
        if not isinstance(record.get("aiTurn"), bool):
            raise RuntimeError(f"graph.nodes[{node_index}].aiTurn must be boolean.")
        raw_edges = record.get("edges")
        if not isinstance(raw_edges, list) or not raw_edges:
            raise RuntimeError(f"graph.nodes[{node_index}].edges must be non-empty.")
        edges: list[Edge] = []
        seen_actions: set[str] = set()
        for edge_index, raw_edge in enumerate(raw_edges):
            edge = require_object(raw_edge, f"graph.nodes[{node_index}].edges[{edge_index}]")
            has_next = "next" in edge
            has_terminal = "terminal" in edge
            has_oracle = "oracle" in edge
            if sum((has_next, has_terminal, has_oracle)) != 1:
                raise RuntimeError(
                    f"graph.nodes[{node_index}].edges[{edge_index}] must contain exactly one "
                    "of next, terminal, or oracle."
                )
            action = parse_action(
                edge.get("action"),
                f"graph.nodes[{node_index}].edges[{edge_index}].action",
            )
            if has_next:
                next_node = require_index(
                    edge.get("next"),
                    f"graph.nodes[{node_index}].edges[{edge_index}].next",
                    node_count,
                )
                parsed = Edge(next_node, None, action, "graph")
            else:
                field = "terminal" if has_terminal else "oracle"
                name = edge.get(field)
                if name not in VALUES:
                    raise RuntimeError(
                        f"graph.nodes[{node_index}].edges[{edge_index}].{field} is invalid."
                    )
                parsed = Edge(None, VALUES[name], action, field)
            action_identity = json.dumps(action, sort_keys=True, separators=(",", ":"))
            if action_identity in seen_actions:
                raise RuntimeError(f"graph.nodes[{node_index}] contains a duplicate action.")
            seen_actions.add(action_identity)
            edges.append(parsed)
        nodes.append(Node(record["aiTurn"], tuple(edges)))

    for node_index, node in enumerate(nodes):
        for edge_index, edge in enumerate(node.edges):
            if edge.next_node is not None and nodes[edge.next_node].ai_turn == node.ai_turn:
                raise RuntimeError(
                    f"graph.nodes[{node_index}].edges[{edge_index}] must alternate the side to move."
                )

    raw_roots = graph.get("roots")
    if not isinstance(raw_roots, list) or not raw_roots:
        raise RuntimeError("graph.roots must be a non-empty list.")
    roots = [require_index(value, f"graph.roots[{index}]", node_count)
             for index, value in enumerate(raw_roots)]
    if len(set(roots)) != len(roots):
        raise RuntimeError("graph.roots contains duplicates.")
    return graph, nodes, roots


def solve_nodes(nodes: list[Node], roots: list[int]) -> dict[str, Any]:
    count = len(nodes)
    predecessors: list[list[tuple[int, int]]] = [[] for _ in nodes]
    for parent, node in enumerate(nodes):
        for edge_index, edge in enumerate(node.edges):
            if edge.next_node is not None:
                predecessors[edge.next_node].append((parent, edge_index))

    values = [UNKNOWN] * count
    ranks = [0] * count
    best_edges = [-1] * count
    decisive_counts = [0] * count
    maximum_child_rank = [0] * count
    queue: list[tuple[int, int]] = []

    def mark(index: int, value: int, rank: int, edge: int) -> None:
        if values[index] != UNKNOWN:
            return
        values[index] = value
        ranks[index] = rank
        best_edges[index] = edge
        heapq.heappush(queue, (rank, index))

    for index, node in enumerate(nodes):
        immediate_win = -1
        immediate_loss = -1
        forced_count = 0
        for edge_index, edge in enumerate(node.edges):
            if edge.value is None:
                continue
            if node.ai_turn:
                if edge.value == VALUES["win"] and immediate_win < 0:
                    immediate_win = edge_index
                if edge.value == VALUES["loss"]:
                    forced_count += 1
                    immediate_loss = edge_index
            else:
                if edge.value == VALUES["loss"] and immediate_loss < 0:
                    immediate_loss = edge_index
                if edge.value == VALUES["win"]:
                    forced_count += 1
                    immediate_win = edge_index
        decisive_counts[index] = forced_count
        if node.ai_turn:
            if immediate_win >= 0:
                mark(index, VALUES["win"], 1, immediate_win)
            elif forced_count == len(node.edges):
                mark(index, VALUES["loss"], 1, immediate_loss)
        else:
            if immediate_loss >= 0:
                mark(index, VALUES["loss"], 1, immediate_loss)
            elif forced_count == len(node.edges):
                mark(index, VALUES["win"], 1, immediate_win)

    while queue:
        rank, child = heapq.heappop(queue)
        if ranks[child] != rank:
            continue
        child_value = values[child]
        for parent, edge_index in predecessors[child]:
            if values[parent] != UNKNOWN:
                continue
            node = nodes[parent]
            if node.ai_turn:
                if child_value == VALUES["win"]:
                    mark(parent, VALUES["win"], rank + 1, edge_index)
                elif child_value == VALUES["loss"]:
                    decisive_counts[parent] += 1
                    if rank >= maximum_child_rank[parent]:
                        maximum_child_rank[parent] = rank
                        best_edges[parent] = edge_index
                    if decisive_counts[parent] == len(node.edges):
                        mark(
                            parent,
                            VALUES["loss"],
                            maximum_child_rank[parent] + 1,
                            best_edges[parent],
                        )
            else:
                if child_value == VALUES["loss"]:
                    mark(parent, VALUES["loss"], rank + 1, edge_index)
                elif child_value == VALUES["win"]:
                    decisive_counts[parent] += 1
                    if rank >= maximum_child_rank[parent]:
                        maximum_child_rank[parent] = rank
                        best_edges[parent] = edge_index
                    if decisive_counts[parent] == len(node.edges):
                        mark(
                            parent,
                            VALUES["win"],
                            maximum_child_rank[parent] + 1,
                            best_edges[parent],
                        )

    for index, value in enumerate(values):
        if value == UNKNOWN:
            values[index] = VALUES["draw"]

    def edge_value(edge: Edge) -> int:
        return edge.value if edge.value is not None else values[edge.next_node]

    def edge_rank(edge: Edge) -> int:
        return 0 if edge.value is not None else ranks[edge.next_node]

    for index, node in enumerate(nodes):
        if not node.ai_turn:
            continue
        target = values[index]
        candidates = [
            edge_index for edge_index, edge in enumerate(node.edges)
            if edge_value(edge) == target
        ]
        if not candidates:
            raise RuntimeError(f"AI node {index} has no action attaining its exact W/D/L value.")
        if target == VALUES["win"]:
            candidates.sort(key=lambda edge_index: (
                0 if node.edges[edge_index].value == VALUES["win"] else
                ranks[node.edges[edge_index].next_node] + 1,
                edge_index,
            ))
        elif target == VALUES["loss"]:
            candidates.sort(key=lambda edge_index: (
                -(0 if node.edges[edge_index].value == VALUES["loss"] else
                  ranks[node.edges[edge_index].next_node] + 1),
                edge_index,
            ))
        best_edges[index] = candidates[0]

    policy = []
    for index, node in enumerate(nodes):
        if not node.ai_turn:
            continue
        edge_index = best_edges[index]
        if edge_index < 0:
            raise RuntimeError(f"AI node {index} has no exact policy action.")
        edge = node.edges[edge_index]
        if edge_value(edge) != values[index]:
            raise RuntimeError(f"AI node {index} selected a suboptimal action.")
        policy.append({
            "node": index,
            "edge": edge_index,
            "action": edge.action,
            "value": VALUE_NAMES[values[index]],
            "rank": ranks[index],
        })

    all_chosen_actions_optimal = True
    ranked_winning_progress_verified = True
    draw_region_closed_verified = True
    for index, node in enumerate(nodes):
        edge_values = [edge_value(edge) for edge in node.edges]
        if node.ai_turn:
            selected = best_edges[index]
            if selected < 0 or edge_values[selected] != max(edge_values):
                all_chosen_actions_optimal = False
        if values[index] == VALUES["win"]:
            if node.ai_turn:
                selected_edge = node.edges[best_edges[index]]
                if edge_value(selected_edge) != VALUES["win"] or edge_rank(selected_edge) >= ranks[index]:
                    ranked_winning_progress_verified = False
            elif any(
                edge_value(edge) != VALUES["win"] or edge_rank(edge) >= ranks[index]
                for edge in node.edges
            ):
                ranked_winning_progress_verified = False
        elif values[index] == VALUES["draw"]:
            if node.ai_turn:
                if not any(value == VALUES["draw"] for value in edge_values):
                    draw_region_closed_verified = False
            elif min(edge_values) != VALUES["draw"]:
                draw_region_closed_verified = False

    if not all_chosen_actions_optimal:
        raise RuntimeError("The emitted AI policy contains a suboptimal action.")
    if not ranked_winning_progress_verified:
        raise RuntimeError("Winning ranks do not make strict finite progress.")
    if not draw_region_closed_verified:
        raise RuntimeError("The unresolved draw region is not closed under optimal play.")

    counts = {name: sum(value == code for value in values) for name, code in VALUES.items()}
    root_values = [VALUE_NAMES[values[index]] for index in roots]
    return {
        "values": [VALUE_NAMES[value] for value in values],
        "ranks": ranks,
        "policy": policy,
        "rootValues": root_values,
        "counts": counts,
        "allChosenActionsOptimal": all_chosen_actions_optimal,
        "rankedWinningProgressVerified": ranked_winning_progress_verified,
        "drawRegionClosedVerified": draw_region_closed_verified,
    }


def solve_file(input_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    graph, nodes, roots = parse_graph(input_path)
    graph_bytes = input_path.read_bytes()
    solved = solve_nodes(nodes, roots)
    graph_edges = sum(edge.next_node is not None for node in nodes for edge in node.edges)
    terminal_edges = sum(edge.source == "terminal" for node in nodes for edge in node.edges)
    oracle_edges = sum(edge.source == "oracle" for node in nodes for edge in node.edges)
    result = {
        "format": SOLUTION_FORMAT,
        "objective": OBJECTIVE,
        "role": graph["role"],
        "sourceGraph": {
            "sha256": hashlib.sha256(graph_bytes).hexdigest(),
            "nodes": len(nodes),
            "edges": graph_edges + terminal_edges + oracle_edges,
            "graphEdges": graph_edges,
            "terminalEdges": terminal_edges,
            "oracleEdges": oracle_edges,
        },
        "roots": roots,
        **solved,
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("solve",))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = solve_file(args.input, args.output)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
