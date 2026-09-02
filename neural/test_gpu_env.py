"""Equivalence test: the GPU environment must reproduce neural/chaos_game.py
transition for transition - outcome codes, child planes, shapes and
heights - on random states across board shapes and both rule sets,
including the rotations that swap rows and columns.

Usage: python -m neural.test_gpu_env [device]
"""

from __future__ import annotations

import random
import sys

import torch

from .chaos_game import ACTION_INDEX, NOT_TERMINAL, State, empty_state, successors
from .gpu_env import BoardBatch, CANVAS, step

SHAPES = [
    (4, 4, 3, True), (4, 5, 4, True), (5, 7, 4, True), (6, 7, 4, True),
    (7, 6, 4, False), (6, 6, 4, False), (10, 10, 5, True), (5, 10, 4, True),
    (10, 5, 6, False), (8, 8, 4, True),
]


def random_states(rows, columns, connect, chaos, count, rng):
    states = []
    while len(states) < count:
        state = empty_state(rows, columns)
        for _ in range(rng.randrange(0, rows * columns + 6)):
            edges = [e for e in successors(state, connect, chaos) if e.terminal == NOT_TERMINAL]
            if not edges:
                break
            state = rng.choice(edges).child
        states.append(state)
    return states


def to_planes(state: State):
    mover = torch.zeros((CANVAS, CANVAS), dtype=torch.bool)
    opponent = torch.zeros((CANVAS, CANVAS), dtype=torch.bool)
    stride = state.rows + 1
    for column in range(state.columns):
        for row in range(state.rows):
            bit = 1 << (column * stride + row)
            if state.mover & bit:
                mover[row, column] = True
            elif state.opponent & bit:
                opponent[row, column] = True
    return mover, opponent


def main(device: str = "cpu") -> None:
    rng = random.Random(20260902)
    states, meta = [], []
    for rows, columns, connect, chaos in SHAPES:
        for state in random_states(rows, columns, connect, chaos, 30, rng):
            states.append(state)
            meta.append((connect, chaos))
    n = len(states)
    board = BoardBatch([s.rows for s in states], [s.columns for s in states],
                       [m[0] for m in meta], [m[1] for m in meta], device)
    for i, state in enumerate(states):
        mover, opponent = to_planes(state)
        board.mover[i], board.opponent[i] = mover.to(device), opponent.to(device)
        board.heights[i, :state.columns] = torch.tensor(state.heights, device=device)
        board.pieces[i] = state.pieces

    legal = board.legal().cpu()
    checked = mismatches = 0
    for i, state in enumerate(states):
        expected = {ACTION_INDEX[e.action] for e in successors(state, meta[i][0], meta[i][1])}
        actual = {a for a in range(13) if legal[i][a]}
        if expected != actual:
            mismatches += 1
            print(f"legal mismatch at state {i}: expected {sorted(expected)} got {sorted(actual)}")

    for action in range(13):
        child, outcome = step(board, torch.full((n,), action, dtype=torch.int64, device=device))
        outcome = outcome.cpu()
        for i, state in enumerate(states):
            connect, chaos = meta[i]
            edges = {ACTION_INDEX[e.action]: e for e in successors(state, connect, chaos)}
            if action not in edges:
                continue
            edge = edges[action]
            checked += 1
            if outcome[i].item() != edge.terminal:
                mismatches += 1
                print(f"outcome mismatch state {i} action {action}: gpu {outcome[i].item()} py {edge.terminal}")
                continue
            if edge.terminal != NOT_TERMINAL:
                continue
            ref_mover, ref_opponent = to_planes(edge.child)
            got_mover, got_opponent = child.mover[i].cpu(), child.opponent[i].cpu()
            shape_ok = (child.rows[i].item(), child.cols[i].item()) == (edge.child.rows, edge.child.columns)
            heights_ok = child.heights[i, :edge.child.columns].cpu().tolist() == list(edge.child.heights)
            if not (torch.equal(ref_mover, got_mover) and torch.equal(ref_opponent, got_opponent)
                    and shape_ok and heights_ok):
                mismatches += 1
                print(f"child mismatch state {i} action {action} "
                      f"(shape_ok={shape_ok} heights_ok={heights_ok})")
    print(f"states {n}, transitions checked {checked}, mismatches {mismatches}")
    if mismatches:
        raise SystemExit(1)
    print("GPU ENV EQUIVALENT")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cpu")
