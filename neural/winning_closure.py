"""Measures the size of a first-player winning-strategy closure on a solved
board: from the root, the winner commits to ONE winning move per state
(immediate wins first, then winning drops, then winning transforms) while
every reply of the loser is explored. The closure is exactly what a
winning-strategy certificate would have to contain, so its size on 5x7
c4 (a known first-player win) calibrates whether a certificate search
could settle 6x7 without full enumeration.

Also reports how often the winner had to rely on a transform-only win -
the cases where values alone cannot guarantee progress and a settling
round (or search) would be needed to certify termination.

Usage: python -m neural.winning_closure <dir> <rows> <cols> <connect> [cap]
"""

from __future__ import annotations

import sys
import time
from collections import deque

from .chaos_game import LOSS, NOT_TERMINAL, WIN, empty_state, successors
from .pair_tables import PairTable, canonical_pair_slot, pair_of


def state_key(table, state):
    pair_id = pair_of(state.pieces, state.mover_count)
    slot = canonical_pair_slot(table.geometry, state, pair_id)
    return (state.pieces << 52) | (pair_id << 44) | slot


def measure(directory, rows, columns, connect, cap):
    table = PairTable(directory, rows, columns, connect, chaos=True)
    root = empty_state(rows, columns)
    if table.value_of(root) != WIN:
        raise SystemExit("root is not a first-player win")

    seen = {state_key(table, root)}
    frontier = deque([(root, True)])         # (state, winner_to_move)
    winner_states = loser_states = 0
    transform_only = immediate = 0
    started = time.time()
    while frontier:
        state, winner_turn = frontier.popleft()
        edges = successors(state, connect, chaos=True)
        if winner_turn:
            winner_states += 1
            wins = [e for e in edges if table.edge_value_for_mover(e) == WIN]
            terminal = [e for e in wins if e.terminal != NOT_TERMINAL]
            if terminal:
                immediate += 1
                continue
            drops = [e for e in wins if not e.same_layer]
            chosen = drops[0] if drops else wins[0]
            if not drops:
                transform_only += 1
            key = state_key(table, chosen.child)
            if key not in seen:
                seen.add(key)
                frontier.append((chosen.child, False))
        else:
            loser_states += 1
            for edge in edges:
                if edge.terminal != NOT_TERMINAL:
                    continue                   # a loser's terminal edge is a lost line
                key = state_key(table, edge.child)
                if key not in seen:
                    seen.add(key)
                    frontier.append((edge.child, True))
        total = winner_states + loser_states
        if total % 200_000 == 0:
            print(f"closure so far: {total} states ({winner_states} winner-to-move, "
                  f"{loser_states} loser-to-move), frontier {len(frontier)}, "
                  f"{time.time() - started:.0f}s", flush=True)
        if total >= cap:
            print(f"CAP REACHED at {total} states; frontier still {len(frontier)}", flush=True)
            break
    print(f"closure: {winner_states + loser_states} states "
          f"({winner_states} winner-to-move, {loser_states} loser-to-move); "
          f"immediate wins {immediate}; transform-only wins {transform_only}; "
          f"{time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    directory, rows, columns, connect = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    cap = int(sys.argv[5]) if len(sys.argv) > 5 else 20_000_000
    measure(directory, rows, columns, connect, cap)
