"""Validates the Python table stack against solved boards: the game value
equation must hold at every sampled state through this decoder, and the
empty board must carry the recorded root value.

Usage: python -m neural.validate_tables <dir> <rows> <columns> <connect> [samples]
"""

from __future__ import annotations

import random
import sys

from .chaos_game import DRAW, LOSS, WIN, empty_state, successors
from .pair_tables import PairTable, canonical_pair_slot, pair_of


def validate(directory: str, rows: int, columns: int, connect: int,
             samples: int = 400, seed: int = 20260901) -> dict:
    table = PairTable(directory, rows, columns, connect)
    rng = random.Random(seed)

    root = empty_state(rows, columns)
    root_value = table.value_of(root)

    tallies = {WIN: 0, DRAW: 0, LOSS: 0}
    for _ in range(samples):
        state, value = table.sample_state(rng)
        any_win = False
        all_loss = True
        for edge in successors(state, connect, chaos=True):
            for_mover = table.edge_value_for_mover(edge)
            if for_mover == WIN:
                any_win = True
            if for_mover != LOSS:
                all_loss = False
        expected = WIN if any_win else (LOSS if all_loss else DRAW)
        if value != expected:
            raise AssertionError(
                f"value equation failed at pieces={state.pieces} "
                f"pair={pair_of(state.pieces, state.mover_count)} "
                f"slot={canonical_pair_slot(table.geometry, state, pair_of(state.pieces, state.mover_count))}: "
                f"table={value} expected={expected}")
        tallies[value] += 1
    return {"root": root_value, "checked": samples, "tallies": tallies}


if __name__ == "__main__":
    directory, rows, columns, connect = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    samples = int(sys.argv[5]) if len(sys.argv) > 5 else 400
    report = validate(directory, rows, columns, connect, samples)
    print(f"{rows}x{columns} c{connect}: root={report['root']} "
          f"checked={report['checked']} tallies={report['tallies']} ALL CONSISTENT")
