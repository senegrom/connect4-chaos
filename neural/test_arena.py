"""The arena must be symmetric: which network is named first may not matter.

Every opening is played twice with the colours swapped, so a network against
itself scores exactly 50%. That alone left one asymmetry: the first game of
each pair picked its opening moves from the searches of whichever network
was to move, so A chose every first-player opening move and B every
second-player one, and swapping the names played different games. Here the
same seed with the networks named the other way round must play the same
games from the same positions, so every board's result mirrors exactly.

Usage: python -m neural.test_arena [cpu|cuda]   (CPU by default, as in CI)
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import torch

from . import arena
from .gpu_selfplay import _prepare_network, parse_shapes
from .model import PolicyValueNet

DEVICE = "cpu"
SHAPES = "4x4c3chaos,4x5c3classic,5x4c3chaos,5x5c4classic"
GAMES = 6
SIMS = 2


def network(seed):
    """A small random network, prepared for inference as a checkpoint is."""
    torch.manual_seed(seed)
    net = PolicyValueNet(8, 1, 8)
    return _prepare_network({"arch": (8, 1, 8), "model": net.state_dict()}, DEVICE)


def match(first, second, seed=3):
    tally, unfinished, _seconds, distinct, settled = arena.play(
        first, second, parse_shapes(SHAPES), GAMES, SIMS, seed, DEVICE)
    return tally, unfinished, distinct, settled


class ArenaSymmetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a, cls.b = network(11), network(12)

    def test_odd_game_counts_are_refused(self):
        with self.assertRaisesRegex(ValueError, "must be even"):
            arena.play(self.a, self.a, parse_shapes(SHAPES), 3, SIMS, 1, DEVICE)

    def test_a_network_against_itself_scores_exactly_half(self):
        tally, unfinished, distinct, settled = match(self.a, self.a)
        self.assertEqual(unfinished, 0)
        for board in distinct:
            wins, draws, losses = tally.get(board, (0, 0, 0))
            with self.subTest(board=board):
                self.assertEqual(wins, losses)
                # A pair over inside the opening is not scored: both of its
                # games played the same moves, whoever was playing.
                self.assertEqual(wins + draws + losses + 2 * settled.get(board, 0), GAMES)
                # The twin replays its pair's opening, so at most one per pair.
                self.assertLessEqual(distinct[board], GAMES // 2)
        self.assertTrue(settled, 'Connect 3 on these boards often ends inside eight plies')

    def test_naming_the_networks_the_other_way_round_mirrors_every_board(self):
        forward, unfinished, distinct, settled = match(self.a, self.b)
        backward, unfinished_back, distinct_back, settled_back = match(self.b, self.a)
        self.assertEqual((unfinished, unfinished_back), (0, 0))
        self.assertEqual(distinct, distinct_back)
        self.assertEqual(settled, settled_back)
        self.assertEqual(set(forward), set(backward))
        for board, (wins, draws, losses) in forward.items():
            with self.subTest(board=board):
                self.assertEqual(backward[board], [losses, draws, wins])
        # Not a tie by construction: the pair diverges once the opening ends.
        self.assertTrue(any(wins != losses for wins, _draws, losses in forward.values()),
                        forward)

    def test_a_pair_is_scored_whole_or_not_at_all(self):
        picks = [(4, 4, 3, True)] * 8
        opening = [[3, 2, 1, 0, 3, 2, 1, 0]] * 8
        # Finished past the opening; one game cut off; both cut off; over
        # inside the opening.
        result = [1, -1, 1, 9, 9, 9, 0, 0]
        ended = [20, 21, 20, arena.MAX_PLIES, arena.MAX_PLIES, arena.MAX_PLIES, 4, 4]
        tally, settled, distinct = arena.score_pairs(picks, result, ended, opening)
        self.assertEqual(tally, {"4x4c3chaos": [1, 0, 1]}, "the cut-off pair's finished win is not counted")
        self.assertEqual(settled, {"4x4c3chaos": 1})
        self.assertEqual(distinct, {"4x4c3chaos": 1})

    def test_games_cut_off_by_the_ply_limit_leave_their_pairs_unscored(self):
        with patch.object(arena, "MAX_PLIES", 12):
            tally, unfinished, _distinct, settled = match(self.a, self.b)
        self.assertGreater(unfinished, 0)
        for board, (wins, draws, losses) in tally.items():
            with self.subTest(board=board):
                self.assertEqual((wins + draws + losses) % 2, 0, "only whole pairs are scored")
                self.assertLessEqual(wins + draws + losses + 2 * settled.get(board, 0), GAMES)
        scored = sum(map(sum, tally.values())) + 2 * sum(settled.values())
        self.assertLess(scored, GAMES * len(parse_shapes(SHAPES)))

    def test_the_seed_decides_the_match(self):
        self.assertEqual(match(self.a, self.b, seed=3), match(self.a, self.b, seed=3))

    def test_a_match_turns_cudnn_autotuning_off(self):
        # Autotuning picks kernels by timing them, so two processes could
        # play the same match differently; gpu_selfplay turns it on.
        torch.backends.cudnn.benchmark = True
        match(self.a, self.a)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("cpu", "cuda"):
        DEVICE = sys.argv.pop(1)
    unittest.main()
