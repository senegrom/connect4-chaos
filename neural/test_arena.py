"""The arena must be symmetric: which network is named first may not matter.

Every opening is played twice with the colours swapped, so a network against
itself scores exactly 50%. That alone left one asymmetry: the first game of
each pair picked its opening moves from the searches of whichever network
was to move, so A chose every first-player opening move and B every
second-player one, and swapping the names played different games. Here the
same seed with the networks named the other way round must play the same
games from the same positions, so every board's result mirrors exactly.

Symmetry says nothing about which side a score belongs to: every check of it
still holds with each result inverted. A search against its own network's
raw policy pins that, and the search must see each game's history.

Usage: python -m neural.test_arena [cpu|cuda]   (CPU by default, as in CI)
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import torch

from . import arena
from .gpu_env import FLIP
from .gpu_history import history_counts
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

    def test_a_search_outscores_its_own_raw_policy(self):
        # The search sees the immediate wins and losses the policy walks
        # into, so A, searching, wins most games against B, which only asks
        # the policy (sims_b=0). With A's results scored the wrong way round,
        # a weaker generation would read as the stronger one. Four opening
        # plies leave most of each game to the search: 92% here, and 83% to
        # 92% on seeds 1 to 8. With the eight the loop draws, the openings
        # decide most of these small games and the score ran from 62.5% to 100%.
        with patch.object(arena, "OPENING_PLIES", 4):
            tally, unfinished, _seconds, distinct, settled = arena.play(
                self.a, self.a, parse_shapes("4x4c3classic,5x5c4classic"), 6, 16, 3, DEVICE, sims_b=0)
        overall, text = arena.report(tally, unfinished, 0.0, "search", "policy", distinct, settled)
        self.assertGreater(overall, 0.7, text)

    def test_each_move_is_chosen_with_its_own_games_history(self):
        # The stand-ins flip wherever they can and otherwise drop in the
        # lowest open column. A Chaos game then comes back to a position every
        # second ply, so its era grows by one a ply and the third occurrence
        # ends it at ply 3; a classic game only drops, and a drop starts a new
        # era. Handed no history, or another game's, the search could not see
        # that a flip draws by repetition.
        calls, plies = [], []
        real_step = arena.step

        def step(board, action, **kwargs):
            plies.append(len(board))
            return real_step(board, action, **kwargs)

        def move(kind, board, rep1, rep2, side, history, keys):
            calls.append((kind, len(plies), board.chaos.clone(), rep1.clone(), rep2.clone(),
                          None if history is None else history.lengths.clone(),
                          history_counts(history, board.position_hash(keys, side))))
            legal = board.legal()
            lowest = legal.long().argmax(dim=1)
            return torch.where(legal[:, FLIP], torch.full_like(lowest, FLIP), lowest)

        def opened(_sides, board, rep1, rep2, side, history, keys, _generator):
            return move("open", board, rep1, rep2, side, history, keys)

        def chosen(_net, board, rep1, rep2, _sims, side, history, keys):
            return move("choose", board, rep1, rep2, side, history, keys)

        # Two opening plies, so the openings are drawn and the rest chosen.
        with patch.object(arena, "OPENING_PLIES", 2), patch.object(arena, "step", step), \
                patch.object(arena, "_open", opened), patch.object(arena, "_choose", chosen):
            arena.play(None, None, parse_shapes("4x4c3chaos,4x4c3classic"), 4, SIMS, 3, DEVICE)
        self.assertEqual({call[0] for call in calls}, {"open", "choose"})
        for kind, ply, chaos, rep1, rep2, lengths, counts in calls:
            with self.subTest(kind=kind, ply=ply):
                self.assertIsNotNone(lengths, "the search was handed no history")
                self.assertEqual(lengths.tolist(), (chaos.long() * ply).tolist())
                # From ply 2 on, a Chaos game stands where it stood two plies
                # before: once in its history, and shown as repeated once.
                again = chaos & (ply >= 2)
                self.assertEqual(counts.tolist(), again.long().tolist())
                self.assertEqual(rep1.tolist(), again.tolist())
                self.assertFalse(bool(rep2.any()))
        # Moves chosen after the opening met repeated positions too.
        self.assertTrue(any(kind == "choose" and bool(rep1.any())
                            for kind, _ply, _chaos, rep1, *_rest in calls))

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
