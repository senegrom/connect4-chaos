"""Review regressions using the real CPU tensor and helper implementations."""
from contextlib import ExitStack, redirect_stdout
import io
import os
from pathlib import Path
import random
import re
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import torch
from .training_config import DEFAULT_SIMS, validate_selfplay
from .data_split import SPLIT_VERSION, validation_mask
from .distill import load_shards, without_heldout_positions
from .gpu_env import FLIP, NOT_TERMINAL, ROT_CCW, ROT_CW, BoardBatch, step
from .gpu_mcts import improved_policy, sample_actions, search_root
from .model import PolicyValueNet
from .search_quality import blunder_rate
from .test_support import function

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def shard(shapes, replay=False, split='train'):
    """A shard in the one format there is: uint8 planes scaled by 10, a split
    of this version - declared by an exact shard, a row mask in replay - and
    Q targets (all unknown here)."""
    n = len(shapes)
    planes = torch.zeros(n, 7, 10, 10, dtype=torch.uint8)
    legal = torch.zeros(n, 13, dtype=torch.bool)
    for i, (r, c, k, chaos) in enumerate(shapes):
        planes[i, 2, :r, :c] = 10
        planes[i, 3] = k
        planes[i, 4] = 10 * chaos
        legal[i, :c] = True
        legal[i, 10:] = chaos
    policy = torch.zeros(n, 13)
    policy[:, 0] = 1
    result = dict(planes=planes, planes_scale=10, legal=legal, policy=policy,
                  wdl=torch.ones(n, dtype=torch.long), q=torch.full((n, 13), 3, dtype=torch.long),
                  config=shapes[0][:3], split_version=SPLIT_VERSION)
    if replay:
        result.update(source='selfplay', validation=validation_mask(planes))
    else:
        result['split'] = split
    return result


class Net:
    def __call__(self, planes, legal):
        n = len(planes)
        return (torch.zeros(n, 13), torch.zeros(n, 3), torch.zeros(n, 13, 3))


class ReviewTests(unittest.TestCase):
    def test_defaults_and_invalid_configuration(self):
        validate_selfplay(10, DEFAULT_SIMS)
        ps = (ROOT / 'scripts/launch-modal-loop.ps1').read_text()
        self.assertEqual(int(re.search(r'\$Sims = (\d+)', ps)[1]), DEFAULT_SIMS)
        # The launcher hands the driver its exact corpus as the 21st argument.
        self.assertEqual(re.search(r"\[string\]\$ExactSubdir = '([^']+)'", ps)[1], 'datasets-v3')
        launched = re.search(r"\$args = @\((.*)\)", ps)[1].split(', ')
        self.assertEqual(launched[:2], ["'-m'", "'neural.modal_loop'"])
        self.assertEqual(launched.index('$ExactSubdir') - 1, 21)
        for games, sims, shapes, targets, share in [(0, 128, 'all', 0, .25), (1, 0, 'all', 0, .25),
                (1, 128, '12x4c4chaos', 0, .25), (1, 128, 'all', -1, .25),
                (1, 128, 'all', 0, float('nan'))]:
            with self.assertRaises(ValueError): validate_selfplay(games, sims, shapes, targets, share)
        # The random openings: a share of the games, and at most this many plies.
        validate_selfplay(1, 128, 'all', 0, .25, 0, 0)
        for random_share, random_plies in [(-.1, 4), (1.5, 4), (float('nan'), 4), (.5, -1), (.5, 1.5), (.5, True)]:
            with self.subTest(random_share=random_share, random_plies=random_plies), \
                    self.assertRaisesRegex(ValueError, 'random_'):
                validate_selfplay(1, 128, 'all', 0, .25, random_share, random_plies)

    def test_late_checkpoint_read_never_changes_new_pointer(self):
        old_started, release, finished = threading.Event(), threading.Event(), threading.Event()
        class Volume:
            def read_file(self, name):
                if name.endswith('old.pt'):
                    old_started.set()
                    release.wait(2)
                    finished.set()
                return [name.encode()]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = dict(Path=Path, ROOT=root, MODELS=root/'models', vol=Volume(), threading=threading)
            for name in ('with_timeout', 'read_model', 'mirror_model'):
                function(ROOT / 'neural/modal_loop.py', name, env)
            deadline = env['with_timeout']
            env['with_timeout'] = lambda _seconds, work, *args: deadline(.02, work, *args)
            try:
                with self.assertRaises(TimeoutError): env['mirror_model']('old.pt')
                self.assertTrue(old_started.is_set())
                env['mirror_model']('new.pt')
                release.set()
                self.assertTrue(finished.wait(2))
                self.assertEqual(Path((root/'current-model.txt').read_text().strip()).name, 'new.pt')
                self.assertFalse((root/'models'/'old.pt').exists())
            finally: release.set()

    def test_mixed_replay_excludes_exact_and_rotated_heldout_boards(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,
                DISTILL_HOLDOUT_CONFIGS='6x6c4classic,4x6c4chaos'):
            root = Path(temp)
            torch.save(shard([(6,6,4,False)], split='validation'), root/'6x6c4classic-0000.pt')
            torch.save(shard([(6,6,4,False)]), root/'6x6c4classic-0001.pt')
            torch.save(shard([(6,6,4,False),(6,4,4,True),(4,6,4,True),(5,5,4,False)], True), root/'gpu-sp-1.pt')
            train, held = load_shards(root)
            self.assertEqual(sum(len(s['wdl']) for s in train), 1)
            self.assertEqual(len(held), 1)
            self.assertEqual(int((train[0]['planes'][0,2,:,0] > 0).sum()), 5)

    def test_a_shard_of_an_older_format_is_refused_by_name(self):
        # Float planes, a missing split, Q targets or replay mask: every
        # shard written that way went with the Volume's wipes, and the
        # readers that converted them are gone.
        exact, replay = shard([(5,5,4,False)]), shard([(5,5,4,False)], True)
        planes = exact['planes'].float() / 10
        for data, problem in ((dict(exact, planes=planes), 'planes that are not uint8'),
                              (dict(exact, planes_scale=1), 'planes that are not uint8'),
                              ({k: v for k, v in exact.items() if k != 'split_version'}, 'no split'),
                              ({k: v for k, v in exact.items() if k != 'q'}, 'no Q targets'),
                              ({k: v for k, v in replay.items() if k != 'validation'}, 'no validation mask')):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as temp, \
                    patch.dict(os.environ, DISTILL_HOLDOUT_CONFIGS=''):
                torch.save(data, Path(temp)/'gpu-sp-1.pt' if data.get('source') else Path(temp)/'5x5c4classic-0001.pt')
                with self.assertRaisesRegex(ValueError, f'predates the current shard format \\({problem}'):
                    load_shards(temp)

    def test_replay_window_filters_only_needed_newest_shards(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,
                DISTILL_HOLDOUT_CONFIGS='6x6c4classic', DISTILL_REPLAY_WINDOW='1'):
            root = Path(temp)
            for index in range(3):
                path = root/f'gpu-sp-{index}.pt'
                torch.save(shard([(6,6,4,False),(5,5,4,False)], True), path)
                os.utime(path, (index + 1, index + 1))
            with patch('neural.distill.without_heldout_positions', wraps=without_heldout_positions) as filter_shard:
                train, held = load_shards(root)
            self.assertEqual(filter_shard.call_count, 1)
            self.assertEqual(sum(len(s['wdl']) for s in train), 1)
            self.assertEqual(train[0]['mtime'], 3)
            self.assertFalse(held)

    def test_empty_training_split_never_promotes_holdout(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, DISTILL_HOLDOUT_CONFIGS='6x6c4classic'):
            root = Path(temp)
            torch.save(shard([(6,6,4,False)], split='validation'), root/'6x6c4classic-0000.pt')
            with self.assertRaisesRegex(ValueError, 'No training positions'): load_shards(root)

    def test_measurement_bounds_every_tensor(self):
        data = shard([(4,4,4,False)] * 600)
        for limit in (1, 127, 511, 512, 513, 599, 600, 1000):
            with self.subTest(limit=limit):
                rate, count = blunder_rate(Net(), data, 0, limit, 'cpu')
                self.assertEqual(count, min(limit, 600))
                self.assertEqual(rate, 0)
        for limit in (0, -1, 1.2, True):
            with self.assertRaises(ValueError): blunder_rate(Net(), data, 0, limit, 'cpu')
        rate, count = blunder_rate(Net(), data, 2, 3, 'cpu')
        self.assertEqual(count, 3)


def legal_choices(board, choice):
    return bool(board.legal().gather(1, choice[:, None]).all())


def selfplay(shapes, games=8, **settings):
    """The shard of one CPU self-play run of a tiny random network, with
    gpu_selfplay's module settings replaced by `settings` (SIMS 2 unless
    given). The CPU is chosen here rather than by torch.cuda.is_available():
    a test that runs with a cleared environment lets torch see a GPU that
    CUDA_VISIBLE_DEVICES hid, for the rest of the process."""
    from . import gpu_selfplay
    with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
        model = Path(temp) / 'tiny.pt'
        torch.manual_seed(0)
        torch.save({'model': PolicyValueNet(4, 1, 4).state_dict(), 'arch': (4, 1, 4)}, model)
        for name, value in {'SIMS': 2, **settings}.items():
            stack.enter_context(patch.object(gpu_selfplay, name, value))
        stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
        stack.enter_context(redirect_stdout(io.StringIO()))
        gpu_selfplay.run(str(model), temp, games, shapes, seed=5)
        return torch.load(next(Path(temp).glob('gpu-sp-*.pt')), weights_only=True)


class LegalityTests(unittest.TestCase):
    def test_sampling_never_picks_an_illegal_action(self):
        legal = torch.zeros(2, 13, dtype=torch.bool)
        legal[:, :2] = True
        policy = torch.zeros(2, 13)
        policy[0, 5] = 1.0                    # every bit of weight on an illegal drop
        policy[1, 0], policy[1, 12] = 0.25, 0.75
        generator = torch.Generator().manual_seed(3)
        seen = set()
        for greedy in (False, True):
            for _ in range(200):
                choice = sample_actions(policy, torch.full((2,), greedy), legal, generator)
                self.assertTrue(bool(legal.gather(1, choice[:, None]).all()), choice)
                seen.update(enumerate(choice.tolist()))
        # No legal weight: uniform over the legal moves. Otherwise the legal
        # weights decide, and an unweighted legal action is never drawn.
        self.assertTrue({(0, 0), (0, 1), (1, 0)} <= seen)
        self.assertNotIn((1, 1), seen)
        with self.assertRaises(RuntimeError):
            sample_actions(torch.ones(1, 13), torch.zeros(1, dtype=torch.bool),
                           torch.zeros(1, 13, dtype=torch.bool))

    def test_arena_openings_sample_only_legal_moves(self):
        from . import arena
        board = BoardBatch([4, 4], [4, 4], [3, 3], [False, True], 'cpu')
        policy = torch.zeros(2, 13)
        policy[0, ROT_CW] = 1.0               # a rotation, illegal on the classic board
        policy[1, 5] = 1.0                    # a column off the 4-wide board
        zeros = torch.zeros(2, dtype=torch.bool)
        with patch.object(arena, 'search', return_value=(None, None)), \
                patch.object(arena, 'visit_policy', return_value=policy):
            for _ in range(50):
                choice = arena._open(((None, 4), (None, 4)), board, zeros, zeros, False,
                                     None, None, None)
                self.assertTrue(legal_choices(board, choice), choice)

    def test_checked_steps_reject_illegal_actions(self):
        board = BoardBatch([4, 4], [4, 4], [3, 3], [False, True], 'cpu')
        for _ in range(4):                    # alternate colours up column 0: no line
            board, outcome = step(board, torch.tensor([0, 0]), check=True)
            self.assertTrue(bool((outcome == NOT_TERMINAL).all()))
        for actions in ([0, 1], [1, 0], [ROT_CW, 1], [5, 1], [-1, 1], [1, 13]):
            with self.subTest(actions=actions), self.assertRaisesRegex(ValueError, 'illegal action'):
                step(board, torch.tensor(actions), check=True)
        with self.assertRaisesRegex(ValueError, 'one action per game'):
            step(board, torch.tensor([1]), check=True)
        step(board, torch.tensor([1, FLIP]), check=True)

    def test_unchecked_transform_leaves_a_classic_board_untouched(self):
        # The search steps masked-out rows too; in a mixed batch a transform
        # meant for nobody used to flip or rotate the classic game.
        board = BoardBatch([4, 4], [5, 5], [4, 4], [False, True], 'cpu')
        board, _ = step(board, torch.tensor([0, 0]))
        for action in (FLIP, ROT_CW, ROT_CCW):
            with self.subTest(action=action):
                child, outcome = step(board, torch.tensor([action, action]))
                self.assertEqual(int(outcome[0]), NOT_TERMINAL)
                for name in ('mover', 'opponent', 'heights', 'rows', 'cols', 'pieces'):
                    self.assertTrue(torch.equal(getattr(child, name)[0], getattr(board, name)[0]), name)
                self.assertFalse(torch.equal(child.mover[1], board.mover[1]), 'the Chaos game still moves')

    def test_cpu_selfplay_steps_only_checked_legal_moves(self):
        from . import gpu_selfplay
        checks = []
        real_step = gpu_selfplay.step

        def recorded(board, action, **kwargs):
            checks.append(kwargs.get('check'))
            return real_step(board, action, **kwargs)

        shard = selfplay([(4, 4, 3, True), (4, 4, 3, False)], step=Mock(side_effect=recorded))
        self.assertTrue(checks and all(checks))
        self.assertGreater(len(shard['wdl']), 0)


class SelfPlayTests(unittest.TestCase):
    """How self-play opens its games, and what each of its rows teaches. An
    inverted W/D/L target or a broken improved policy used to pass every CPU
    test, and none ran the Gumbel or the deep-ply branch at all."""

    def test_a_limit_of_zero_opens_no_game_at_random(self):
        # --random-plies 0 still opened a share of the games with one random
        # move, while the run's summary line said there were no random openings.
        from . import gpu_selfplay
        picks = [(4, 1, 3, False), (6, 7, 4, True), (10, 10, 5, False)] * 20
        for share in (.5, 1.):
            with self.subTest(share=share), patch.object(gpu_selfplay, 'RANDOM_OPENING_PLIES', 0), \
                    patch.object(gpu_selfplay, 'RANDOM_OPENING_SHARE', share):
                self.assertEqual(set(gpu_selfplay._random_openings(picks, random.Random(5))), {0})
        with patch.object(gpu_selfplay, 'RANDOM_OPENING_PLIES', 4), \
                patch.object(gpu_selfplay, 'RANDOM_OPENING_SHARE', 1.):
            plies = gpu_selfplay._random_openings(picks, random.Random(5))
        # One move up to the limit, at most an eighth of the cells but at least one.
        for (rows, cols, _connect, _chaos), count in zip(picks, plies):
            self.assertTrue(1 <= count <= max(1, min(4, rows * cols // 8)), (rows, cols, count))
        self.assertEqual({count for (rows, *_rest), count in zip(picks, plies) if rows == 4}, {1})
        self.assertIn(4, plies)

    def test_each_row_learns_the_result_for_its_mover(self):
        from .gpu_selfplay import _finish_shard
        plies, games = 6, 4
        planes = torch.zeros(plies, games, 7, 10, 10, dtype=torch.uint8)
        policy, root = torch.zeros(plies, games, 13), torch.zeros(plies, games)
        for ply in range(plies):
            for game in range(games):
                planes[ply, game, 0, 0, 0] = 10 * game + ply          # names the row
                policy[ply, game, (game + ply) % 13] = 1
                root[ply, game] = (10 * game + ply) / 100
        # The last move won for the player who made it, lost for them (a
        # Chaos transform can complete only the opponent's line), drew by
        # repetition, and a fourth game was still running at the cap.
        outcomes, end_ply = torch.tensor([1, -1, 0, 9]), torch.tensor([4, 3, 2, -1])
        valid = torch.arange(plies)[:, None] <= torch.tensor([4, 3, 2, plies - 1])[None, :]
        legal = torch.ones(plies, games, 13, dtype=torch.bool)
        result, capped, positions = _finish_shard(planes, legal, policy, valid, outcomes, end_ply, root)
        self.assertEqual((capped, positions), (1, 12))
        rows = [(code // 10, code % 10) for code in result['planes'][:, 0, 0, 0].tolist()]
        # Ply by ply, game by game within a ply (distill.filtered_chunks
        # samples a shard rather than slicing it for that reason).
        self.assertEqual(rows, sorted(rows, key=lambda row: (row[1], row[0])))
        # 2 a win, 1 a draw, 0 a loss for the player to move: the last move's
        # result for whoever made it, flipped at every ply before it.
        self.assertEqual(dict(zip(rows, result['wdl'].tolist())),
                         {(0, 4): 2, (0, 3): 0, (0, 2): 2, (0, 1): 0, (0, 0): 2,
                          (1, 3): 0, (1, 2): 2, (1, 1): 0, (1, 0): 2,
                          (2, 2): 1, (2, 1): 1, (2, 0): 1})
        # The other targets stay with their rows.
        self.assertEqual(result['policy'].argmax(dim=1).tolist(), [(game + ply) % 13 for game, ply in rows])
        self.assertTrue(torch.allclose(result['root_value'].float(),
                                       torch.tensor([(10 * game + ply) / 100 for game, ply in rows]), atol=1e-3))

    def test_the_improved_policy_on_a_worked_example(self):
        legal = torch.zeros(1, 13, dtype=torch.bool)
        legal[0, :3] = True
        prior, visits, value_sum = torch.zeros(1, 13), torch.zeros(1, 13), torch.zeros(1, 13)
        prior[0, :3] = torch.tensor([.5, .3, .2])
        visits[0, :2] = torch.tensor([3., 1.])
        value_sum[0, :2] = torch.tensor([1.5, -.5])
        # The visited actions complete to their search values, 0.5 and -0.5.
        # The unvisited one gets v_mix = (0.1 + 4 * 0.125) / (1 + 4) = 0.12:
        # 0.1 is the network's value of the root, 4 the visits and 0.125 the
        # prior-weighted value of the visited actions, (.5 * .5 - .3 * .5) / .8.
        completed = torch.tensor([.5, -.5, .12])
        for c_visit in (0., 50.):
            with self.subTest(c_visit=c_visit):
                policy = improved_policy(prior, visits, value_sum, torch.tensor([.1]), legal, c_visit, 1.)
                # softmax(log prior + (c_visit + most visits) * c_scale * (completed + 1) / 2)
                expected = torch.softmax(torch.log(prior[0, :3]) + (c_visit + 3) * ((completed + 1) / 2), dim=0)
                self.assertTrue(torch.allclose(policy[0, :3], expected, atol=1e-6), policy)
                self.assertFalse(policy[0, 3:].any())
                self.assertEqual(int(policy.argmax()), 0)
        # Nothing visited: every action completes to the network's value, so
        # the target is the prior over the legal actions.
        legal[0, 2] = False
        policy = improved_policy(prior, torch.zeros(1, 13), torch.zeros(1, 13), torch.tensor([-.3]), legal, 50., 1.)
        self.assertTrue(torch.allclose(policy[0], torch.tensor([.5 / .8, .3 / .8] + [0.] * 11)), policy)

    def test_gumbel_targets_teach_every_ply_from_the_prior_before_noise(self):
        searched, improved = [], []

        def search(net, forward, board, rep1, rep2, sims, **kwargs):
            # What the network itself says of each root, before the search
            # mixes exploration noise into its prior.
            legal = board.legal()
            with torch.no_grad():
                logits, wdl, _q = forward(net, board.planes(rep1, rep2), legal)
            value = torch.softmax(wdl, dim=1)
            searched.append((torch.softmax(logits.masked_fill(~legal, float('-inf')), dim=1),
                             value[:, 2] - value[:, 0], sims))
            return search_root(net, forward, board, rep1, rep2, sims, **kwargs)

        def target(prior, visits, value_sum, net_value, legal, c_visit, c_scale):
            policy = improved_policy(prior, visits, value_sum, net_value, legal, c_visit, c_scale)
            improved.append((prior, net_value, policy))
            return policy

        # Classic 4x4 games end within 16 plies, so none is capped, and the
        # shard holds every ply's targets in order.
        shard = selfplay([(4, 4, 3, False)], POLICY_TARGET='gumbel', TARGET_SIMS=4, TARGET_SHARE=.5,
                         search_root=search, improved_policy=target)
        self.assertEqual({sims for *_, sims in searched}, {2, 4})     # shallow and deep plies
        self.assertEqual(len(improved), len(searched))
        for (prior, value, _sims), (used_prior, used_value, _policy) in zip(searched, improved):
            self.assertTrue(torch.allclose(used_prior, prior, atol=1e-5))
            self.assertTrue(torch.allclose(used_value, value, atol=1e-5))
        self.assertTrue(torch.equal(shard['policy'], torch.cat([policy for *_, policy in improved])))
        self.assertTrue(torch.allclose(shard['policy'].sum(dim=1), torch.ones(len(shard['policy']))))
        self.assertFalse(shard['policy'].masked_fill(shard['legal'], 0).any())

    def test_visit_targets_teach_only_the_deep_plies(self):
        budgets = []

        def search(net, forward, board, rep1, rep2, sims, **kwargs):
            budgets.append((len(board), sims))
            return search_root(net, forward, board, rep1, rep2, sims, **kwargs)

        improved = Mock()
        shard = selfplay([(4, 4, 3, False)], TARGET_SIMS=4, TARGET_SHARE=.5, search_root=search,
                         improved_policy=improved)
        improved.assert_not_called()
        self.assertEqual({sims for _width, sims in budgets}, {2, 4})
        # No classic 4x4 game is capped, so each ply's rows are its live games.
        start = 0
        for width, sims in budgets:
            policy, legal = shard['policy'][start:start + width], shard['legal'][start:start + width]
            start += width
            if sims == 4:
                self.assertTrue(torch.allclose(policy.sum(dim=1), torch.ones(width)))
                self.assertFalse(policy.masked_fill(legal, 0).any())
            else:
                # A shallow ply teaches the value head alone.
                self.assertFalse(policy.any())
        self.assertEqual(start, len(shard['policy']))


if __name__ == '__main__': unittest.main()
