"""Budget accounting regressions through the real evaluation CLI and sweep."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import search_quality as quality
from .gpu_env import BoardBatch
from .model import PolicyValueNet
from .test_review import shard


class SearchQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cli_budgets_are_unique_for_files_and_directories(self):
        for directory in (False, True):
            for sims in (0, 16, 32, 128, 512):
                with self.subTest(directory=directory, sims=sims), tempfile.TemporaryDirectory() as temp:
                    target = Path(temp) if directory else Path(temp) / 'shard.pt'
                    if directory:
                        (target / 'classic-4x4-c3-0000.pt').touch()
                    calls = []
                    def score(_net, _shard, budget, _limit, _device):
                        calls.append(budget)
                        return 0.6, 10
                    def sweep(_net, _paths, budgets, _limit, _device):
                        calls.extend(budgets)
                    with patch.object(sys, 'argv', ['search_quality', 'model.pt', str(target), str(sims), '10']), \
                            patch.object(quality, 'load', return_value=object()), \
                            patch.object(quality, 'sweep', side_effect=sweep), \
                            patch.object(quality, 'blunder_rate', side_effect=score), \
                            patch.object(torch, 'load', return_value=shard([(4, 4, 3, False)])), \
                            patch.object(torch.cuda, 'is_available', return_value=False), redirect_stdout(io.StringIO()):
                        quality.main()
                    expected = (0,) if sims == 0 else tuple(dict.fromkeys((0, 32, sims, 2 * sims if directory else 512)))
                    self.assertEqual(tuple(calls), expected)

    def test_a_corpus_without_held_out_shards_fails_instead_of_scoring_nothing(self):
        # It printed "0 held-out shards" and exited 0, and boards whose
        # training shards had no -0000 were skipped without a word.
        for names, board in (((), 'any board'), (('classic-4x4-c3-0001.pt',), 'classic-4x4-c3'),
                             (('classic-4x4-c3-0000.pt', 'chaos-4x4-c3-0001.pt'), 'chaos-4x4-c3')):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as temp:
                for name in names:
                    (Path(temp) / name).touch()
                with patch.object(sys, 'argv', ['search_quality', 'model.pt', temp, '0', '10']), \
                        patch.object(quality, 'load', return_value=object()), \
                        patch.object(quality, 'sweep', side_effect=AssertionError('nothing to score')), \
                        patch.object(torch.cuda, 'is_available', return_value=False), \
                        self.assertRaisesRegex(SystemExit, f'no held-out -0000 shard for {board}'):
                    quality.main()

    def test_a_validation_shard_is_read_as_the_learner_reads_it(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / '5x5c4classic-0000.pt'
            torch.save(shard([(5, 5, 4, False)] * 20, split='validation'), path)
            self.assertEqual(len(quality.load_validation_shard(path, 13)['planes']), 13)
            # The learner refuses a first shard that says it is training
            # data; so does this, rather than scoring positions it trained on.
            torch.save(shard([(5, 5, 4, False)] * 20), path)
            with self.assertRaisesRegex(ValueError, "declares 'train', expected validation"):
                quality.load_validation_shard(path, 13)

    def test_a_shard_of_an_older_format_is_refused_by_name_in_both_modes(self):
        # Float planes divided by 10 read as empty boards: the single-shard
        # mode used to score those and print plausible rates.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / '5x5c4classic-0000.pt'
            old = shard([(5, 5, 4, False)] * 20, split='validation')
            old['planes'] = old['planes'].float() / 10
            torch.save(old, path)
            with self.assertRaises(ValueError) as caught:
                quality.load_validation_shard(path, 13)
            self.assertIn(f'{path} predates the current shard format', str(caught.exception))
            with patch.object(sys, 'argv', ['search_quality', 'model.pt', str(path), '0', '10']), \
                    patch.object(quality, 'load', return_value=object()), \
                    patch.object(torch.cuda, 'is_available', return_value=False), \
                    self.assertRaises(ValueError) as caught:
                quality.main()
            self.assertIn(f'{path} predates the current shard format', str(caught.exception))
        with self.assertRaisesRegex(ValueError, 'not uint8 scaled by 10'):
            quality.decode_planes(torch.zeros(1, 7, 10, 10))

    def test_duplicate_sweeps_use_each_budgets_own_denominator(self):
        for budgets in ((0, 32, 16, 32), (0, 32, 32, 64), (0, 32, 128, 256)):
            with self.subTest(budgets=budgets):
                paths = [Path('4x4c3classic-0000.pt'), Path('4x4c3chaos-0000.pt')]
                calls = []
                def score(_net, shard, budget, _limit, _device):
                    calls.append((shard, budget))
                    # Vary the denominator: using the first budget's count is
                    # independently wrong even after deduplicating the budget.
                    return 0.6, 5 if budget == 32 else 10
                stats = dict.fromkeys(quality.HEAD_KEYS, 0.0)
                stats['count'] = 10
                with patch.object(quality, 'load_validation_shard', side_effect=lambda path, _limit: path), \
                        patch.object(quality, 'blunder_rate', side_effect=score), \
                        patch.object(quality, 'head_stats', return_value=stats), redirect_stdout(io.StringIO()):
                    lines = quality.sweep(object(), paths, iter(budgets), 10, 'cpu')
                unique = tuple(dict.fromkeys(budgets))
                self.assertEqual(len(calls), len(paths) * len(unique))
                pooled = [line for line in lines if line.startswith('pooled ')]
                self.assertEqual(len(pooled), 3)
                for line in pooled:
                    for budget in unique:
                        self.assertEqual(line.count(f'{quality.label(budget)} 0.6000'), 1, line)
                    self.assertNotIn('1.2000', line)

    def test_empty_budget_is_reported_without_fabricating_zero_error(self):
        stats = dict.fromkeys(quality.HEAD_KEYS, 0.0)
        with patch.object(quality, 'load_validation_shard', return_value={}), \
                patch.object(quality, 'blunder_rate', side_effect=lambda _n, _s, b, *_: (0.5, 4) if b else (0, 0)), \
                patch.object(quality, 'head_stats', return_value=stats), redirect_stdout(io.StringIO()):
            lines = quality.sweep(object(), [Path('4x4c3classic-0000.pt')], (0, 32), 10, 'cpu')
        pooled = [line for line in lines if line.startswith('pooled ')]
        self.assertEqual(len(pooled), 2)
        for line in pooled:
            self.assertIn('policy n/a', line)
            self.assertIn('32 sims 0.5000', line)

    def test_invalid_budgets_fail_before_evaluation(self):
        for budgets in ((), (-1,), (True,), (1.5,), ('32',)):
            with self.subTest(budgets=budgets), self.assertRaises(ValueError):
                quality.unique_budgets(budgets)
        for sims in (-1, True, 1.5):
            with self.subTest(sims=sims), self.assertRaises(ValueError):
                quality.evaluation_budgets(sims, directory=True)

    def test_real_cpu_network_and_search_respect_sample_limits(self):
        board = BoardBatch([4] * 5, [4] * 5, [3] * 5, [False] * 5, 'cpu')
        zeros = torch.zeros(5, dtype=torch.bool)
        legal = board.legal()
        shard = dict(planes=(board.planes(zeros, zeros) * 10).round().to(torch.uint8), legal=legal,
                     policy=legal.float() / legal.sum(1, keepdim=True),
                     wdl=torch.ones(5, dtype=torch.long), q=torch.ones((5, 13), dtype=torch.long))
        net = PolicyValueNet(4, 1, 4).eval()
        for budget in (0, 4):
            rate, count = quality.blunder_rate(net, shard, budget, 3, 'cpu')
            self.assertEqual(count, 3)
            self.assertEqual(rate, 0.0)  # this fixture marks every legal move optimal


if __name__ == '__main__':
    unittest.main()
