"""Search settings must have identical meaning in eager and captured searches."""
from unittest.mock import patch
import unittest

import torch

from . import gpu_mcts as mcts
from .gpu_env import BoardBatch, hash_keys


class FixtureNet(torch.nn.Module):
    def forward(self, planes, legal):
        n = len(planes)
        logits = torch.zeros((n, 13), device=planes.device)
        logits[:, 0] = 2.0
        wdl = torch.zeros((n, 3), device=planes.device)
        q = torch.zeros((n, 13, 3), device=planes.device)
        q[:, 0, 0] = 2.0
        q[:, 1, 2] = 2.0
        return logits.masked_fill(~legal, float('-inf')), wdl, q


def forward(net, planes, legal):
    return net(planes, legal)


class SearchSettingsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.stack = []
        for name, value in (('_WORKSPACES', {}), ('_GRAPH_POOL', None), ('C_PUCT', 1.5), ('Q_SEED', True)):
            context = patch.object(mcts, name, value)
            context.start()
            self.addCleanup(context.stop)

    def test_cache_identity_includes_c_puct_and_q_seed(self):
        net = FixtureNet()
        cached = []
        for c in (1.5, 0.1, 3.0, 1.5):
            mcts.C_PUCT = c
            ws = mcts.workspace(net, forward, 1, 4, 'cpu', 3, False)
            self.assertEqual(ws.forest.c_puct, c)
            cached.append(ws)
        self.assertIs(cached[0], cached[3])
        self.assertEqual(len({id(ws) for ws in cached}), 3)
        mcts.Q_SEED = False
        different = mcts.workspace(net, forward, 1, 4, 'cpu', 3, False)
        self.assertIsNot(different, cached[0])
        self.assertFalse(different.forest.q_seed)
        mcts.Q_SEED = True
        self.assertIs(mcts.workspace(net, forward, 1, 4, 'cpu', 3, False), cached[0])

    def test_existing_forest_keeps_the_settings_it_was_created_with(self):
        forests = []
        for c in (0.1, 3.0):
            mcts.C_PUCT = c
            forest = mcts.Forest(1, 2, 'cpu')
            forest.legal[0, 0, :2] = True
            forest.prior[0, 0, :2] = torch.tensor([0.9, 0.1])
            forest.edge_value[0, 0, :2] = torch.tensor([-0.4, 0.4])
            forests.append(forest)
        root = torch.zeros(1, dtype=torch.long)
        self.assertEqual(int(forests[0].puct(root).argmax()), 1)
        self.assertEqual(int(forests[1].puct(root).argmax()), 0)
        mcts.Q_SEED = True
        seeded = mcts.Forest(1, 2, 'cpu')
        mcts.Q_SEED = False
        unseeded = mcts.Forest(1, 2, 'cpu')
        legal = torch.ones((1, 13), dtype=torch.bool)
        logits, _value, q = forward(FixtureNet(), torch.zeros((1, 7, 10, 10)), legal)
        for forest in (seeded, unseeded):
            forest.install(root, logits, legal, q)
        self.assertGreater(float(seeded.edge_value.abs().sum()), 0)
        self.assertEqual(float(unseeded.edge_value.abs().sum()), 0)

    def test_bad_settings_fail_before_workspace_allocation(self):
        for value in (-1, float('nan'), float('inf'), True, '1.5'):
            with self.subTest(value=value), patch.object(mcts, 'C_PUCT', value), self.assertRaises(ValueError):
                mcts.workspace(FixtureNet(), forward, 1, 1, 'cpu', 3, False)
        with patch.object(mcts, 'Q_SEED', '0'), self.assertRaises(ValueError):
            mcts.workspace(FixtureNet(), forward, 1, 1, 'cpu', 3, False)
        self.assertFalse(mcts._WORKSPACES)

    def compare_searches(self, device, graphs):
        net = FixtureNet().to(device).eval()
        board = BoardBatch([4] * 2, [4] * 2, [3] * 2, [True] * 2, device)
        zeros = torch.zeros(2, dtype=torch.bool, device=device)
        keys = hash_keys(device, torch.Generator(device=device).manual_seed(7))
        observed = []
        for c, q_seed in ((1.5, True), (0.1, True), (3.0, True), (1.5, False), (1.5, True)):
            mcts.C_PUCT, mcts.Q_SEED = c, q_seed
            with patch.object(mcts, 'USE_GRAPHS', graphs):
                actual = mcts.search(net, forward, board, zeros, zeros, 16, add_noise=False, keys=keys)
            # Fresh eager workspace: this is independent of every cached graph.
            with patch.object(mcts, '_WORKSPACES', {}), patch.object(mcts, 'USE_GRAPHS', False):
                expected = mcts.search(net, forward, board, zeros, zeros, 16, add_noise=False, keys=keys)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
            observed.append(actual[0])
        torch.testing.assert_close(observed[0], observed[-1], rtol=0, atol=0)
        self.assertFalse(torch.equal(observed[1], observed[2]), 'fixture must exercise the exploration change')

    def test_successive_eager_settings_match_fresh_searches(self):
        self.compare_searches('cpu', False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware is required for actual graph replay')
    def test_successive_cuda_settings_match_fresh_eager_searches(self):
        self.compare_searches('cuda', True)


if __name__ == '__main__':
    unittest.main()
