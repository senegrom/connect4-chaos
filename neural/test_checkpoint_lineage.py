"""Checkpoint history must follow accepted parents, never file ordering."""
import json
from pathlib import Path
import tempfile
import unittest

from .checkpoint_lineage import lineage_record, read_history, write_lineage


class LineageTests(unittest.TestCase):
    def chain(self, root):
        models = root / 'models'
        models.mkdir()
        names = [f'big{n}-abcdef.pt' for n in range(11)]
        for n in range(1, 11):
            write_lineage(models, names[n], names[n - 1], n)
        return models, names

    def test_duplicate_failed_and_unrelated_models_do_not_change_lag(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            models, names = self.chain(root)
            (models / 'big6-deadbeef.pt').write_bytes(b'retained after failed evaluation')
            (models / 'big9-bad.pt.partial').write_bytes(b'not finished')
            write_lineage(models, 'big9-beef.pt', 'other-seed.pt', 9)
            write_lineage(models, 'big999-beef.pt', 'experiment-seed.pt', 999)
            read = lambda path: [ (root / path).read_bytes() ]
            history = read_history(read, names[10])
            self.assertEqual(history, names)
            self.assertEqual(history[-1 - 5], names[5])
            self.assertEqual(read_history(read, 'big9-beef.pt'), ['other-seed.pt', 'big9-beef.pt'])

    def test_limited_history_reads_only_the_newest_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, names = self.chain(root)
            reads = []
            def read(path):
                reads.append(path)
                return [(root / path).read_bytes()]
            self.assertEqual(read_history(read, names[10], 6), names[5:])
            self.assertEqual(len(reads), 5)
            reads.clear()
            self.assertEqual(read_history(read, names[10], 1), names[10:])
            self.assertEqual(reads, [])
            # A limit past the root returns the whole history.
            self.assertEqual(read_history(read, names[3], 20), names[:4])
            for limit in (0, -1, 1.5, True):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    read_history(read, names[10], limit)

    def test_legacy_seed_is_a_root_not_a_guessed_history(self):
        def missing(path):
            raise FileNotFoundError(path)
        self.assertEqual(read_history(missing, 'legacy.pt'), ['legacy.pt'])

    def test_generations_above_old_experiment_cutoff_are_valid_ancestors(self):
        records = {'models/big901-abc.pt.lineage.json': lineage_record('big901-abc.pt', 'big900-abc.pt', 901),
                   'models/big900-abc.pt.lineage.json': lineage_record('big900-abc.pt', 'seed.pt', 900)}
        def read(path):
            if path not in records:
                raise FileNotFoundError(path)
            return [json.dumps(records[path]).encode()]
        self.assertEqual(read_history(read, 'big901-abc.pt'), ['seed.pt', 'big900-abc.pt', 'big901-abc.pt'])

    def test_transport_failure_is_not_treated_as_legacy(self):
        def fail(path):
            raise ConnectionError('network unavailable')
        with self.assertRaises(ConnectionError):
            read_history(fail, 'model.pt')

    def test_rejects_malformed_wrong_model_and_cyclic_records(self):
        records = [b'not json', b'[]', b'{"version": 9}',
                   json.dumps(lineage_record('other.pt', 'seed.pt', 1)).encode(),
                   b'{"version": 1, "model": "model.pt", "parent": "model.pt", "generation": 1}']
        for raw in records:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                read_history(lambda path: [raw], 'model.pt')
        cycle = {'models/a.pt.lineage.json': lineage_record('a.pt', 'b.pt', 2),
                 'models/b.pt.lineage.json': lineage_record('b.pt', 'a.pt', 1)}
        with self.assertRaisesRegex(ValueError, 'Cycle'):
            read_history(lambda path: [json.dumps(cycle[path]).encode()], 'a.pt')

    def test_rejects_nonincreasing_generation_and_path_names(self):
        chain = {'models/a.pt.lineage.json': lineage_record('a.pt', 'b.pt', 1),
                 'models/b.pt.lineage.json': lineage_record('b.pt', 'seed.pt', 1)}
        with self.assertRaisesRegex(ValueError, 'generations'):
            read_history(lambda path: [json.dumps(chain[path]).encode()], 'a.pt')
        for name in ('../bad.pt', 'dir\\bad.pt', '', None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                lineage_record(name, 'seed.pt', 1)
        for gen in (True, -1, 1.5):
            with self.subTest(gen=gen), self.assertRaises(ValueError):
                lineage_record('model.pt', 'seed.pt', gen)

    def test_publication_is_idempotent_and_cannot_reparent_a_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = write_lineage(root, 'model.pt', 'seed.pt', 1)
            before = path.read_bytes()
            self.assertEqual(write_lineage(root, 'model.pt', 'seed.pt', 1), path)
            with self.assertRaisesRegex(ValueError, 'different lineage'):
                write_lineage(root, 'model.pt', 'other.pt', 1)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(list(root.glob('*.partial')))


if __name__ == '__main__':
    unittest.main()
