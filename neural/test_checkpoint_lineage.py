"""Checkpoint history must follow accepted parents, never file ordering."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from . import prune
from .checkpoint_lineage import lineage_record, read_generation, read_history, write_lineage


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

    def test_a_generation_comes_from_the_models_own_record(self):
        records = {'models/big901-abc.pt.lineage.json': lineage_record('big901-abc.pt', 'big900-abc.pt', 901)}
        def read(path):
            if path not in records:
                raise FileNotFoundError(path)
            return [json.dumps(records[path]).encode()]
        self.assertEqual(read_generation(read, 'big901-abc.pt'), 901)
        self.assertIsNone(read_generation(read, 'imported.pt'), 'no sidecar: a root sets no generation')
        for raw in (b'not json', json.dumps(lineage_record('other.pt', 'seed.pt', 1)).encode()):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                read_generation(lambda path: [raw], 'model.pt')
        with self.assertRaises(ValueError):
            read_generation(read, '../escape.pt')

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


NOW = 1_000_000
OLD = NOW - 7 * 3600          # older than any writer could still be working on


def volume_listing():
    """models/ after a long run: lineage big1..big10, each with both sidecars."""
    files = {f'big{n}-abc.pt{suffix}': (100, OLD) for n in range(1, 11)
             for suffix in ('', '.opt', '.lineage.json')}
    files.update({
        'big0-seed.pt': (100, OLD),                 # the legacy root, no sidecars
        'big7-branch.pt': (100, OLD),               # an experiment off the lineage
        'big7-branch.pt.lineage.json': (1, OLD),
        'big3-orphan.pt.opt': (300, OLD),           # moments whose checkpoint is gone
        'big11-live.pt.partial': (100, NOW - 60),   # a learner publishing right now
        'big12-dead.pt.partial': (100, OLD),        # a writer killed long ago
        'big12-fresh.pt': (100, NOW - 60),          # just published, not yet the current model
        'notes.txt': (1, OLD),                      # not a checkpoint file: never touched
    })
    return files, [f'big{n}-abc.pt' for n in range(1, 11)]


class PruneTests(unittest.TestCase):
    def test_plan_keeps_the_newest_lineage_and_milestones_with_all_their_files(self):
        files, lineage = volume_listing()
        delete, keep = prune.plan_prune(files, lineage, keep=6, milestones=['big2-abc.pt'], now=NOW)
        group = lambda n: {f'big{n}-abc.pt', f'big{n}-abc.pt.opt', f'big{n}-abc.pt.lineage.json'}
        self.assertEqual(set(delete), group(1) | group(3) | group(4) | {
            'big0-seed.pt', 'big7-branch.pt', 'big7-branch.pt.lineage.json', 'big3-orphan.pt.opt',
            'big12-dead.pt.partial'})
        self.assertEqual(set(keep), set().union(*(group(n) for n in (2, 5, 6, 7, 8, 9, 10))) | {
            'big11-live.pt.partial', 'big12-fresh.pt', 'notes.txt'})
        self.assertEqual(sorted(delete + keep), sorted(files))

    def test_a_checkpoints_files_are_decided_together(self):
        files, lineage = volume_listing()
        files['big4-abc.pt.opt'] = (300, NOW - 60)    # one fresh file protects the whole group
        delete, keep = prune.plan_prune(files, lineage, keep=6, now=NOW)
        self.assertTrue({'big4-abc.pt', 'big4-abc.pt.opt', 'big4-abc.pt.lineage.json'} <= set(keep))
        delete, keep = prune.plan_prune(files, lineage, keep=1, now=NOW)
        self.assertEqual({name for name in keep if name.startswith('big10-')},
                         {'big10-abc.pt', 'big10-abc.pt.opt', 'big10-abc.pt.lineage.json'})
        self.assertTrue({'big9-abc.pt', 'big9-abc.pt.opt', 'big9-abc.pt.lineage.json'} <= set(delete))

    def test_plan_refuses_names_it_cannot_find(self):
        files, lineage = volume_listing()
        with self.assertRaisesRegex(ValueError, 'milestones not in models/: big2-abd.pt'):
            prune.plan_prune(files, lineage, milestones=['big2-abd.pt'], now=NOW)
        with self.assertRaisesRegex(ValueError, 'current model big99-x.pt'):
            prune.plan_prune(files, lineage + ['big99-x.pt'], now=NOW)
        for keep in (0, -1, 1.5, True):
            with self.subTest(keep=keep), self.assertRaises(ValueError):
                prune.plan_prune(files, lineage, keep=keep, now=NOW)
        with self.assertRaises(ValueError):
            prune.plan_prune(files, [], now=NOW)

    def test_volume_wrapper_deletes_only_with_apply(self):
        files, lineage = volume_listing()
        records = {f'models/{lineage[n]}.lineage.json': lineage_record(lineage[n], lineage[n - 1], n + 1)
                   for n in range(1, 10)}
        removed = []

        def read_file(path):
            if path not in records:
                raise FileNotFoundError(path)
            return [json.dumps(records[path]).encode()]

        entries = [SimpleNamespace(path=f'models/{name}', size=size, mtime=mtime, type=1)
                   for name, (size, mtime) in files.items()]
        entries.append(SimpleNamespace(path='models/subdir', size=0, mtime=OLD, type=2))
        volume = SimpleNamespace(listdir=lambda path: entries, read_file=read_file,
                                 remove_file=removed.append)
        modal = ModuleType('modal')
        modal.Volume = SimpleNamespace(from_name=lambda name: volume)
        expected, _ = prune.plan_prune(files, lineage, keep=6, milestones=['big2-abc.pt'], now=NOW)
        for apply in (False, True):
            with self.subTest(apply=apply), patch.dict(sys.modules, modal=modal), \
                    patch.object(prune.time, 'time', return_value=NOW), redirect_stdout(io.StringIO()) as out:
                code = prune.main(['big10-abc.pt', '--milestone', 'big2-abc.pt'] + (['--apply'] if apply else []))
            self.assertEqual(code, 0)
            self.assertEqual(removed, [f'models/{name}' for name in expected] if apply else [])
            self.assertIn('dry run' if not apply else f'removed {len(expected)} files', out.getvalue())


if __name__ == '__main__':
    unittest.main()
