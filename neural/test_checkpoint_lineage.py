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


def gap_listing():
    """models/ after the documented prune (current big620, milestone big504)
    and twenty more generations: big504 is an imported root with no record,
    and big615's record names big614, which that prune deleted."""
    files, parents = {'big504-root.pt': (100, OLD)}, {}
    for n in range(615, 641):
        files.update({f'big{n}-abc.pt{suffix}': (100, OLD) for suffix in ('', '.opt', '.lineage.json')})
        parents[f'big{n}-abc.pt'] = f'big{n - 1}-abc.pt'
    return files, parents


def run_prune(files, records, argv):
    """prune.main on a fake Volume: (removed paths, read paths, output).
    records maps a sidecar path to its record, or to raw bytes."""
    removed, reads = [], []

    def read_file(path):
        reads.append(path)
        if path not in records:
            raise FileNotFoundError(path)
        record = records[path]
        return [record if isinstance(record, bytes) else json.dumps(record).encode()]

    entries = [SimpleNamespace(path=f'models/{name}', size=size, mtime=mtime, type=1)
               for name, (size, mtime) in files.items()]
    volume = SimpleNamespace(listdir=lambda path: entries, read_file=read_file, remove_file=removed.append)
    modal = ModuleType('modal')
    modal.Volume = SimpleNamespace(from_name=lambda name: volume)
    with patch.dict(sys.modules, modal=modal), patch.object(prune.time, 'time', return_value=NOW), \
            redirect_stdout(io.StringIO()) as out:
        prune.main(argv)
    return removed, reads, out.getvalue()


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

    def test_a_stale_current_model_cannot_delete_its_descendants(self):
        # A model name copied from an older log line: big9 and big10 are the
        # live run's newest checkpoints, and they used to go with their moments.
        files, lineage = volume_listing()
        parents = {f'big{n}-abc.pt': f'big{n - 1}-abc.pt' for n in range(2, 11)}
        parents['big7-branch.pt'] = 'big6-abc.pt'
        with self.assertRaises(ValueError) as caught:
            prune.plan_prune(files, lineage[:8], keep=6, now=NOW, parents=parents)
        self.assertIn('2 checkpoint(s) to delete may descend from the current model big8-abc.pt', str(caught.exception))
        for model in ('big9-abc.pt', 'big10-abc.pt'):
            self.assertIn(f'{model}: descends from the current model big8-abc.pt', str(caught.exception))
        delete, _keep = prune.plan_prune(files, lineage[:8], keep=6, now=NOW, parents=parents, force=True)
        self.assertIn('big10-abc.pt', delete)
        # The newest checkpoint as current, a descendant still publishing, or a
        # branch off an ancestor: nothing at risk.
        prune.plan_prune(files, lineage, keep=6, now=NOW, parents=parents)
        files['big9-abc.pt'] = files['big10-abc.pt'] = (100, NOW - 60)
        prune.plan_prune(files, lineage[:8], keep=6, now=NOW, parents=parents)

    def test_a_chain_broken_by_an_earlier_prune_proves_nothing(self):
        # The documented prune (current big620, milestone big504) deleted
        # big505..big614 with their records; twenty generations later big504,
        # the run's -Init and its first log line, was named as current. The
        # walk from big615 stopped at the missing big614, and the whole run
        # was deleted.
        files, parents = gap_listing()
        run = [f'big{n}-abc.pt' for n in range(615, 641)]
        with self.assertRaises(ValueError) as caught:
            prune.plan_prune(files, ['big504-root.pt'], keep=6, now=NOW, parents=parents)
        self.assertIn('26 checkpoint(s) to delete may descend from the current model big504-root.pt', str(caught.exception))
        self.assertIn('big615-abc.pt: its ancestry cannot be traced past big614-abc.pt, which is no longer in models/',
                      str(caught.exception))
        # A stale name inside the kept window: its descendants are refused.
        with self.assertRaisesRegex(ValueError, 'big640-abc.pt: descends from the current model big617-abc.pt'):
            prune.plan_prune(files, run[:3], keep=6, now=NOW, parents=parents)
        # The right current model: its own ancestors go, as do side branches
        # off them; an experiment whose parent is gone is kept from deletion
        # until a milestone or --force decides it.
        delete, keep = prune.plan_prune(files, run, keep=6, now=NOW, parents=parents)
        self.assertIn('big615-abc.pt.lineage.json', delete)
        self.assertIn('big635-abc.pt', keep)
        files.update({'exp-630.pt': (100, OLD), 'exp-630.pt.lineage.json': (1, OLD),
                      'exp-610.pt': (100, OLD), 'exp-610.pt.lineage.json': (1, OLD)})
        parents.update({'exp-630.pt': 'big630-abc.pt', 'exp-610.pt': 'big610-abc.pt'})
        with self.assertRaises(ValueError) as caught:
            prune.plan_prune(files, run, keep=6, now=NOW, parents=parents)
        self.assertIn('1 checkpoint(s)', str(caught.exception))
        self.assertIn('exp-610.pt: its ancestry cannot be traced past big610-abc.pt', str(caught.exception))
        delete, keep = prune.plan_prune(files, run, keep=6, now=NOW, parents=parents, milestones=['exp-610.pt'])
        self.assertIn('exp-630.pt', delete)
        self.assertIn('exp-610.pt', keep)
        delete, _keep = prune.plan_prune(files, run, keep=6, now=NOW, parents=parents, force=True)
        self.assertIn('exp-610.pt', delete)

    def test_an_unreadable_record_is_named_and_decided_by_a_milestone_or_force(self):
        files, lineage = volume_listing()
        parents = {f'big{n}-abc.pt': f'big{n - 1}-abc.pt' for n in range(2, 11)}
        files.update({'exp-x.pt': (100, OLD), 'exp-x.pt.lineage.json': (1, OLD)})
        parents['exp-x.pt'] = 'big7-branch.pt'
        reason = 'models/big7-branch.pt.lineage.json: ValueError: Checkpoint lineage names a different model'
        unreadable = {'big7-branch.pt': reason}
        with self.assertRaises(ValueError) as caught:
            prune.plan_prune(files, lineage, keep=6, now=NOW, parents=parents, unreadable=unreadable)
        self.assertIn(f'big7-branch.pt: {reason}', str(caught.exception))
        self.assertIn('exp-x.pt: its ancestry passes through big7-branch.pt, whose lineage record is unreadable',
                      str(caught.exception))
        prune.plan_prune(files, lineage, keep=6, now=NOW, parents=parents, unreadable=unreadable,
                         milestones=['big7-branch.pt', 'exp-x.pt'])
        delete, _keep = prune.plan_prune(files, lineage, keep=6, now=NOW, parents=parents,
                                         unreadable=unreadable, force=True)
        self.assertTrue({'big7-branch.pt', 'exp-x.pt'} <= set(delete))

    def test_main_reads_every_record_before_it_plans(self):
        # Nothing reached the refusal through main(): with the parents
        # unread, a stale current model deleted its descendants again.
        files, lineage = volume_listing()
        records = {f'models/{lineage[n]}.lineage.json': lineage_record(lineage[n], lineage[n - 1], n + 1)
                   for n in range(1, 10)}
        with self.assertRaisesRegex(ValueError, 'big10-abc.pt: descends from the current model big8-abc.pt'):
            run_prune(files, records, ['big8-abc.pt', '--apply'])
        removed, reads, _out = run_prune(files, records, ['big8-abc.pt', '--apply', '--force'])
        self.assertTrue({'models/big9-abc.pt', 'models/big10-abc.pt.opt'} <= set(removed))
        self.assertNotIn('models/big10-abc.pt.lineage.json', reads, '--force reads no record it cannot use')
        # A record that cannot be read is named, and a milestone decides it.
        records['models/big7-branch.pt.lineage.json'] = b''
        with self.assertRaisesRegex(ValueError, r'big7-branch.pt: models/big7-branch.pt.lineage.json: JSONDecodeError'):
            run_prune(files, records, ['big10-abc.pt'])
        run_prune(files, records, ['big10-abc.pt', '--milestone', 'big7-branch.pt'])
        # The gap after an earlier prune, through main().
        files, parents = gap_listing()
        records = {f'models/{model}.lineage.json': lineage_record(model, parent, int(model[3:6]))
                   for model, parent in parents.items()}
        with self.assertRaisesRegex(ValueError, '26 checkpoint'):
            run_prune(files, records, ['big504-root.pt', '--apply'])

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
