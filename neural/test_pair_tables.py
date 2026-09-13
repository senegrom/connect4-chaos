"""Validate table identity and consistency, including real native dataset builds."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from .build_dataset import build
from .chaos_game import ACTION_INDEX, empty_state, successors, to_planes
from .data_split import state_is_validation
from .pair_tables import GROUP_WORDS, HEADER, HEADER_BYTES, Geometry, PairTable

ROOT = Path(__file__).resolve().parents[1]


def fixture(directory, *, shape=(4, 5, 4), pieces=17, pair=9, slots=None, chaos=True):
    """Structurally valid multi-group block, not an assertion of game values."""
    geometry = Geometry(*shape)
    size = geometry.pair_slots(pieces, pair)
    selected = slots if slots is not None else [0, GROUP_WORDS * 64 + 7, size - 1]
    words = (size + 63) // 64
    bits = bytearray(words * 8)
    for slot in selected:
        bits[slot // 8] |= 1 << (slot % 8)
    ranks = bytearray()
    count = 0
    for start in range(0, len(bits), GROUP_WORDS * 8):
        ranks.extend(struct.pack('<Q', count))
        count += int.from_bytes(bits[start:start + GROUP_WORDS * 8], 'little').bit_count()
    base = directory / f'pair-{pieces}-{pair}'
    identity = (b'C4PAIR2\0', *shape)
    kind = 0 if chaos else 2
    base.with_suffix('.bits').write_bytes(HEADER.pack(*identity, kind, pieces, pair, words) + bits)
    base.with_suffix('.ranks').write_bytes(ranks)
    if selected:
        base.with_suffix('.values').write_bytes(HEADER.pack(*identity, kind + 1, pieces, pair, count)
                                                + bytes(i % 3 for i in range(count)))
    return base, size, selected


class PairTableValidationTests(unittest.TestCase):
    def test_valid_multigroup_rank_and_zero_count_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, selected = fixture(root)
            with PairTable(root, 4, 5, 4) as table:
                for i, slot in enumerate(selected):
                    self.assertEqual(table.rank(17, 9, slot), i)
                    self.assertEqual(table.value_at(17, 9, slot), i % 3 - 1)
                self.assertEqual(table.block_count(17, 9), len(selected))
                with self.assertRaises(KeyError):
                    table.rank(17, 9, 1)
                with patch.object(table, '_map', side_effect=AssertionError('revalidated')):
                    self.assertEqual(table.block_count(17, 9), 3)
            self.assertFalse(table._blocks)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, slots=[])
            with PairTable(root, 4, 5, 4) as table:
                self.assertEqual(table.block_count(17, 9), 0)  # no .values is valid here
                with self.assertRaisesRegex(ValueError, 'no solved reachable'):
                    table.validate()

    def test_mapping_cache_is_bounded_and_replaced_files_are_revalidated(self):
        with tempfile.TemporaryDirectory() as temp, patch('neural.pair_tables.MAX_MAPPED_BLOCKS', 1):
            root = Path(temp)
            base, _, _ = fixture(root)
            fixture(root, pieces=16, pair=8, slots=[0])
            with PairTable(root, 4, 5, 4) as table:
                table.block_count(17, 9)
                first = table._blocks[17, 9].bits
                table.block_count(16, 8)
                self.assertTrue(first.closed)
                self.assertEqual(len(table._blocks), 1)
                with patch.object(table, '_count', side_effect=AssertionError('rescanned')):
                    self.assertEqual(table.block_count(17, 9), 3)
                table.block_count(16, 8)
                ranks = base.with_suffix('.ranks')
                data = bytearray(ranks.read_bytes())
                struct.pack_into('<Q', data, 8, 2)
                replacement = ranks.with_suffix('.new')
                replacement.write_bytes(data); replacement.replace(ranks)
                with self.assertRaisesRegex(ValueError, 'rank prefix'):
                    table.block_count(17, 9)

    def test_every_header_field_is_checked_for_bits_and_values(self):
        for suffix in ('.bits', '.values'):
            for field in range(8):
                with self.subTest(suffix=suffix, field=field), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    base, _, _ = fixture(root)
                    path = base.with_suffix(suffix)
                    original = path.read_bytes()
                    header = list(HEADER.unpack(original[:HEADER_BYTES]))
                    header[field] = b'NOTPAIR\0' if field == 0 else header[field] + 1
                    path.write_bytes(HEADER.pack(*header) + original[HEADER_BYTES:])
                    with PairTable(root, 4, 5, 4) as table:
                        with self.assertRaisesRegex(ValueError, 'identity/header mismatch'):
                            table.block_count(17, 9)
                        self.assertFalse(table._blocks)
                        path.write_bytes(original)
                        self.assertEqual(table.block_count(17, 9), 3)  # retry succeeds

    def test_truncated_extended_missing_and_corrupt_companions_are_rejected(self):
        mutations = ['short-bits', 'extra-bits', 'short-values', 'extra-values',
                     'short-ranks', 'extra-ranks', 'missing-values', 'missing-ranks',
                     'rank-first', 'rank-later', 'padding', 'invalid-wdl', 'unresolved-wdl']
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                base, _, _ = fixture(root)
                suffix = '.bits' if 'bits' in mutation or mutation == 'padding' else (
                    '.ranks' if 'rank' in mutation else '.values')
                path = base.with_suffix(suffix)
                original = path.read_bytes()
                data = bytearray(original)
                if mutation.startswith('missing'):
                    path.unlink()
                else:
                    if mutation.startswith('short'): del data[-1:]
                    elif mutation.startswith('extra'): data += b'\0'
                    elif mutation == 'rank-first': struct.pack_into('<Q', data, 0, 1)
                    elif mutation == 'rank-later': struct.pack_into('<Q', data, 8, 2)
                    elif mutation == 'padding': data[-1] |= 128
                    else: data[-1] = 255 if mutation == 'invalid-wdl' else 3
                    path.write_bytes(data)
                with PairTable(root, 4, 5, 4) as table:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        table.block_count(17, 9)
                    self.assertFalse(table._blocks)
                    path.write_bytes(original)
                    self.assertEqual(table.block_count(17, 9), 3)

    def test_index_bounds_and_empty_directory_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, size, _ = fixture(root)
            with PairTable(root, 4, 5, 4) as table:
                for pieces, pair in [(-1, 0), (21, 11), (17, 8), (17, 18), (True, 1)]:
                    with self.subTest(pieces=pieces, pair=pair), self.assertRaises(ValueError):
                        table.block_count(pieces, pair)
                for slot in (-1, size, True):
                    with self.subTest(slot=slot), self.assertRaises(ValueError):
                        table.rank(17, 9, slot)
        with tempfile.TemporaryDirectory() as temp, PairTable(temp, 4, 4, 4) as table:
            with self.assertRaisesRegex(ValueError, 'no solved reachable'):
                table.sample_state(random.Random(1))
        for shape in ((0, 4, 3), (8, 4, 3), (4, 4, 5), (True, 4, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                PairTable('.', *shape)


class NativeTableDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        compiler = os.environ.get('CXX') or shutil.which('g++') or shutil.which('clang++')
        if not compiler:
            raise unittest.SkipTest('A C++20 compiler is required for native integration')
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        binary = cls.root / ('paired.exe' if sys.platform == 'win32' else 'paired')
        flags = ['-static'] if sys.platform == 'win32' else []
        def run(command):
            result = subprocess.run(command, capture_output=True, text=True, timeout=90, cwd=ROOT)
            if result.returncode:
                raise AssertionError(f'{command}\n{result.stdout}\n{result.stderr}')
        run([compiler, '-std=c++20', '-O2', '-pthread', *flags,
             str(ROOT / 'native/perfect-chaos-paired.cpp'), '-o', str(binary)])
        cls.tables = {}
        for rows, cols, connect, chaos in [(4, 4, 3, False), (4, 4, 4, False),
                                            (3, 4, 3, True), (3, 4, 3, False)]:
            path = cls.root / f'{rows}x{cols}c{connect}-{chaos}'
            run([str(binary), '--rows', str(rows), '--columns', str(cols), '--connect', str(connect),
                 '--threads', '1', '--output', str(path), *([] if chaos else ['--classic'])])
            run([sys.executable, str(ROOT / 'scripts/build-pair-rank-sidecars.py'), str(path)])
            cls.tables[rows, cols, connect, chaos] = path

    def test_native_rules_mismatches_fail_before_dataset_publication(self):
        cases = [((4, 4, 3, False), (4, 4, 4, False)),
                 ((3, 4, 3, True), (3, 4, 3, False)),
                 ((3, 4, 3, False), (3, 4, 3, True)),
                 ((3, 4, 3, True), (4, 3, 3, True))]
        for actual, requested in cases:
            with self.subTest(actual=actual, requested=requested), tempfile.TemporaryDirectory() as temp:
                source = self.tables[actual]
                with PairTable(source, *requested[:3], chaos=requested[3]) as table:
                    with self.assertRaises(ValueError):
                        table.value_of(empty_state(*requested[:2]))
                rows, cols, connect, chaos = requested
                with self.assertRaises(ValueError), patch.object(torch, 'save') as save:
                    build(Path(temp), 1, f'{source}:{rows}:{cols}:{connect}:{"chaos" if chaos else "classic"}', 4, 1)
                save.assert_not_called()
                self.assertEqual(list(Path(temp).iterdir()), [])

    def test_late_mixed_block_is_rejected_before_the_first_shard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'mixed'
            shutil.copytree(self.tables[4, 4, 4, False], source)
            blocks = sorted(source.glob('*.values'))
            last = blocks[-1]
            data = bytearray(last.read_bytes())
            data[10] = 3  # only the Connect length changes in a late block
            last.write_bytes(data)
            output = root / 'dataset'; output.mkdir()
            with self.assertRaises(ValueError), patch.object(torch, 'save') as save:
                build(output, 1, f'{source}:4:4:4:classic', 4, 1)
            save.assert_not_called()
            self.assertEqual(list(output.iterdir()), [])

    def test_valid_native_tables_keep_wdl_q_policy_and_sampling(self):
        for rules, source in self.tables.items():
            with self.subTest(rules=rules), tempfile.TemporaryDirectory() as temp:
                r, c, k, chaos = rules
                with PairTable(source, r, c, k, chaos=chaos) as table:
                    table.validate()
                    self.assertGreater(table._total, 0)
                    if (r, c) == (4, 4):
                        self.assertEqual(table.value_of(empty_state(r, c)), 1 if k == 3 else 0)
                    expected = []
                    rng = random.Random(4)
                    while len(expected) < 8:
                        state, value = table.sample_state(rng)
                        if not state_is_validation(state, k, chaos):
                            edges = successors(state, k, chaos=chaos)
                            q = {ACTION_INDEX[e.action]: table.edge_value_for_mover(e) + 1 for e in edges}
                            expected.append((state, value, q))
                with redirect_stdout(io.StringIO()):
                    build(Path(temp), 8, f'{source}:{r}:{c}:{k}:{"chaos" if chaos else "classic"}', 4, 1)
                saved = torch.load(next(Path(temp).glob('*.pt')), weights_only=True)
                for i, (state, value, q) in enumerate(expected):
                    self.assertTrue(torch.equal(saved['planes'][i],
                        (torch.tensor(to_planes(state, k, chaos=chaos)) * 10).round().to(torch.uint8)))
                    self.assertEqual(saved['wdl'][i].item(), value + 1)
                    best = [a for a, v in q.items() if v == value + 1]
                    self.assertTrue(best)
                    for a in range(13):
                        self.assertEqual(saved['q'][i, a].item(), q.get(a, 3))
                        self.assertEqual(saved['legal'][i, a].item(), a in q)
                        self.assertAlmostEqual(saved['policy'][i, a].item(),
                                               1 / len(best) if a in best else 0)


if __name__ == '__main__':
    unittest.main()
