#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name('perfect-chaos-auto-advance.py')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_sums(directory: Path, names: list[str]) -> None:
    lines = []
    for name in names:
        data = (directory / name).read_bytes()
        lines.append(f'{digest(data)}  {name}')
    (directory / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')


def make_fixture(root: Path, *, new_rejections: int = 3) -> tuple[Path, Path, Path]:
    round_directory = root / 'round'
    evidence_directory = root / 'evidence'
    round_directory.mkdir()
    evidence_directory.mkdir()

    role = 'yellow'
    from_pieces = 14
    target_pieces = 16
    existing = 12
    cumulative = existing + new_rejections
    rejected_data = b'new-rejections' + bytes([new_rejections])
    cumulative_data = b'cumulative-rejections' + bytes([cumulative])
    policy_data = b'policy'
    frontier_data = b'frontier'

    classification = {
        'format': 'connect4-chaos-frontier-classification-merged-v1',
        'role': role,
        'fromPieces': from_pieces,
        'targetPieces': target_pieces,
        'shards': 8,
        'inputRoots': 100,
        'rejectedRoots': new_rejections,
        'safeInputRoots': 100 - new_rejections,
        'classificationComplete': True,
        'safePolicyEntries': 51,
        'safeFrontierStates': 63,
        'policyConflicts': 0,
        'duplicateRejectedRecords': 0,
        'duplicateFrontierRecords': 7,
        'attempts': 10,
        'splitEvents': 2,
        'maximumSplitDepth': 1,
        'safeLeaves': 5,
        'rejectedLeaves': 3,
        'targetRejectSha256': None,
        'artifacts': {
            'rejected': {
                'path': 'new-reject-14.bin',
                'bytes': len(rejected_data),
                'sha256': digest(rejected_data),
            },
            'policy': {
                'path': '14-16.policy.bin',
                'bytes': len(policy_data),
                'sha256': digest(policy_data),
            },
            'frontier': {
                'path': '14-16.frontier.bin',
                'bytes': len(frontier_data),
                'sha256': digest(frontier_data),
            },
        },
    }
    summary = {
        **classification,
        'existingRejectedRoots': existing,
        'newRejectedRoots': new_rejections,
        'cumulativeRejectedRoots': cumulative,
        'rejectionProgress': new_rejections,
    }

    files = {
        'campaign-summary.json': (json.dumps(summary, indent=2) + '\n').encode(),
        'classification.json': (json.dumps(classification, indent=2) + '\n').encode(),
        'new-reject-14.bin': rejected_data,
        'reject-14.bin': cumulative_data,
        '14-16.policy.bin': policy_data,
        '14-16.frontier.bin': frontier_data,
    }
    for name, data in files.items():
        (round_directory / name).write_bytes(data)
    for name in ('campaign-summary.json', 'classification.json', 'new-reject-14.bin', 'reject-14.bin'):
        (evidence_directory / name).write_bytes(files[name])
    (evidence_directory / 'raw-shard-audit.json').write_text('{"audit":"pass"}\n')
    evidence_names = [
        'campaign-summary.json',
        'classification.json',
        'new-reject-14.bin',
        'raw-shard-audit.json',
        'reject-14.bin',
    ]
    if new_rejections == 0:
        replay = {
            'replay': {
                'role': role,
                'segments': [
                    {'fromPieces': 0, 'frontierPieces': 14},
                    {'fromPieces': 14, 'frontierPieces': 16},
                ],
            },
        }
        replay_bytes = (json.dumps(replay, indent=2) + '\n').encode()
        (round_directory / 'yellow-16-replay.json').write_bytes(replay_bytes)
        (evidence_directory / 'closure-replay.json').write_bytes(replay_bytes)
        files['yellow-16-replay.json'] = replay_bytes
        evidence_names.append('closure-replay.json')

    write_sums(round_directory, list(files))
    write_sums(evidence_directory, evidence_names)

    state = {
        'role': role,
        'sourceRun': 8,
        'sourceSha': '1' * 40,
        'sourceArtifact': 'previous-round',
        'existingRejections': 9,
        'cumulativeRejections': existing,
        'prepareShards': 4,
        'prepareWorkers': 2,
        'shardCount': 8,
    }
    state_path = root / 'state.json'
    state_path.write_text(json.dumps(state, indent=2) + '\n')
    return state_path, round_directory, evidence_directory


def run_tool(root: Path, state: Path, round_directory: Path, evidence_directory: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            '--state', str(state),
            '--round-directory', str(round_directory),
            '--evidence-directory', str(evidence_directory),
            '--role', 'yellow',
            '--from-pieces', '14',
            '--target-pieces', '16',
            '--run-id', '99',
            '--run-sha', '2' * 40,
            '--result-artifact', 'perfect-chaos-yellow-16-12-round',
            '--next-state', str(root / 'next.json'),
            '--decision', str(root / 'decision.json'),
        ],
        text=True,
        capture_output=True,
    )


class AutoAdvanceTests(unittest.TestCase):
    def test_valid_audited_round_emits_next_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state, round_directory, evidence_directory = make_fixture(root)
            result = run_tool(root, state, round_directory, evidence_directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            next_state = json.loads((root / 'next.json').read_text())
            self.assertEqual(next_state['sourceRun'], 99)
            self.assertEqual(next_state['sourceSha'], '2' * 40)
            self.assertEqual(next_state['existingRejections'], 12)
            self.assertEqual(next_state['cumulativeRejections'], 15)
            decision = json.loads((root / 'decision.json').read_text())
            self.assertFalse(decision['closedCandidate'])
            self.assertEqual(decision['newRejectedRoots'], 3)

    def test_independent_evidence_disagreement_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state, round_directory, evidence_directory = make_fixture(root)
            (evidence_directory / 'classification.json').write_text('{}\n')
            write_sums(
                evidence_directory,
                [
                    'campaign-summary.json',
                    'classification.json',
                    'new-reject-14.bin',
                    'raw-shard-audit.json',
                    'reject-14.bin',
                ],
            )
            result = run_tool(root, state, round_directory, evidence_directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('independent auditor disagree', result.stderr)
            self.assertFalse((root / 'next.json').exists())

    def test_wrong_starting_rejection_count_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state, round_directory, evidence_directory = make_fixture(root)
            value = json.loads(state.read_text())
            value['cumulativeRejections'] = 13
            state.write_text(json.dumps(value) + '\n')
            result = run_tool(root, state, round_directory, evidence_directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('certified cumulative rejection count', result.stderr)

    def test_zero_counterexample_requires_independent_closure_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state, round_directory, evidence_directory = make_fixture(root, new_rejections=0)
            (evidence_directory / 'closure-replay.json').unlink()
            write_sums(
                evidence_directory,
                [
                    'campaign-summary.json',
                    'classification.json',
                    'new-reject-14.bin',
                    'raw-shard-audit.json',
                    'reject-14.bin',
                ],
            )
            result = run_tool(root, state, round_directory, evidence_directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('independent closure replay evidence', result.stderr)
            self.assertFalse((root / 'next.json').exists())

    def test_zero_counterexample_round_is_only_a_closure_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state, round_directory, evidence_directory = make_fixture(root, new_rejections=0)
            result = run_tool(root, state, round_directory, evidence_directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / 'next.json').exists())
            decision = json.loads((root / 'decision.json').read_text())
            self.assertTrue(decision['closedCandidate'])
            self.assertIsNone(decision['nextState'])
            self.assertEqual(decision['checksums']['closureReplay'], digest(
                (evidence_directory / 'closure-replay.json').read_bytes(),
            ))


if __name__ == '__main__':
    unittest.main()
