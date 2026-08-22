#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name('perfect-chaos-certify-closure.py')
FRONTIER_MAGIC = b'C4CFRN1\0'
POLICY_MAGIC = b'C4CPOL1\0'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def empty_table(magic: bytes, role: int, boundary: int, record_size: int) -> bytes:
    header = bytearray(16)
    header[:8] = magic
    header[8] = 1
    header[9] = role
    header[10] = boundary
    header[11] = record_size
    struct.pack_into('<I', header, 12, 0)
    return bytes(header)


def write_sums(directory: Path) -> None:
    names = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob('*')
        if path.is_file() and path.name != 'SHA256SUMS'
    )
    lines = [f'{digest((directory / name).read_bytes())}  {name}' for name in names]
    (directory / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')


def artifact(name: str, run: int, sha: str, artifact_id: int) -> dict:
    return {
        'id': artifact_id,
        'name': name,
        'size_in_bytes': 100,
        'digest': f'sha256:{str(artifact_id % 10) * 64}',
        'created_at': '2026-08-17T00:00:00Z',
        'expires_at': '2026-09-17T00:00:00Z',
        'workflow_run': {'id': run, 'head_sha': sha},
    }


def make_fixture(root: Path) -> tuple[Path, Path, Path]:
    producer = root / 'producer'
    evidence = root / 'evidence'
    producer.mkdir()
    evidence.mkdir()
    role = 'yellow'
    role_code = 2
    from_pieces = 14
    target_pieces = 16
    run = 77
    sha = 'a' * 40

    tables = {
        'new-reject-14.bin': empty_table(FRONTIER_MAGIC, role_code, 14, 19),
        'reject-14.bin': empty_table(FRONTIER_MAGIC, role_code, 14, 19),
        '14-16.policy.bin': empty_table(POLICY_MAGIC, role_code, 16, 20),
        '14-16.frontier.bin': empty_table(FRONTIER_MAGIC, role_code, 16, 19),
    }
    proof = {
        'newRejectSha256': digest(tables['new-reject-14.bin']),
        'cumulativeRejectSha256': digest(tables['reject-14.bin']),
        'policySha256': digest(tables['14-16.policy.bin']),
        'frontierSha256': digest(tables['14-16.frontier.bin']),
    }
    classification = {
        'format': 'connect4-chaos-frontier-classification-merged-v1',
        'role': role,
        'fromPieces': from_pieces,
        'targetPieces': target_pieces,
        'shards': 4,
        'inputRoots': 1,
        'rejectedRoots': 0,
        'safeInputRoots': 1,
        'classificationComplete': True,
        'safePolicyEntries': 0,
        'safeFrontierStates': 0,
        'policyConflicts': 0,
        'artifacts': {
            'rejected': {
                'path': 'new-reject-14.bin',
                'bytes': len(tables['new-reject-14.bin']),
                'sha256': proof['newRejectSha256'],
            },
            'policy': {
                'path': '14-16.policy.bin',
                'bytes': len(tables['14-16.policy.bin']),
                'sha256': proof['policySha256'],
            },
            'frontier': {
                'path': '14-16.frontier.bin',
                'bytes': len(tables['14-16.frontier.bin']),
                'sha256': proof['frontierSha256'],
            },
        },
    }
    summary = {
        **classification,
        'existingRejectedRoots': 0,
        'newRejectedRoots': 0,
        'cumulativeRejectedRoots': 0,
        'rejectionProgress': 0,
    }
    replay = {
        'compiler': '/usr/bin/g++',
        'directory': '/tmp/assembled',
        'replay': {
            'role': role,
            'segments': [{
                'fromPieces': 0,
                'frontierPieces': 16,
                'frontierStates': 0,
                'policyEntries': 0,
            }],
        },
    }
    audit = {
        'format': 'connect4-chaos-independent-sharded-round-audit-v1',
        'status': 'pass',
        'role': role,
        'fromPieces': 14,
        'targetPieces': 16,
        'shards': 4,
        'inputRoots': 1,
        'existingRejectedRoots': 0,
        'newRejectedRoots': 0,
        'cumulativeRejectedRoots': 0,
        'safeInputRoots': 1,
        'safePolicyEntries': 0,
        'safeFrontierStates': 0,
        'policyConflicts': 0,
        'proofTables': proof,
    }

    for name, data in tables.items():
        (producer / name).write_bytes(data)
    (producer / 'campaign-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (producer / 'classification.json').write_text(json.dumps(classification, indent=2) + '\n')
    (producer / 'yellow-16-replay.json').write_text(json.dumps(replay, indent=2) + '\n')
    prepared = producer / 'yellow-prepared'
    assembled = producer / 'assembled' / 'yellow'
    prepared.mkdir()
    assembled.mkdir(parents=True)
    for name, data in tables.items():
        if name != 'new-reject-14.bin':
            (prepared / name).write_bytes(data)
            (assembled / name).write_bytes(data)
    write_sums(producer)

    for name in ('campaign-summary.json', 'classification.json', 'new-reject-14.bin', 'reject-14.bin'):
        (evidence / name).write_bytes((producer / name).read_bytes())
    (evidence / 'raw-shard-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (evidence / 'closure-replay.json').write_text(json.dumps(replay, indent=2) + '\n')
    write_sums(evidence)

    candidate = {
        'format': 'connect4-chaos-auto-advance-decision-v1',
        'role': role,
        'fromPieces': 14,
        'targetPieces': 16,
        'run': run,
        'runSha': sha,
        'resultArtifact': 'perfect-chaos-yellow-16-0-round',
        'existingRejectedRoots': 0,
        'newRejectedRoots': 0,
        'cumulativeRejectedRoots': 0,
        'closedCandidate': True,
        'nextState': None,
        'checksums': {
            'roundManifest': digest((producer / 'SHA256SUMS').read_bytes()),
            'evidenceManifest': digest((evidence / 'SHA256SUMS').read_bytes()),
            'closureReplay': digest((evidence / 'closure-replay.json').read_bytes()),
        },
        'artifacts': {
            'producer': artifact('perfect-chaos-yellow-16-0-round', run, sha, 101),
            'independentEvidence': artifact('perfect-chaos-yellow-16-0-evidence', run, sha, 102),
        },
    }
    candidate_path = root / 'candidate.json'
    candidate_path.write_text(json.dumps(candidate, indent=2) + '\n')
    return candidate_path, producer, evidence


def run_tool(root: Path, candidate: Path, producer: Path, evidence: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            '--candidate', str(candidate),
            '--producer', str(producer),
            '--evidence', str(evidence),
            '--output', str(root / 'certificate.json'),
        ],
        text=True,
        capture_output=True,
    )


class CertifyClosureTests(unittest.TestCase):
    def test_valid_closure_emits_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, producer, evidence = make_fixture(root)
            result = run_tool(root, candidate, producer, evidence)
            self.assertEqual(result.returncode, 0, result.stderr)
            certificate = json.loads((root / 'certificate.json').read_text())
            self.assertEqual(certificate['format'], 'connect4-chaos-certified-prefix-closure-v2')
            self.assertEqual(certificate['role'], 'yellow')
            self.assertEqual(certificate['frontierPieces'], 16)
            self.assertEqual(certificate['classification']['newRejectedRoots'], 0)

    def test_replay_disagreement_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, producer, evidence = make_fixture(root)
            replay = json.loads((evidence / 'closure-replay.json').read_text())
            replay['replay']['segments'][0]['policyEntries'] = 1
            (evidence / 'closure-replay.json').write_text(json.dumps(replay) + '\n')
            write_sums(evidence)
            value = json.loads(candidate.read_text())
            value['checksums']['evidenceManifest'] = digest((evidence / 'SHA256SUMS').read_bytes())
            value['checksums']['closureReplay'] = digest((evidence / 'closure-replay.json').read_bytes())
            candidate.write_text(json.dumps(value) + '\n')
            result = run_tool(root, candidate, producer, evidence)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('closure replays differ', result.stderr)

    def test_nonzero_rejection_round_is_not_a_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, producer, evidence = make_fixture(root)
            value = json.loads(candidate.read_text())
            value['newRejectedRoots'] = 1
            candidate.write_text(json.dumps(value) + '\n')
            result = run_tool(root, candidate, producer, evidence)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('not a zero-counterexample closure', result.stderr)

    def test_assembled_byte_change_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, producer, evidence = make_fixture(root)
            (producer / 'assembled' / 'yellow' / '14-16.policy.bin').write_bytes(b'changed')
            write_sums(producer)
            value = json.loads(candidate.read_text())
            value['checksums']['roundManifest'] = digest((producer / 'SHA256SUMS').read_bytes())
            candidate.write_text(json.dumps(value) + '\n')
            result = run_tool(root, candidate, producer, evidence)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('disagree on assembled/14-16.policy.bin', result.stderr)


if __name__ == '__main__':
    unittest.main()
