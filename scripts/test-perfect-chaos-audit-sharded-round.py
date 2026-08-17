#!/usr/bin/env python3
"""Regression tests for durable manifest identity in the independent sharded auditor."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("perfect-chaos-audit-sharded-round.py")
SPEC = importlib.util.spec_from_file_location("perfect_chaos_audit_sharded_round", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load the independent sharded auditor.")
AUDITOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDITOR
SPEC.loader.exec_module(AUDITOR)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class DurableManifestIdentityTests(unittest.TestCase):
    def test_legacy_missing_transient_entry_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory(prefix="perfect-chaos-audit-manifest-") as temporary:
            root = Path(temporary)
            proof = b"proof"
            (root / "proof.bin").write_bytes(proof)
            (root / "SHA256SUMS").write_text(
                f"{'0' * 64}  yellow/.incremental-repair-10-12/affected-existing-input.bin\n"
                f"{digest(proof)}  proof.bin\n"
            )
            self.assertEqual(
                AUDITOR.verify_sha256sums(root),
                {"proof.bin": digest(proof)},
            )

    def test_present_transient_files_are_not_durable_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="perfect-chaos-audit-manifest-") as temporary:
            root = Path(temporary)
            proof = b"proof"
            scratch = root / "red" / ".incremental-repair-12-14"
            scratch.mkdir(parents=True)
            (scratch / "repaired.policy.bin").write_bytes(b"scratch")
            (root / "proof.bin").write_bytes(proof)
            (root / "SHA256SUMS").write_text(f"{digest(proof)}  proof.bin\n")
            self.assertEqual(
                AUDITOR.verify_sha256sums(root),
                {"proof.bin": digest(proof)},
            )

    def test_near_match_remains_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="perfect-chaos-audit-manifest-") as temporary:
            root = Path(temporary)
            proof = b"proof"
            (root / "proof.bin").write_bytes(proof)
            (root / "SHA256SUMS").write_text(
                f"{'0' * 64}  yellow/.incremental-repair-ten-twelve/missing.bin\n"
                f"{digest(proof)}  proof.bin\n"
            )
            with self.assertRaisesRegex(AUDITOR.AuditError, "missing or unsafe file"):
                AUDITOR.verify_sha256sums(root)

    def test_duplicate_transient_entries_remain_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="perfect-chaos-audit-manifest-") as temporary:
            root = Path(temporary)
            proof = b"proof"
            transient = "yellow/.incremental-repair-10-12/affected-existing-input.bin"
            (root / "proof.bin").write_bytes(proof)
            (root / "SHA256SUMS").write_text(
                f"{'0' * 64}  {transient}\n"
                f"{'1' * 64}  {transient}\n"
                f"{digest(proof)}  proof.bin\n"
            )
            with self.assertRaisesRegex(AUDITOR.AuditError, "Duplicate SHA256SUMS path"):
                AUDITOR.verify_sha256sums(root)


if __name__ == "__main__":
    unittest.main()
