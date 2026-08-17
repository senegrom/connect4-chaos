#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("perfect-chaos-download-artifact.py")
SPEC = importlib.util.spec_from_file_location("perfect_chaos_download_artifact", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

ArtifactDownloadError = MODULE.ArtifactDownloadError


def archive(files: dict[str, bytes], *, year: int = 2025, symlink: bool = False) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name, data in files.items():
            info = zipfile.ZipInfo(name, (year, 1, 1, 0, 0, 0))
            if symlink:
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zipped.writestr(info, data)
    return buffer.getvalue()


def row(artifact_id: int, payload: bytes, *, name: str, run: int, sha: str) -> dict:
    return {
        "id": artifact_id,
        "name": name,
        "expired": False,
        "size_in_bytes": len(payload),
        "digest": f"sha256:{hashlib.sha256(payload).hexdigest()}",
        "created_at": f"2026-08-17T00:00:{artifact_id:02d}Z",
        "expires_at": "2026-09-17T00:00:00Z",
        "workflow_run": {"id": run, "head_sha": sha},
    }


class DownloadArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run = 42
        self.sha = "a" * 40
        self.name = "perfect-chaos-red-18-40047-round"

    def test_accepts_equivalent_rerun_archives_and_selects_latest_id(self) -> None:
        first = archive({"nested/table.bin": b"proof", "summary.json": b"{}\n"}, year=2025)
        second = archive({"nested/table.bin": b"proof", "summary.json": b"{}\n"}, year=2026)
        self.assertNotEqual(hashlib.sha256(first).digest(), hashlib.sha256(second).digest())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = MODULE.download_equivalent_artifact(
                repository="owner/repo",
                run_id=self.run,
                run_sha=self.sha,
                artifact_name=self.name,
                output=root / "out",
                metadata=root / "audit.json",
                token="unused",
                artifacts=[
                    row(10, first, name=self.name, run=self.run, sha=self.sha),
                    row(11, second, name=self.name, run=self.run, sha=self.sha),
                ],
                payloads={10: first, 11: second},
            )
            self.assertEqual(summary["matchingArchives"], 2)
            self.assertEqual(summary["selectedArtifactId"], 11)
            self.assertEqual((root / "out/nested/table.bin").read_bytes(), b"proof")
            self.assertEqual(json.loads((root / "audit.json").read_text())["files"], summary["files"])

    def test_rejects_same_name_with_different_extracted_bytes(self) -> None:
        first = archive({"proof.bin": b"safe"})
        second = archive({"proof.bin": b"different"})
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ArtifactDownloadError, "not byte-equivalent"):
                MODULE.download_equivalent_artifact(
                    repository="owner/repo",
                    run_id=self.run,
                    run_sha=self.sha,
                    artifact_name=self.name,
                    output=Path(temporary) / "out",
                    metadata=Path(temporary) / "audit.json",
                    token="unused",
                    artifacts=[
                        row(1, first, name=self.name, run=self.run, sha=self.sha),
                        row(2, second, name=self.name, run=self.run, sha=self.sha),
                    ],
                    payloads={1: first, 2: second},
                )

    def test_rejects_wrong_identity_and_bad_archive_digest(self) -> None:
        payload = archive({"proof.bin": b"safe"})
        wrong = row(1, payload, name=self.name, run=self.run + 1, sha=self.sha)
        with self.assertRaisesRegex(ArtifactDownloadError, "wrong producer identity"):
            MODULE.validate_named_artifacts(
                [wrong], run_id=self.run, run_sha=self.sha, artifact_name=self.name
            )
        valid = row(1, payload, name=self.name, run=self.run, sha=self.sha)
        valid["digest"] = f"sha256:{'0' * 64}"
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ArtifactDownloadError, "archive digest mismatch"):
                MODULE.download_equivalent_artifact(
                    repository="owner/repo",
                    run_id=self.run,
                    run_sha=self.sha,
                    artifact_name=self.name,
                    output=Path(temporary) / "out",
                    metadata=Path(temporary) / "audit.json",
                    token="unused",
                    artifacts=[valid],
                    payloads={1: payload},
                )

    def test_rejects_unsafe_zip_entries_and_nonempty_output(self) -> None:
        for payload in (
            archive({"../escape": b"x"}),
            archive({"link": b"target"}, symlink=True),
        ):
            with self.subTest(payload=hashlib.sha256(payload).hexdigest()):
                with self.assertRaisesRegex(ArtifactDownloadError, "unsafe ZIP entry"):
                    MODULE.inspect_archive(payload)
        payload = archive({"proof.bin": b"safe"})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "out"
            output.mkdir()
            (output / "existing").write_bytes(b"x")
            with self.assertRaisesRegex(ArtifactDownloadError, "output must be absent or an empty"):
                MODULE.download_equivalent_artifact(
                    repository="owner/repo",
                    run_id=self.run,
                    run_sha=self.sha,
                    artifact_name=self.name,
                    output=output,
                    metadata=Path(temporary) / "audit.json",
                    token="unused",
                    artifacts=[row(1, payload, name=self.name, run=self.run, sha=self.sha)],
                    payloads={1: payload},
                )

    def test_rejects_missing_or_expired_artifact(self) -> None:
        payload = archive({"proof.bin": b"safe"})
        expired = row(1, payload, name=self.name, run=self.run, sha=self.sha)
        expired["expired"] = True
        with self.assertRaisesRegex(ArtifactDownloadError, "No unexpired artifact"):
            MODULE.validate_named_artifacts(
                [expired], run_id=self.run, run_sha=self.sha, artifact_name=self.name
            )


if __name__ == "__main__":
    unittest.main()
