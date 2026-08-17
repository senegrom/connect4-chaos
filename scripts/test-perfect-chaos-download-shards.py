#!/usr/bin/env python3
"""Unit tests for paginated, digest-bound Perfect Chaos shard downloads."""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).with_name("perfect-chaos-download-shards.py")
SPEC = importlib.util.spec_from_file_location("perfect_chaos_download_shards", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load the shard downloader.")
DOWNLOADER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DOWNLOADER
SPEC.loader.exec_module(DOWNLOADER)

RUN_SHA = "a" * 40


def artifact(index: int, *, name: str | None = None, run: int = 77, sha: str = RUN_SHA) -> dict:
    return {
        "id": 1000 + index,
        "name": name or f"prefix-{index}",
        "size_in_bytes": 123,
        "expired": False,
        "digest": "sha256:" + (f"{index + 1:064x}"[-64:]),
        "workflow_run": {"id": run, "head_sha": sha},
        "created_at": "2026-01-01T00:00:00Z",
        "expires_at": "2026-02-01T00:00:00Z",
    }


def archive(index: int, *, unsafe: bool = False) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as zipped:
        for name in DOWNLOADER.expected_files(index):
            zipped.writestr(name, name.encode())
        if unsafe:
            zipped.writestr("../escape.bin", b"escape")
    return output.getvalue()


class ShardDownloaderTests(unittest.TestCase):
    def test_exact_contiguous_artifact_set_is_bound_to_run_and_sha(self) -> None:
        rows = DOWNLOADER.validate_artifacts(
            [artifact(2), artifact(0), artifact(1), artifact(99, name="other")],
            run_id=77,
            run_sha=RUN_SHA,
            prefix="prefix-",
            shard_count=3,
        )
        self.assertEqual([row["index"] for row in rows], [0, 1, 2])
        self.assertEqual([row["id"] for row in rows], [1000, 1001, 1002])

    def test_missing_duplicate_and_wrong_producer_fail_closed(self) -> None:
        with self.assertRaisesRegex(DOWNLOADER.ShardDownloadError, "Missing shard"):
            DOWNLOADER.validate_artifacts(
                [artifact(0)], run_id=77, run_sha=RUN_SHA,
                prefix="prefix-", shard_count=2,
            )
        with self.assertRaisesRegex(DOWNLOADER.ShardDownloadError, "Duplicate shard"):
            DOWNLOADER.validate_artifacts(
                [artifact(0), artifact(0, name="prefix-00")],
                run_id=77, run_sha=RUN_SHA, prefix="prefix-", shard_count=1,
            )
        with self.assertRaisesRegex(DOWNLOADER.ShardDownloadError, "producer identity"):
            DOWNLOADER.validate_artifacts(
                [artifact(0, run=78)], run_id=77, run_sha=RUN_SHA,
                prefix="prefix-", shard_count=1,
            )

    def test_archive_extracts_only_the_four_expected_flat_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="perfect-chaos-shards-") as temporary:
            output = Path(temporary)
            summary = DOWNLOADER.extract_archive(archive(7), index=7, output=output)
            self.assertEqual(set(summary["files"]), DOWNLOADER.expected_files(7))
            self.assertEqual(
                {path.name for path in output.iterdir()},
                DOWNLOADER.expected_files(7),
            )

    def test_extra_or_unsafe_archive_entries_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="perfect-chaos-shards-") as temporary:
            with self.assertRaisesRegex(DOWNLOADER.ShardDownloadError, "entries differ"):
                DOWNLOADER.extract_archive(
                    archive(3, unsafe=True), index=3, output=Path(temporary)
                )


if __name__ == "__main__":
    unittest.main()
