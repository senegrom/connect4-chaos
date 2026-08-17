#!/usr/bin/env python3
"""Unit tests for paginated, digest-bound Perfect Chaos shard downloads."""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from unittest import mock
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
        with self.assertRaisesRegex(DOWNLOADER.ShardDownloadError, "Missing 1 shard"):
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

    def test_bounded_missing_set_is_reported_for_merge_recovery(self) -> None:
        rows, missing = DOWNLOADER.select_artifacts(
            [artifact(0), artifact(2), artifact(4)],
            run_id=77,
            run_sha=RUN_SHA,
            prefix="prefix-",
            shard_count=5,
            allow_missing=2,
        )
        self.assertEqual([row["index"] for row in rows], [0, 2, 4])
        self.assertEqual(missing, [1, 3])
        with self.assertRaisesRegex(DOWNLOADER.ShardDownloadError, "Missing 2 shard"):
            DOWNLOADER.select_artifacts(
                [artifact(0), artifact(2), artifact(4)],
                run_id=77,
                run_sha=RUN_SHA,
                prefix="prefix-",
                shard_count=5,
                allow_missing=1,
            )
        with self.assertRaisesRegex(DOWNLOADER.ShardDownloadError, "allow-missing"):
            DOWNLOADER.select_artifacts(
                [artifact(0)],
                run_id=77,
                run_sha=RUN_SHA,
                prefix="prefix-",
                shard_count=1,
                allow_missing=9,
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


    def test_artifact_retry_schedule_and_successful_retry(self) -> None:
        url = "https://api.github.com/repos/owner/repo/actions/artifacts/123/zip"
        delays = DOWNLOADER.artifact_retry_delays(url)
        self.assertEqual(delays, DOWNLOADER.artifact_retry_delays(url))
        self.assertEqual(len(delays), 7)
        self.assertTrue(all(left <= right for left, right in zip(delays, delays[1:])))
        self.assertLessEqual(max(delays), 61.0)

        failed = mock.Mock(returncode=22, stdout=b"", stderr=b"HTTP 503")
        succeeded = mock.Mock(returncode=0, stdout=b"proof", stderr=b"")
        with mock.patch.object(DOWNLOADER.subprocess, "run", side_effect=[failed, succeeded]) as run, \
                mock.patch.object(DOWNLOADER.time, "sleep") as sleep:
            payload = DOWNLOADER.request_artifact_bytes(url, "token")
        self.assertEqual(payload, b"proof")
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(delays[0])
        command = run.call_args_list[0].args[0]
        self.assertIn("--connect-timeout", command)
        self.assertIn("--max-time", command)

    def test_artifact_retry_exhaustion_fails_closed(self) -> None:
        url = "https://api.github.com/repos/owner/repo/actions/artifacts/999/zip"
        failed = mock.Mock(returncode=22, stdout=b"", stderr=b"HTTP 503")
        with mock.patch.object(DOWNLOADER.subprocess, "run", return_value=failed) as run, \
                mock.patch.object(DOWNLOADER.time, "sleep") as sleep:
            with self.assertRaisesRegex(DOWNLOADER.ShardDownloadError, "after 8 outer attempts"):
                DOWNLOADER.request_artifact_bytes(url, "token")
        self.assertEqual(run.call_count, 8)
        self.assertEqual(sleep.call_count, 7)


if __name__ == "__main__":
    unittest.main()
