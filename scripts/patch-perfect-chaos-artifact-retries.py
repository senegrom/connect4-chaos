#!/usr/bin/env python3
"""Add bounded exponential retries to Perfect Chaos artifact downloads."""

from __future__ import annotations

from pathlib import Path

TARGETS = [
    (Path("scripts/perfect-chaos-download-shards.py"), "ShardDownloadError"),
    (Path("scripts/perfect-chaos-download-artifact.py"), "ArtifactDownloadError"),
]
TESTS = [
    (Path("scripts/test-perfect-chaos-download-shards.py"), "DOWNLOADER", "ShardDownloadError"),
    (Path("scripts/test-perfect-chaos-download-artifact.py"), "MODULE", "ArtifactDownloadError"),
]


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one replacement target, got {count}.")
    return text.replace(old, new, 1)


def retry_block(error_class: str) -> str:
    return f'''ARTIFACT_DOWNLOAD_ATTEMPTS = 8
ARTIFACT_DOWNLOAD_BASE_DELAY_SECONDS = 2.0
ARTIFACT_DOWNLOAD_MAX_DELAY_SECONDS = 60.0


def artifact_retry_delays(url: str, *, attempts: int = ARTIFACT_DOWNLOAD_ATTEMPTS) -> list[float]:
    """Return a deterministic exponential schedule with small per-URL jitter."""
    if attempts < 1 or attempts > 16:
        raise {error_class}("artifact download attempts must be between 1 and 16.")
    seed = int(hashlib.sha256(url.encode("utf-8")).hexdigest()[:8], 16)
    delays: list[float] = []
    for attempt in range(attempts - 1):
        base = min(
            ARTIFACT_DOWNLOAD_MAX_DELAY_SECONDS,
            ARTIFACT_DOWNLOAD_BASE_DELAY_SECONDS * (2 ** attempt),
        )
        jitter = ((seed >> ((attempt % 8) * 4)) & 0xF) / 16.0
        delays.append(base + jitter)
    return delays


def request_artifact_bytes(url: str, token: str) -> bytes:
    """Follow GitHub's signed redirect with bounded exponential storage retries."""
    delays = artifact_retry_delays(url)
    errors: list[str] = []
    for attempt in range(len(delays) + 1):
        completed = subprocess.run(
            [
                "curl", "--fail", "--location", "--silent", "--show-error",
                "--retry", "2", "--retry-delay", "1", "--retry-all-errors",
                "--connect-timeout", "30", "--max-time", "240",
                "-H", f"Authorization: Bearer {{token}}",
                "-H", "Accept: application/vnd.github+json",
                "-H", f"X-GitHub-Api-Version: {{API_VERSION}}",
                "-H", "User-Agent: connect4-chaos-perfect-proof-auditor",
                url,
            ],
            check=False,
            capture_output=True,
        )
        if completed.returncode == 0:
            return completed.stdout
        details = completed.stderr.decode("utf-8", errors="replace").strip()
        errors.append(f"attempt {{attempt + 1}}: curl {{completed.returncode}}: {{details}}")
        if attempt < len(delays):
            time.sleep(delays[attempt])
    raise {error_class}(
        f"Artifact download failed after {{len(errors)}} outer attempts; {{errors[-1]}}"
    )
'''


def patch_target(path: Path, error_class: str) -> None:
    text = path.read_text()
    text = replace_once(
        text,
        "import subprocess\n",
        "import subprocess\nimport time\n",
        label=str(path),
    )
    start = text.index("def request_artifact_bytes(url: str, token: str) -> bytes:\n")
    end = text.index("\n\ndef request_json", start)
    text = text[:start] + retry_block(error_class) + text[end:]
    if path.name == "perfect-chaos-download-shards.py":
        text = replace_once(
            text,
            '    parser.add_argument("--workers", default=16, type=int)\n',
            '    parser.add_argument("--workers", default=8, type=int)\n',
            label=str(path),
        )
    path.write_text(text)


def patch_test(path: Path, module_name: str, error_class: str) -> None:
    text = path.read_text()
    text = replace_once(
        text,
        "import unittest\n",
        "import unittest\nfrom unittest import mock\n",
        label=str(path),
    )
    anchor = "\n\nif __name__ == \"__main__\":\n"
    addition = f'''
    def test_artifact_retry_schedule_and_successful_retry(self) -> None:
        url = "https://api.github.com/repos/owner/repo/actions/artifacts/123/zip"
        delays = {module_name}.artifact_retry_delays(url)
        self.assertEqual(delays, {module_name}.artifact_retry_delays(url))
        self.assertEqual(len(delays), 7)
        self.assertTrue(all(left <= right for left, right in zip(delays, delays[1:])))
        self.assertLessEqual(max(delays), 61.0)

        failed = mock.Mock(returncode=22, stdout=b"", stderr=b"HTTP 503")
        succeeded = mock.Mock(returncode=0, stdout=b"proof", stderr=b"")
        with mock.patch.object({module_name}.subprocess, "run", side_effect=[failed, succeeded]) as run, \\
                mock.patch.object({module_name}.time, "sleep") as sleep:
            payload = {module_name}.request_artifact_bytes(url, "token")
        self.assertEqual(payload, b"proof")
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(delays[0])
        command = run.call_args_list[0].args[0]
        self.assertIn("--connect-timeout", command)
        self.assertIn("--max-time", command)

    def test_artifact_retry_exhaustion_fails_closed(self) -> None:
        url = "https://api.github.com/repos/owner/repo/actions/artifacts/999/zip"
        failed = mock.Mock(returncode=22, stdout=b"", stderr=b"HTTP 503")
        with mock.patch.object({module_name}.subprocess, "run", return_value=failed) as run, \\
                mock.patch.object({module_name}.time, "sleep") as sleep:
            with self.assertRaisesRegex({module_name}.{error_class}, "after 8 outer attempts"):
                {module_name}.request_artifact_bytes(url, "token")
        self.assertEqual(run.call_count, 8)
        self.assertEqual(sleep.call_count, 7)
'''
    text = replace_once(text, anchor, "\n" + addition + anchor, label=str(path))
    path.write_text(text)


def main() -> None:
    for path, error_class in TARGETS:
        patch_target(path, error_class)
    for path, module_name, error_class in TESTS:
        patch_test(path, module_name, error_class)
    for path, _ in TARGETS:
        if "\t" in path.read_text():
            raise RuntimeError(f"{path}: tabs are forbidden.")
    print("Patched bounded exponential artifact retries into both proof downloaders.")


if __name__ == "__main__":
    main()
