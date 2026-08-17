#!/usr/bin/env python3
"""Patch Perfect Chaos merge jobs to use complete digest-bound shard downloads."""

from __future__ import annotations

from pathlib import Path

DOWNLOADER = Path("scripts/perfect-chaos-download-shards.py")
TESTS = Path("scripts/test-perfect-chaos-download-shards.py")
ROUND_16 = Path(".github/workflows/reusable-perfect-chaos-16-round.yml")
ROUND_18 = Path(".github/workflows/reusable-perfect-chaos-18-round.yml")
ROUND_WORKFLOWS = [ROUND_16, ROUND_18]
UPLOAD_WORKFLOWS = [
    *ROUND_WORKFLOWS,
    Path(".github/workflows/reusable-perfect-chaos-16-evidenced-round.yml"),
    Path(".github/workflows/reusable-perfect-chaos-18-evidenced-round.yml"),
    Path(".github/workflows/reusable-perfect-chaos-auto-advance.yml"),
]


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one replacement target, got {count}.")
    return text.replace(old, new, 1)


def patch_downloader() -> None:
    text = DOWNLOADER.read_text()
    start = text.index("def validate_artifacts(\n")
    end = text.index("\n\ndef extract_archive", start)
    replacement = '''def select_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    run_id: int,
    run_sha: str,
    prefix: str,
    shard_count: int,
    allow_missing: int = 0,
) -> tuple[list[dict[str, Any]], list[int]]:
    if shard_count < 1 or shard_count > 512:
        raise ShardDownloadError("shard-count must be between 1 and 512.")
    if allow_missing < 0 or allow_missing > 8:
        raise ShardDownloadError("allow-missing must be between 0 and 8.")
    if not re.fullmatch(r"[0-9a-f]{40}", run_sha):
        raise ShardDownloadError("run-sha must be a lowercase 40-character SHA.")
    pattern = re.compile(re.escape(prefix) + r"(\\d+)\\Z")
    selected: dict[int, dict[str, Any]] = {}
    for artifact in artifacts:
        name = artifact.get("name")
        match = pattern.fullmatch(name) if isinstance(name, str) else None
        if not match:
            continue
        index = int(match.group(1))
        if index < 0 or index >= shard_count:
            raise ShardDownloadError(f"Shard artifact index is out of range: {name!r}.")
        if index in selected:
            raise ShardDownloadError(f"Duplicate shard artifact index: {index}.")
        if artifact.get("expired") is not False:
            raise ShardDownloadError(f"Shard artifact is expired: {name!r}.")
        workflow = artifact.get("workflow_run")
        if not isinstance(workflow, dict) or workflow.get("id") != run_id \
                or workflow.get("head_sha") != run_sha:
            raise ShardDownloadError(f"Shard artifact has the wrong producer identity: {name!r}.")
        digest = artifact.get("digest")
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ShardDownloadError(f"Shard artifact has no valid SHA-256 digest: {name!r}.")
        size = artifact.get("size_in_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ShardDownloadError(f"Shard artifact has an invalid size: {name!r}.")
        artifact_id = artifact.get("id")
        if isinstance(artifact_id, bool) or not isinstance(artifact_id, int) or artifact_id < 1:
            raise ShardDownloadError(f"Shard artifact has an invalid id: {name!r}.")
        selected[index] = {
            "index": index,
            "id": artifact_id,
            "name": name,
            "sha256": digest.removeprefix("sha256:"),
            "sizeInBytes": size,
            "createdAt": artifact.get("created_at"),
            "expiresAt": artifact.get("expires_at"),
        }
    missing = sorted(set(range(shard_count)).difference(selected))
    if len(missing) > allow_missing:
        preview = missing[:16]
        suffix = "..." if len(missing) > len(preview) else ""
        raise ShardDownloadError(
            f"Missing {len(missing)} shard artifact indexes: {preview}{suffix}; "
            f"allow-missing is {allow_missing}."
        )
    return [selected[index] for index in sorted(selected)], missing


def validate_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    run_id: int,
    run_sha: str,
    prefix: str,
    shard_count: int,
) -> list[dict[str, Any]]:
    rows, _ = select_artifacts(
        artifacts,
        run_id=run_id,
        run_sha=run_sha,
        prefix=prefix,
        shard_count=shard_count,
        allow_missing=0,
    )
    return rows
'''
    text = text[:start] + replacement + text[end:]
    text = replace_once(
        text,
        '    parser.add_argument("--shard-count", required=True, type=int)\n',
        '    parser.add_argument("--shard-count", required=True, type=int)\n'
        '    parser.add_argument("--allow-missing", default=0, type=int)\n',
        label=str(DOWNLOADER),
    )
    text = replace_once(
        text,
        '''    rows = validate_artifacts(
        artifacts,
        run_id=args.run_id,
        run_sha=args.run_sha,
        prefix=args.artifact_prefix,
        shard_count=args.shard_count,
    )
''',
        '''    rows, missing = select_artifacts(
        artifacts,
        run_id=args.run_id,
        run_sha=args.run_sha,
        prefix=args.artifact_prefix,
        shard_count=args.shard_count,
        allow_missing=args.allow_missing,
    )
''',
        label=str(DOWNLOADER),
    )
    text = replace_once(
        text,
        '''    if [item["index"] for item in results] != list(range(args.shard_count)):
        raise ShardDownloadError("Downloaded shard set is incomplete.")
''',
        '''    expected_indexes = [row["index"] for row in rows]
    if [item["index"] for item in results] != expected_indexes:
        raise ShardDownloadError("Downloaded shard set differs from the selected artifacts.")
''',
        label=str(DOWNLOADER),
    )
    text = replace_once(
        text,
        '''        "shardCount": args.shard_count,
        "artifacts": results,
''',
        '''        "shardCount": args.shard_count,
        "downloadedShards": len(results),
        "missingShards": missing,
        "artifacts": results,
''',
        label=str(DOWNLOADER),
    )
    text = replace_once(
        text,
        '''        "shards": len(results),
        "files": len(results) * 4,
''',
        '''        "shards": len(results),
        "missing": len(missing),
        "files": len(results) * 4,
''',
        label=str(DOWNLOADER),
    )
    DOWNLOADER.write_text(text)


def patch_tests() -> None:
    text = TESTS.read_text()
    anchor = '''    def test_archive_extracts_only_the_four_expected_flat_files(self) -> None:
'''
    addition = '''    def test_bounded_missing_set_is_reported_for_merge_recovery(self) -> None:
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

'''
    text = replace_once(text, anchor, addition + anchor, label=str(TESTS))
    TESTS.write_text(text)


def add_overwrite(path: Path) -> None:
    lines = path.read_text().splitlines()
    output: list[str] = []
    index = 0
    changed = 0
    while index < len(lines):
        line = lines[index]
        output.append(line)
        if line.strip() == "- uses: actions/upload-artifact@v4":
            step_indent = len(line) - len(line.lstrip())
            cursor = index + 1
            while cursor < len(lines):
                candidate = lines[cursor]
                candidate_indent = len(candidate) - len(candidate.lstrip())
                if candidate.strip().startswith("- ") and candidate_indent == step_indent:
                    raise RuntimeError(f"{path}: upload step has no with block.")
                output.append(candidate)
                if candidate.strip() == "with:":
                    with_indent = candidate_indent
                    next_index = cursor + 1
                    has_overwrite = False
                    while next_index < len(lines):
                        following = lines[next_index]
                        following_indent = len(following) - len(following.lstrip())
                        if following.strip() and following_indent <= with_indent:
                            break
                        if following.strip().startswith("overwrite:"):
                            has_overwrite = True
                            break
                        next_index += 1
                    if not has_overwrite:
                        output.append(" " * (with_indent + 2) + "overwrite: true")
                        changed += 1
                    index = cursor
                    break
                cursor += 1
        index += 1
    if changed == 0:
        raise RuntimeError(f"{path}: no upload steps were patched.")
    path.write_text("\n".join(output) + "\n")


def strict_download_step() -> str:
    return '''      - name: Download the complete deterministic shard set with digest verification
        shell: bash
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          set -euo pipefail
          rm -rf shards merge-artifact-audit
          mkdir -p shards merge-artifact-audit
          python3 scripts/perfect-chaos-download-shards.py \\
            --repository "$GITHUB_REPOSITORY" \\
            --run-id "$GITHUB_RUN_ID" \\
            --run-sha "$GITHUB_SHA" \\
            --artifact-prefix '${{ inputs.shard_prefix }}' \\
            --shard-count "$SHARD_COUNT" \\
            --output shards \\
            --metadata merge-artifact-audit/shards.json
          rm -rf merge-artifact-audit/.shard-archives
          test "$(jq -r '.missingShards | length' merge-artifact-audit/shards.json)" = '0'

'''


def bounded_download_step() -> str:
    return '''      - name: Download all available deterministic shards with digest verification
        shell: bash
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          set -euo pipefail
          rm -rf shards merge-artifact-audit
          mkdir -p shards merge-artifact-audit
          python3 scripts/perfect-chaos-download-shards.py \\
            --repository "$GITHUB_REPOSITORY" \\
            --run-id "$GITHUB_RUN_ID" \\
            --run-sha "$GITHUB_SHA" \\
            --artifact-prefix '${{ inputs.shard_prefix }}' \\
            --shard-count "$SHARD_COUNT" \\
            --allow-missing 8 \\
            --output shards \\
            --metadata merge-artifact-audit/shards.json
          rm -rf merge-artifact-audit/.shard-archives

'''


def patch_round_workflows() -> None:
    strict_old = '''      - uses: actions/download-artifact@v5
        with:
          pattern: ${{ inputs.shard_prefix }}*
          path: shards
          merge-multiple: true

'''
    bounded_old = '''      - uses: actions/download-artifact@v5
        continue-on-error: true
        with:
          pattern: ${{ inputs.shard_prefix }}*
          path: shards
          merge-multiple: true

'''
    text16 = ROUND_16.read_text()
    text16 = replace_once(text16, strict_old, strict_download_step(), label=str(ROUND_16))
    if "pattern: ${{ inputs.shard_prefix }}*" in text16:
        raise RuntimeError(f"{ROUND_16}: stale wildcard shard download remains.")
    ROUND_16.write_text(text16)

    text18 = ROUND_18.read_text()
    text18 = replace_once(text18, bounded_old, bounded_download_step(), label=str(ROUND_18))
    old_recovery = '''          : > /tmp/recovered-shards.txt
          if (( ${#missing[@]} > 0 )); then
'''
    new_recovery = '''          : > /tmp/actual-missing-shards.txt
          if (( ${#missing[@]} > 0 )); then
            printf '%s\\n' "${missing[@]}" > /tmp/actual-missing-shards.txt
          fi
          jq -r '.missingShards[]' merge-artifact-audit/shards.json \\
            > /tmp/recorded-missing-shards.txt
          diff -u /tmp/recorded-missing-shards.txt /tmp/actual-missing-shards.txt

          : > /tmp/recovered-shards.txt
          if (( ${#missing[@]} > 0 )); then
'''
    text18 = replace_once(text18, old_recovery, new_recovery, label=str(ROUND_18))
    if "pattern: ${{ inputs.shard_prefix }}*" in text18:
        raise RuntimeError(f"{ROUND_18}: stale wildcard shard download remains.")
    ROUND_18.write_text(text18)


def main() -> None:
    patch_downloader()
    patch_tests()
    patch_round_workflows()
    for path in UPLOAD_WORKFLOWS:
        add_overwrite(path)
    for path in [DOWNLOADER, TESTS, *UPLOAD_WORKFLOWS]:
        text = path.read_text()
        if "\t" in text:
            raise RuntimeError(f"{path}: tabs are forbidden.")
    print("Patched paginated merge downloads, bounded recovery, and rerun-safe uploads.")


if __name__ == "__main__":
    main()
