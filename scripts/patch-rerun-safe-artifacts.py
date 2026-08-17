#!/usr/bin/env python3
from pathlib import Path
import re

ROUND_FILES = [
    Path('.github/workflows/reusable-perfect-chaos-16-round.yml'),
    Path('.github/workflows/reusable-perfect-chaos-18-round.yml'),
]

round_pattern = re.compile(
    r'(?ms)^          artifact_count=\$\(gh api --paginate .*?'
    r'^            --name "\$SOURCE_ARTIFACT" --dir previous\n'
)
round_replacement = '''          rm -rf previous seed prepared source-artifact-audit
          mkdir -p "seed/$ROLE" source-artifact-audit
          python3 scripts/perfect-chaos-download-artifact.py \\
            --repository "$REPO" \\
            --run-id "$SOURCE_RUN" \\
            --run-sha "$SOURCE_SHA" \\
            --artifact-name "$SOURCE_ARTIFACT" \\
            --output previous \\
            --metadata source-artifact-audit/download.json
          rm -rf source-artifact-audit/.named-artifact-archives
'''

for path in ROUND_FILES:
    text = path.read_text()
    text, count = round_pattern.subn(round_replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f'{path}: expected exactly one unique-source download block, got {count}')
    if 'artifact_count=$(gh api --paginate' in text or '--name "$SOURCE_ARTIFACT" --dir previous' in text:
        raise RuntimeError(f'{path}: stale unique-artifact logic remains')
    path.write_text(text)

auto_path = Path('.github/workflows/reusable-perfect-chaos-auto-advance.yml')
auto = auto_path.read_text()
auto_pattern = re.compile(
    r'(?ms)^      - name: Download the unique producer and independent evidence artifacts\n'
    r'.*?(?=^      - name: Derive the next state only from matching independently audited bytes\n)'
)
auto_replacement = '''      - name: Download equivalent producer and independent evidence artifacts
        id: artifacts
        shell: bash
        env:
          RUN_ID: ${{ steps.source.outputs.run_id }}
          RUN_SHA: ${{ steps.source.outputs.run_sha }}
          RESULT_ARTIFACT: ${{ steps.source.outputs.result_artifact }}
          EVIDENCE_ARTIFACT: ${{ steps.source.outputs.evidence_artifact }}
        run: |
          set -euo pipefail
          rm -rf producer evidence artifact-download-audit
          mkdir -p artifact-download-audit
          python3 scripts/perfect-chaos-download-artifact.py \\
            --repository "$REPOSITORY" \\
            --run-id "$RUN_ID" \\
            --run-sha "$RUN_SHA" \\
            --artifact-name "$RESULT_ARTIFACT" \\
            --output producer \\
            --metadata artifact-download-audit/producer.json
          python3 scripts/perfect-chaos-download-artifact.py \\
            --repository "$REPOSITORY" \\
            --run-id "$RUN_ID" \\
            --run-sha "$RUN_SHA" \\
            --artifact-name "$EVIDENCE_ARTIFACT" \\
            --output evidence \\
            --metadata artifact-download-audit/evidence.json
          rm -rf artifact-download-audit/.named-artifact-archives

          jq '. as $root | $root.artifacts[]
              | select(.id == $root.selectedArtifactId)
              | {
                  id,
                  name,
                  size_in_bytes: .sizeInBytes,
                  digest: ("sha256:" + .sha256),
                  created_at: .createdAt,
                  expires_at: .expiresAt,
                  workflow_run: {id: $root.run, head_sha: $root.runSha},
                  equivalent_archives: $root.matchingArchives,
                  content_manifest_sha256: $root.contentManifestSha256
                }' artifact-download-audit/producer.json > producer-artifact.json
          jq '. as $root | $root.artifacts[]
              | select(.id == $root.selectedArtifactId)
              | {
                  id,
                  name,
                  size_in_bytes: .sizeInBytes,
                  digest: ("sha256:" + .sha256),
                  created_at: .createdAt,
                  expires_at: .expiresAt,
                  workflow_run: {id: $root.run, head_sha: $root.runSha},
                  equivalent_archives: $root.matchingArchives,
                  content_manifest_sha256: $root.contentManifestSha256
                }' artifact-download-audit/evidence.json > evidence-artifact.json

'''
auto, count = auto_pattern.subn(auto_replacement, auto, count=1)
if count != 1:
    raise RuntimeError(f'{auto_path}: expected one artifact-download step, got {count}')
if 'Expected one unexpired artifact named' in auto or 'gh run download "$RUN_ID"' in auto:
    raise RuntimeError(f'{auto_path}: stale unique-artifact logic remains')
needle = '''            next-state.json
'''
replacement = '''            next-state.json
            artifact-download-audit/producer.json
            artifact-download-audit/evidence.json
'''
if auto.count(needle) != 1:
    raise RuntimeError(f'{auto_path}: expected one transition audit path insertion point')
auto = auto.replace(needle, replacement, 1)
auto_path.write_text(auto)

ci_path = Path('.github/workflows/ci.yml')
ci = ci_path.read_text()
anchor = '''      - name: Test paginated Chaos shard downloader
        run: python3 scripts/test-perfect-chaos-download-shards.py
'''
addition = anchor + '''
      - name: Test rerun-safe named artifact downloader
        run: python3 scripts/test-perfect-chaos-download-artifact.py
'''
if ci.count(anchor) != 1:
    raise RuntimeError(f'{ci_path}: expected paginated downloader test anchor once')
ci = ci.replace(anchor, addition, 1)
ci_path.write_text(ci)

for path in [*ROUND_FILES, auto_path, ci_path]:
    text = path.read_text()
    if '\t' in text:
        raise RuntimeError(f'{path}: tabs are not allowed')

print('Patched rerun-safe artifact selection into production workflows.')
