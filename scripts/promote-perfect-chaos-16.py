#!/usr/bin/env python3
"""Prepare a provenance-preserving 16-piece Perfect Chaos prefix promotion.

The 16-piece certificate was generated on a historical branch.  This tool copies
only the certificate and exact 18-piece rejection seeds, preserves the generator
source and original manifest, then teaches the current verifier to distinguish
historical generation identity from the current independent replay identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


EXPECTED_BOUNDARIES = [8, 10, 12, 14, 16]
EXPECTED_FINAL = {
    "red": {
        "policyEntries": 326_031,
        "frontierStates": 339_682,
        "closureStates": 747_775,
    },
    "yellow": {
        "policyEntries": 1_059_068,
        "frontierStates": 1_164_120,
        "closureStates": 2_498_257,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one replacement target, found {count}.")
    return text.replace(old, new, 1)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise RuntimeError(f"Certificate source is missing or unsafe: {source}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def validate_candidate(candidate: Path) -> dict:
    manifest_path = candidate / "data/perfect-chaos-prefix/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "connect4-chaos-layered-prefix-manifest-v1":
        raise RuntimeError("Candidate prefix manifest has the wrong format.")
    if manifest.get("boundaries") != EXPECTED_BOUNDARIES:
        raise RuntimeError("Candidate prefix does not reach exactly sixteen pieces.")

    for role, expected in EXPECTED_FINAL.items():
        segments = manifest.get("roles", {}).get(role, {}).get("replay", {}).get("segments")
        if not isinstance(segments, list) or not segments:
            raise RuntimeError(f"Candidate {role} replay has no segments.")
        final = segments[-1]
        actual = {
            "policyEntries": final.get("policyEntries"),
            "frontierStates": final.get("frontierStates"),
            "closureStates": final.get("closureStates"),
        }
        if final.get("fromPieces") != 14 or final.get("frontierPieces") != 16:
            raise RuntimeError(f"Candidate {role} final segment has the wrong boundaries.")
        if actual != expected:
            raise RuntimeError(
                f"Candidate {role} final metrics differ: expected {expected}, got {actual}."
            )

    generator = candidate / "native/perfect-chaos-prefix.cpp"
    if sha256(generator) != manifest.get("sourceSha256"):
        raise RuntimeError("Candidate generator source does not match its manifest.")
    return manifest


def patch_runtime(release: Path) -> None:
    path = release / "src/perfect-chaos-prefix.js"
    text = path.read_text()
    layer = "  Object.freeze({ fromBoundary: 14, boundary: 16, file: '14-16.policy.bin' }),"
    if layer in text:
        return
    anchor = "  Object.freeze({ fromBoundary: 12, boundary: 14, file: '12-14.policy.bin' }),\n"
    path.write_text(replace_once(text, anchor, anchor + layer + "\n", label=str(path)))


def patch_verifier(release: Path) -> None:
    path = release / "scripts/perfect-chaos-prefix.mjs"
    text = path.read_text()

    old_import = "import { basename, dirname, join, resolve } from 'node:path';"
    new_import = (
        "import { basename, dirname, isAbsolute, join, relative, resolve } "
        "from 'node:path';"
    )
    if new_import not in text:
        text = replace_once(text, old_import, new_import, label=str(path))

    old_guard = (
        "  const sourceHash = createHash('sha256').update(await readFile(SOURCE)).digest('hex');\n"
        "  if (reference.sourceSha256 !== sourceHash) {\n"
        "    throw new Error('The prefix solver source does not match the committed manifest.');\n"
        "  }\n"
    )
    new_guard = (
        "  const sourceHash = createHash('sha256').update(await readFile(SOURCE)).digest('hex');\n"
        "  const verificationSourceHash = reference.verificationSourceSha256 ?? reference.sourceSha256;\n"
        "  if (verificationSourceHash !== sourceHash) {\n"
        "    throw new Error('The prefix verifier source does not match the committed manifest.');\n"
        "  }\n"
        "  if (reference.generatorSource !== undefined) {\n"
        "    if (typeof reference.generatorSource !== 'string' || reference.generatorSource.length === 0) {\n"
        "      throw new Error('The prefix generator source path is invalid.');\n"
        "    }\n"
        "    const generatorPath = resolve(directory, reference.generatorSource);\n"
        "    const generatorRelative = relative(directory, generatorPath);\n"
        "    if (generatorRelative === '..' || generatorRelative.startsWith('../')\n"
        "        || generatorRelative.startsWith('..\\\\') || isAbsolute(generatorRelative)) {\n"
        "      throw new Error('The prefix generator source escapes the certificate directory.');\n"
        "    }\n"
        "    const generatorSourceHash = createHash('sha256')\n"
        "      .update(await readFile(generatorPath))\n"
        "      .digest('hex');\n"
        "    if (generatorSourceHash !== reference.sourceSha256) {\n"
        "      throw new Error('The preserved prefix generator source does not match the manifest.');\n"
        "    }\n"
        "  }\n"
    )
    if new_guard not in text:
        text = replace_once(text, old_guard, new_guard, label=str(path))
    path.write_text(text)


def patch_package(release: Path) -> None:
    path = release / "package.json"
    text = path.read_text()
    old = (
        '"chaos:prefix:generate": "node scripts/perfect-chaos-prefix.mjs generate '
        '--frontier-pieces 14 --seed-rejections data/perfect-chaos-prefix '
        '--shards 8 --shard-from-pieces 14 '
        '--output generated/perfect-chaos-prefix-14"'
    )
    new = (
        '"chaos:prefix:generate": "node scripts/perfect-chaos-prefix.mjs generate '
        '--frontier-pieces 16 --seed-rejections data/perfect-chaos-prefix '
        '--shards 8 --shard-from-pieces 14 '
        '--output generated/perfect-chaos-prefix-16"'
    )
    if new not in text:
        path.write_text(replace_once(text, old, new, label=str(path)))


def write_manifest_test(release: Path) -> None:
    path = release / "tests/perfect-chaos-prefix-manifest.test.js"
    path.write_text(
        """import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const MANIFEST_URL = new URL('../data/perfect-chaos-prefix/manifest.json', import.meta.url);
const DIRECTORY_URL = new URL('../data/perfect-chaos-prefix/', import.meta.url);
const SOURCE_URL = new URL('../native/perfect-chaos-prefix.cpp', import.meta.url);
const GENERATOR_SOURCE_URL = new URL(
  '../data/perfect-chaos-prefix/provenance/perfect-chaos-prefix-generator.cpp',
  import.meta.url,
);
const ORIGINAL_MANIFEST_URL = new URL(
  '../data/perfect-chaos-prefix/provenance/original-manifest.json',
  import.meta.url,
);

function digest(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

test('the committed Perfect Chaos prefix certificate reaches sixteen pieces', async () => {
  const manifest = JSON.parse(await readFile(MANIFEST_URL, 'utf8'));
  assert.equal(manifest.format, 'connect4-chaos-layered-prefix-manifest-v1');
  assert.deepEqual(manifest.boundaries, [8, 10, 12, 14, 16]);
  assert.equal(manifest.sourceSha256, digest(await readFile(GENERATOR_SOURCE_URL)));
  assert.equal(manifest.verificationSourceSha256, digest(await readFile(SOURCE_URL)));
  assert.equal(manifest.originalManifestSha256, digest(await readFile(ORIGINAL_MANIFEST_URL)));
  assert.equal(manifest.generatorRef, 'claude/connect4-chaos-ai-c1py3r');
  assert.match(manifest.generatorCommit, /^[0-9a-f]{40}$/);

  const expectedFinal = {
    red: { policyEntries: 326_031, frontierStates: 339_682, closureStates: 747_775 },
    yellow: {
      policyEntries: 1_059_068,
      frontierStates: 1_164_120,
      closureStates: 2_498_257,
    },
  };

  for (const role of ['red', 'yellow']) {
    const finalSegment = manifest.roles[role].replay.segments.at(-1);
    assert.equal(finalSegment.fromPieces, 14);
    assert.equal(finalSegment.frontierPieces, 16);
    assert.equal(finalSegment.terminalDraws, 0);
    assert.deepEqual(
      {
        policyEntries: finalSegment.policyEntries,
        frontierStates: finalSegment.frontierStates,
        closureStates: finalSegment.closureStates,
      },
      expectedFinal[role],
    );

    for (const artifact of manifest.artifacts[role]) {
      const bytes = await readFile(new URL(`${role}/${artifact.path}`, DIRECTORY_URL));
      assert.equal(bytes.length, artifact.bytes, `${role}/${artifact.path} byte length`);
      assert.equal(digest(bytes), artifact.sha256, `${role}/${artifact.path} digest`);
    }
  }
});
"""
    )


def install_candidate_files(
    release: Path,
    candidate: Path,
    *,
    candidate_ref: str,
    candidate_commit: str,
    verification_commit: str,
) -> None:
    source_prefix = candidate / "data/perfect-chaos-prefix"
    source_seeds = candidate / "data/perfect-chaos-prefix-seeds-18"
    destination_prefix = release / "data/perfect-chaos-prefix"
    destination_seeds = release / "data/perfect-chaos-prefix-seeds-18"
    copy_tree(source_prefix, destination_prefix)
    copy_tree(source_seeds, destination_seeds)

    provenance = destination_prefix / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    generator_source = provenance / "perfect-chaos-prefix-generator.cpp"
    original_manifest = provenance / "original-manifest.json"
    shutil.copy2(candidate / "native/perfect-chaos-prefix.cpp", generator_source)
    shutil.copy2(source_prefix / "manifest.json", original_manifest)

    manifest_path = destination_prefix / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    current_source = release / "native/perfect-chaos-prefix.cpp"
    if sha256(generator_source) != manifest.get("sourceSha256"):
        raise RuntimeError("Preserved generator source does not match the candidate manifest.")
    manifest.update(
        {
            "generatorRef": candidate_ref,
            "generatorCommit": candidate_commit,
            "generatorSource": "./provenance/perfect-chaos-prefix-generator.cpp",
            "originalManifest": "./provenance/original-manifest.json",
            "originalManifestSha256": sha256(original_manifest),
            "verificationSourceSha256": sha256(current_source),
            "promotionAudit": {
                "format": "connect4-chaos-prefix-cross-version-audit-v1",
                "verificationCommit": verification_commit,
                "historicalSelfReplay": True,
                "currentMainIndependentReplay": True,
                "artifactHashesChecked": True,
                "opponentClosureReplayed": True,
                "frontierEqualityChecked": True,
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    candidate_test = candidate / "tests/perfect-chaos-prefix-runtime.test.js"
    shutil.copy2(candidate_test, release / "tests/perfect-chaos-prefix-runtime.test.js")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--verification-commit", required=True)
    args = parser.parse_args()

    release = args.release.resolve()
    candidate = args.candidate.resolve()
    validate_candidate(candidate)
    install_candidate_files(
        release,
        candidate,
        candidate_ref=args.candidate_ref,
        candidate_commit=args.candidate_commit,
        verification_commit=args.verification_commit,
    )
    patch_runtime(release)
    patch_verifier(release)
    patch_package(release)
    write_manifest_test(release)

    text_paths = [
        release / "data/perfect-chaos-prefix/manifest.json",
        release / "package.json",
        release / "scripts/perfect-chaos-prefix.mjs",
        release / "src/perfect-chaos-prefix.js",
        release / "tests/perfect-chaos-prefix-manifest.test.js",
        release / "tests/perfect-chaos-prefix-runtime.test.js",
    ]
    for path in text_paths:
        text = path.read_text()
        if "\t" in text:
            raise RuntimeError(f"Tabs are forbidden in promoted text: {path}")

    print(
        json.dumps(
            {
                "format": "connect4-chaos-prefix-promotion-preparation-v1",
                "candidateRef": args.candidate_ref,
                "candidateCommit": args.candidate_commit,
                "verificationCommit": args.verification_commit,
                "boundaries": EXPECTED_BOUNDARIES,
                "final": EXPECTED_FINAL,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
