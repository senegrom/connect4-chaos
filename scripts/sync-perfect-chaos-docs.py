#!/usr/bin/env python3
"""Synchronize released Perfect Chaos documentation from the exact manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path("data/perfect-chaos-prefix/manifest.json")
README_PATH = Path("README.md")
DOCS_PATH = Path("docs/PERFECT_CHAOS.md")


def fail(message: str) -> None:
    raise RuntimeError(message)


def comma(value: int) -> str:
    return f"{value:,}"


def require_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{label} must be a non-negative integer.")
    return value


def replace_pattern(
    text: str,
    pattern: str,
    replacement: str,
    label: str,
    *,
    flags: int = 0,
) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        fail(f"Could not replace {label}; match count was {count}.")
    return updated


def load_release() -> tuple[dict[str, Any], list[int], dict[str, dict[str, Any]]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest.get("format") != "connect4-chaos-layered-prefix-manifest-v1":
        fail("Unsupported Perfect Chaos prefix manifest format.")
    boundaries = manifest.get("boundaries")
    if not isinstance(boundaries, list) or not boundaries or boundaries[0] != 8:
        fail("Perfect Chaos prefix boundaries are missing or malformed.")
    expected = [8, *range(10, boundaries[-1] + 1, 2)]
    if boundaries != expected:
        fail(f"Perfect Chaos boundaries are not contiguous: {boundaries}")

    roles = manifest.get("roles")
    if not isinstance(roles, dict):
        fail("Perfect Chaos manifest has no role records.")
    role_data: dict[str, dict[str, Any]] = {}
    for role in ("red", "yellow"):
        record = roles.get(role)
        segments = record.get("replay", {}).get("segments") if isinstance(record, dict) else None
        if not isinstance(segments, list) or len(segments) != len(boundaries):
            fail(f"{role}: replay segment count does not match the boundaries.")
        previous = 0
        for segment, target in zip(segments, boundaries, strict=True):
            if not isinstance(segment, dict):
                fail(f"{role}: replay segment is not an object.")
            if segment.get("fromPieces") != previous or segment.get("frontierPieces") != target:
                fail(f"{role}: segment boundary mismatch at {previous}->{target}.")
            for field in ("fromStates", "policyEntries", "closureStates", "frontierStates"):
                require_integer(segment.get(field), f"{role}.{previous}-{target}.{field}")
            previous = target
        rejected = record.get("rejected")
        if not isinstance(rejected, dict):
            fail(f"{role}: rejection accounting is missing.")
        for target in boundaries[:-1]:
            require_integer(rejected.get(f"at{target}"), f"{role}.rejected.at{target}")
        role_data[role] = {"segments": segments, "rejected": rejected}
    return manifest, boundaries, role_data


def segment_table(role_data: dict[str, dict[str, Any]], role: str) -> str:
    lines = [
        "| Segment | Input roots | Policy entries | Closure states | Output frontier |",
        "|---|---:|---:|---:|---:|",
    ]
    for segment in role_data[role]["segments"]:
        lines.append(
            f"| {segment['fromPieces']} → {segment['frontierPieces']} "
            f"| {comma(segment['fromStates'])} "
            f"| {comma(segment['policyEntries'])} "
            f"| {comma(segment['closureStates'])} "
            f"| {comma(segment['frontierStates'])} |"
        )
    return "\n".join(lines)


def synchronize(readme: str, docs: str) -> tuple[str, str, dict[str, Any]]:
    _manifest, boundaries, role_data = load_release()
    boundary = boundaries[-1]
    final_red = role_data["red"]["segments"][-1]
    final_yellow = role_data["yellow"]["segments"][-1]
    final_from = final_red["fromPieces"]
    if final_yellow["fromPieces"] != final_from:
        fail("The two role certificates have different final segment boundaries.")
    rejected_red = role_data["red"]["rejected"].get(f"at{final_from}", 0)
    rejected_yellow = role_data["yellow"]["rejected"].get(f"at{final_from}", 0)
    final_closure = final_red["closureStates"] + final_yellow["closureStates"]
    boundary_list = ", ".join(
        f"`{0 if index == 0 else boundaries[index - 1]}→{target}`"
        for index, target in enumerate(boundaries)
    )

    if boundary < 36:
        gap_sentence = (
            f"The remaining certified gap runs from the committed {boundary}-piece frontier "
            "to the exact ranked-retrograde endgame handoff at 36 placed pieces."
        )
        uncovered_sentence = (
            f"Beyond {boundary} pieces the runtime returns explicitly to bounded search; "
            "the complete standard 6×7 Chaos game is not yet claimed as solved."
        )
    else:
        gap_sentence = (
            "The committed prefix now reaches the exact ranked-retrograde endgame handoff "
            "at 36 placed pieces."
        )
        uncovered_sentence = (
            "A full-game Perfect claim still requires the final combined release gate and "
            "literal-threefold replay of both starting roles."
        )

    readme = replace_pattern(
        readme,
        r"^- \*\*Certified Chaos prefix\*\* — .*$",
        (
            "- **Certified Chaos prefix** — standard 6×7 Chaos Mode has an independently "
            f"replayed non-losing policy certificate for both starting roles through **{boundary} "
            "placed pieces**; Brutal lazy-loads only the matching certified layer during live play."
        ),
        "README certified-prefix highlight",
        flags=re.M,
    )
    readme = replace_pattern(
        readme,
        r"^\| Brutal \| Certified standard-board Chaos play through \d+ placed pieces,.*$",
        (
            f"| Brutal | Certified standard-board Chaos play through {boundary} placed pieces, "
            "transform-aware bounded search beyond it, and automatic use of the exact Chaos "
            "endgame frontier. |"
        ),
        "README Brutal description",
        flags=re.M,
    )

    readme_section = f"""### Layered non-losing prefix certificate

The released standard 6×7 Chaos policy is a compositional finite-safety-game certificate. At an AI state it stores one action outside the least loss attractor; at an opponent state every legal action remains in the closure. Terminal AI losses are forbidden, while terminal wins, terminal draws, proved repetition cycles and the next exact frontier are safe exits.

The committed boundaries are {boundary_list}. A later layer may prove an incoming frontier root losing, in which case that root is committed as a rejection and propagated backward until the earlier policy can no longer reach it.

| Role | Final segment | Input roots | Rejected incoming roots | Policy entries | Closure states | Output frontier |
|---|---|---:|---:|---:|---:|---:|
| Red | {final_from} → {boundary} | {comma(final_red['fromStates'])} | {comma(rejected_red)} | {comma(final_red['policyEntries'])} | {comma(final_red['closureStates'])} | {comma(final_red['frontierStates'])} |
| Yellow | {final_from} → {boundary} | {comma(final_yellow['fromStates'])} | {comma(rejected_yellow)} | {comma(final_yellow['policyEntries'])} | {comma(final_yellow['closureStates'])} | {comma(final_yellow['frontierStates'])} |

The final two role segments contain {comma(final_closure)} independently replayed canonical closure states. Every stored AI record is reachable, every opponent continuation is explored, and each recomputed sorted frontier must be byte-identical to the committed table. Artifact hashes and binary metadata are checked before runtime loading.

{gap_sentence} {uncovered_sentence}

```bash
npm run chaos:verify
npm run chaos:prefix:verify-reference
```

"""
    readme = replace_pattern(
        readme,
        r"### Layered non-losing prefix certificate\n.*?(?=## Commands\n)",
        readme_section,
        "README layered-prefix section",
        flags=re.S,
    )
    readme = replace_pattern(
        readme,
        (
            r"^(\| `npm run chaos:prefix:verify-reference` \| Independently replay and "
            r"hash-check the committed )\d+(-piece Chaos certificate\. \|)$"
        ),
        rf"\g<1>{boundary}\g<2>",
        "README prefix verification command",
        flags=re.M,
    )

    rejection_lines = []
    for role in ("red", "yellow"):
        counts = [
            f"{comma(role_data[role]['rejected'][f'at{target}'])} at {target}"
            for target in boundaries[:-1]
        ]
        rejection_lines.append(f"- {role.title()}: " + ", ".join(counts) + ".")

    detailed_section = f"""## Layered prefix safety certificate

`native/perfect-chaos-prefix.cpp` and `scripts/perfect-chaos-prefix.mjs` prove that a fixed strategy cannot lose from the empty standard board before a selected exact piece-count frontier. The safety game uses these rules:

- at an AI state, at least one selected action must remain outside the least loss attractor;
- at an opponent state, every legal action is explored;
- terminal AI losses are forbidden;
- terminal AI wins, terminal draws and the next exact frontier are safe exits;
- quotient cycles lift to finite real-board mirror orbits and therefore end under the actual threefold-repetition rule.

### Compositional boundaries

The committed linked segments are {boundary_list}. The output frontier from one segment is the exact sorted input-root set of the next. When a later segment proves an incoming root losing, the root is written to a rejection table and propagated backward until the earlier closure can no longer reach it.

The committed rejection accounting is:

{chr(10).join(rejection_lines)}

### Verified {boundary}-piece closure

The reference in `data/perfect-chaos-prefix/manifest.json` carries a SHA-256 digest for every policy, frontier and rejection table. `src/perfect-chaos-prefix.js` validates each binary header, role, boundary, record size, gravity-valid canonical state and action before lookup. The browser loads only the role and segment needed for the current position.

For the AI playing Red:

{segment_table(role_data, 'red')}

For the AI playing Yellow:

{segment_table(role_data, 'yellow')}

Across the final segments, the independent replay follows {comma(final_closure)} canonical closure states. Every policy record is reachable, every legal opponent action is explored, no AI-loss terminal is reachable, and both recomputed output frontiers are byte-for-byte identical to the committed tables.

The result is a **non-losing prefix certificate**, not by itself a full-game solution. Every adversarial line under the emitted strategy reaches an AI win, a terminal draw, a proved repetition draw, or an explicitly committed {boundary}-piece frontier state. {uncovered_sentence}

### Deterministic sharding and exact repair

Large frontier sets are divided into deterministic shards. Missing or malformed shards, state-limit exits, policy conflicts and incomplete accounting fail the round. Once later counterexamples are known, the dependency partitioner reuses byte-identical unaffected policy slices and re-solves only affected or newly introduced roots. The assembled policy is then replayed as one complete closure; incremental repair is accepted only when it is equivalent to a full exact regeneration on the verification cases.

### Verification commands

- `npm run chaos:prefix:verify` checks the native solver on deterministic small references and cross-checks the JavaScript transition model.
- `npm run chaos:prefix:verify-reference` checks every committed artifact hash and independently replays the full {boundary}-piece reference.
- `npm run chaos:prefix:generate` runs counterexample-guided generation through the configured frontier.
- `npm run chaos:prefix:reproduce` regenerates the committed reference from its rejection tables.

"""
    docs = replace_pattern(
        docs,
        r"## Layered prefix safety certificate\n.*?(?=## Correctness coverage\n)",
        detailed_section,
        "detailed layered-prefix section",
        flags=re.S,
    )

    why_section = f"""## Why the empty 6×7 board is not labelled Perfect yet

The empty standard Chaos position has a much larger reachable graph than classic Connect Four. Flip and rotation moves create large same-piece-count orbits, and rotations alternate between 6×7 and 7×6 orientations. The committed prefix reaches {boundary} placed pieces; the exact runtime endgame handoff begins at 36 pieces. {gap_sentence}

The final committed layer required {comma(rejected_red)} Red and {comma(rejected_yellow)} Yellow rejected roots at its incoming {final_from}-piece boundary before both closures were safe. This is why later frontiers must continue exact counterexample-guided refinement rather than assuming every reachable state is safe.

The UI therefore keeps **Perfect** unavailable for standard 6×7 Chaos until both starting-role closures connect to the exact endgame region and pass the complete literal-threefold replay gate. Brutal uses the released certificate through {boundary} pieces and labels later computation as bounded search.

"""
    docs = replace_pattern(
        docs,
        r"## Why the empty 6×7 board is not labelled Perfect yet\n.*?(?=## Route to a complete Perfect Chaos release\n)",
        why_section,
        "empty-board claim section",
        flags=re.S,
    )

    next_boundary = boundary + 2
    route = f"""## Route to a complete Perfect Chaos release

1. Extend the independently audited prefix from {boundary} to {next_boundary} pieces for both starting roles.
2. Commit each role's exact counterexample state and continue deterministic sharded rounds until a zero-counterexample closure candidate is produced.
3. Re-download producer and independent-evidence artifacts by exact run, commit and digest; reproduce the closure decisions byte for byte.
4. Assemble a fresh two-role reference, replay every legal adversarial continuation, and promote the new runtime layer only after exact and browser release gates pass.
5. Repeat the same process over later even-piece boundaries until the prefix reaches the exact endgame handoff at 36 pieces.
6. Independently replay both complete starting-role closures under the literal threefold rule and verify every runtime lookup.
7. Enable the Perfect option for standard 6×7 Chaos only after the final full-game claim gate succeeds.

The existing classic Perfect strategy remains unchanged and independently verified."""
    docs = replace_pattern(
        docs,
        (
            r"## Route to a complete Perfect Chaos release\n.*?"
            r"The existing classic Perfect strategy remains unchanged and independently verified\."
        ),
        route,
        "route-to-release section",
        flags=re.S,
    )

    summary = {
        "boundary": boundary,
        "boundaries": boundaries,
        "finalClosureStates": final_closure,
        "redFinalFrontier": final_red["frontierStates"],
        "yellowFinalFrontier": final_yellow["frontierStates"],
    }
    return readme, docs, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    original_readme = README_PATH.read_text()
    original_docs = DOCS_PATH.read_text()
    readme, docs, summary = synchronize(original_readme, original_docs)
    changed = []
    if readme != original_readme:
        changed.append(str(README_PATH))
    if docs != original_docs:
        changed.append(str(DOCS_PATH))

    if args.check:
        if changed:
            fail(f"Perfect Chaos release documentation is stale: {changed}")
    else:
        README_PATH.write_text(readme, newline="\n")
        DOCS_PATH.write_text(docs, newline="\n")

    print(json.dumps({**summary, "changed": changed, "check": args.check}, indent=2))


if __name__ == "__main__":
    main()
