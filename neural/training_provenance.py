"""Conservative validation claims across warm starts and model averaging.

The old trainer stamped every child with the current run's split/holdouts.
Consequently legacy metadata alone cannot certify an unbroken clean lineage.
Legacy weights remain usable, but their descendants must say "unknown".
"""
from __future__ import annotations

from .data_split import SPLIT_VERSION
from .training_config import parse_shape_spec

PROVENANCE_FORMAT = 1


def canonical_holdouts(spec: str) -> frozenset:
    if not isinstance(spec, str):
        raise ValueError("Holdout configurations must be a string")
    shapes = set()
    for tag in spec.split(","):
        if not tag.strip():
            continue
        parsed = parse_shape_spec(tag)
        if parsed is None:
            raise ValueError("A training holdout must name specific configurations")
        for rows, cols, connect, chaos in parsed:
            if chaos:
                rows, cols = min(rows, cols), max(rows, cols)
            shapes.add((rows, cols, connect, chaos))
    return frozenset(shapes)


def holdout_spec(shapes) -> str:
    return ",".join(f"{r}x{c}c{k}{'chaos' if mode else 'classic'}"
                    for r, c, k, mode in sorted(shapes))


def certified_partition(payload):
    """Return historical exclusions only for this lineage-aware format."""
    if not isinstance(payload, dict):
        return None
    proof = payload.get("training_provenance")
    if (not isinstance(proof, dict) or proof.get("format") != PROVENANCE_FORMAT
            or proof.get("status") != "clean"
            or payload.get("data_split_version") != SPLIT_VERSION):
        return None
    try:
        return canonical_holdouts(payload["holdout_configs"])
    except (KeyError, ValueError, TypeError):
        return None


def training_provenance(parent, requested: str, *, source="warm-start") -> dict:
    """Separate this run's exclusions from exclusions through the full lineage.

    A new holdout cannot erase exposure in an ancestor. A dropped holdout
    cannot be reclaimed by a later generation. Unknown ancestry stays unknown.
    """
    current = canonical_holdouts(requested)
    previous = current if parent is None else certified_partition(parent)
    clean = previous is not None
    claimed = current & previous if clean else frozenset()
    return {
        "data_split_version": SPLIT_VERSION if clean else "",
        "holdout_configs": holdout_spec(claimed),
        "training_provenance": {
            "format": PROVENANCE_FORMAT,
            "status": "clean" if clean else "unknown",
            "source": "scratch" if parent is None else source,
            "run_split_version": SPLIT_VERSION,
            "run_holdout_configs": holdout_spec(current),
        },
    }


def soup_provenance(payloads, calibration_holdouts: str) -> dict:
    """Averaging or recalibration cannot repair any source's unknown history."""
    current = canonical_holdouts(calibration_holdouts)
    partitions = [certified_partition(payload) for payload in payloads]
    clean = bool(partitions) and all(partition is not None for partition in partitions)
    common = current.intersection(*partitions) if clean else frozenset()
    result = training_provenance(None, holdout_spec(common))
    result["training_provenance"].update(source="soup", run_holdout_configs=holdout_spec(current))
    if not clean:
        result["data_split_version"] = ""
        result["training_provenance"]["status"] = "unknown"
    return result
