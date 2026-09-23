"""Explicit successful-checkpoint ancestry; filenames do not establish lineage."""
from __future__ import annotations

import json
from pathlib import Path

VERSION = 1


def _name(value):
    if (not isinstance(value, str) or not value.endswith(".pt")
            or value in (".pt", "..pt") or "/" in value or "\\" in value
            or "\x00" in value):
        raise ValueError("A checkpoint must be a .pt filename, not a path")
    return value


def lineage_record(model, parent, generation):
    model, parent = _name(model), _name(parent)
    if model == parent:
        raise ValueError("A checkpoint cannot be its own parent")
    if type(generation) is not int or generation < 0:
        raise ValueError("Checkpoint generation must be a nonnegative integer")
    return dict(version=VERSION, model=model, parent=parent, generation=generation)


def write_lineage(directory, model, parent, generation):
    """Publish only for a successful learner, before committing its Volume.

    One immutable sidecar per model avoids shared mutable manifests. Completed
    checkpoints retained after a failed evaluation have no successful lineage.
    """
    record = lineage_record(model, parent, generation)
    path = Path(directory) / f"{model}.lineage.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError("Refusing to replace a checkpoint with different lineage")
        return path
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def read_history(read_file, model, limit=None):
    """Return this checkpoint's ancestry, oldest first, stopping at a legacy root.

    read_file accepts a Volume-relative path and returns an iterable of bytes.
    Only a missing sidecar means legacy ancestry: malformed records, cycles and
    transport failures propagate so the driver can disable historical arenas.
    Never scan models/ or infer predecessors from generation numbers or mtimes.
    `limit` stops after that many checkpoints (the model and its newest
    ancestors): a caller that looks back a fixed distance need not read, one
    Volume round trip each, a history hundreds of generations long.
    """
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("A history limit must be a positive integer")
    current = _name(model)
    newest_first, seen = [], set()
    child_generation = None
    while True:
        if current in seen:
            raise ValueError("Cycle in checkpoint lineage")
        seen.add(current)
        newest_first.append(current)
        if limit is not None and len(newest_first) >= limit:
            break
        try:
            raw = b"".join(read_file(f"models/{current}.lineage.json"))
        except FileNotFoundError:
            break
        record = json.loads(raw)
        if not isinstance(record, dict) or type(record.get("version")) is not int or record["version"] != VERSION:
            raise ValueError("Unsupported checkpoint lineage record")
        checked = lineage_record(record.get("model"), record.get("parent"), record.get("generation"))
        if checked["model"] != current:
            raise ValueError("Checkpoint lineage names a different model")
        generation = checked["generation"]
        if child_generation is not None and generation >= child_generation:
            raise ValueError("Checkpoint lineage generations must increase along the chain")
        child_generation = generation
        current = checked["parent"]
    return list(reversed(newest_first))
