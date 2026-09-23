"""Stage the learner's newest eligible replay window from compressed archives."""
from __future__ import annotations

import gzip
import os
from pathlib import Path
import shutil

import torch

from .distill import filtered_chunks


def validate_window(window):
    if type(window) is not int or window < 0:
        raise ValueError("replay_window must be a nonnegative integer")


def stage_replay(source, destination, window, holdout_shapes=()):
    """Stage whole archives, but count only rows the training loader can use.

    The destination must be private to this invocation. Callers own its cleanup.
    Original mtimes and lexical ties match load_shards(), which applies the same
    filter and cap when consuming the staged files (it samples the rows of the
    last shard; only their count matters here). Excluded shards never stop the
    scan before older eligible data can fill the window.
    """
    validate_window(window)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    stats = dict(positions=0, shards=0, skipped=0, excluded=0, errors=[])
    archives = sorted(Path(source).glob("*.pt.gz"),
                      key=lambda path: (-path.stat().st_mtime, path.name))
    for path in archives:
        if stats["positions"] >= window:
            break
        out = destination / path.name[:-3]
        try:
            with gzip.open(path, "rb") as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            mtime = path.stat().st_mtime
            os.utime(out, (mtime, mtime))
            payload = torch.load(out, map_location="cpu", weights_only=True, mmap=True)
            try:
                if payload.get("source") != "selfplay":
                    raise ValueError("Replay archive does not contain a self-play shard")
                eligible = sum(len(chunk["planes"]) for chunk in filtered_chunks(
                    payload, holdout_shapes, limit=window - stats["positions"]))
            finally:
                del payload  # release the mmap before removing an excluded shard
            if not eligible:
                stats["excluded"] += 1
                out.unlink()
                continue
            stats["positions"] += eligible
            stats["shards"] += 1
        except Exception as exc:  # a bad archive must not discard the entire generation
            stats["skipped"] += 1
            out.unlink(missing_ok=True)
            if len(stats["errors"]) < 3:
                stats["errors"].append(f"{path.name}: {type(exc).__name__}: {str(exc)[:120]}")
    return stats
