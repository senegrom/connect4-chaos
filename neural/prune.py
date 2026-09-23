"""Prunes checkpoints on the Volume without orphaning their sidecars.

A checkpoint is up to three files, and they go together: X.pt, its optimizer
moments X.pt.opt and its lineage record X.pt.lineage.json. The 13-09 prune
deleted *.pt only and left 93 GB of orphaned .opt files behind. This keeps
the newest checkpoints of the current model's lineage - ARENA_LAG + 1 by
default, so the next arena still finds its opponent - and any milestone
named explicitly, and deletes every other checkpoint with all of its files,
orphaned sidecars included, along with .partial files that no writer can
still be finishing. Nothing younger than RECENT_SECONDS is touched: a
learner may be publishing while this runs.

plan_prune() decides from names, sizes and times alone, so the decision is
tested without a Volume. main() is the thin Modal wrapper, a dry run unless
--apply is given; run it from the Modal environment (it needs no torch):

  python -m neural.prune <current model> [--keep N] [--milestone NAME ...] [--apply]
"""

from __future__ import annotations

import argparse
from pathlib import PurePosixPath
import sys
import time

from .checkpoint_lineage import read_history

VOLUME_NAME = "connect4-tables"
DEFAULT_KEEP = 6                # ARENA_LAG + 1 at the driver's default lag of 5
RECENT_SECONDS = 6 * 60 * 60    # longer than any Modal function that writes models/ may run
SIDECARS = (".opt", ".lineage.json")


def checkpoint_of(name):
    """The checkpoint a models/ file belongs to, or None when it is not one
    of a checkpoint's files (such files are never deleted)."""
    if name.endswith(".pt"):
        return name
    for suffix in SIDECARS:
        if name.endswith(".pt" + suffix):
            return name[:-len(suffix)]
    return None


def plan_prune(files, lineage, *, keep=DEFAULT_KEEP, milestones=(), now, recent=RECENT_SECONDS):
    """(delete, keep): sorted names of the models/ files to delete and to keep.

    files maps every file directly in models/ to (size, mtime). lineage is the
    current model's ancestry, oldest first and ending with the current model
    (checkpoint_lineage.read_history); its newest `keep` entries are kept.
    A checkpoint's files are kept or deleted together, and kept if any of
    them is younger than `recent` seconds. A .partial file is deleted once it
    is older than that. The current model and every milestone must exist: a
    misspelt name must not delete the checkpoint it meant to keep.
    """
    if type(keep) is not int or keep < 1:
        raise ValueError("keep must be a positive integer")
    if not lineage:
        raise ValueError("the lineage must end with the current model")
    if lineage[-1] not in files:
        raise ValueError(f"current model {lineage[-1]} is not in models/")
    absent = sorted(set(milestones) - set(files))
    if absent:
        raise ValueError(f"milestones not in models/: {', '.join(absent)}")
    kept = set(lineage[-keep:]) | set(milestones)
    groups, delete, retain = {}, [], []
    for name, (_size, mtime) in files.items():
        if name.endswith(".partial"):
            (delete if now - mtime > recent else retain).append(name)
            continue
        owner = checkpoint_of(name)
        if owner is None:
            retain.append(name)
        else:
            groups.setdefault(owner, []).append(name)
    for owner, members in groups.items():
        fresh = any(now - files[name][1] <= recent for name in members)
        (retain if owner in kept or fresh else delete).extend(members)
    return sorted(delete), sorted(retain)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m neural.prune", description=__doc__.split("\n\n")[0])
    parser.add_argument("current", help="the newest checkpoint of the lineage to keep, e.g. big612-abc.pt")
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP,
                        help=f"checkpoints of that lineage to keep, newest first (default {DEFAULT_KEEP})")
    parser.add_argument("--milestone", action="append", default=[], help="another checkpoint to keep")
    parser.add_argument("--apply", action="store_true", help="delete; without it nothing changes")
    args = parser.parse_args(argv)

    import modal       # the Modal client environment; imported here so the planner needs none

    volume = modal.Volume.from_name(VOLUME_NAME)
    files = {PurePosixPath(entry.path).name: (entry.size, entry.mtime)
             for entry in volume.listdir("models") if int(entry.type) == 1}
    lineage = read_history(volume.read_file, args.current, args.keep)
    delete, keep = plan_prune(files, lineage, keep=args.keep, milestones=args.milestone, now=time.time())
    size = lambda names: sum(files[name][0] for name in names) / 1e9
    print(f"models/: {len(files)} files, {size(files):.1f} GB; lineage kept: {', '.join(lineage[-args.keep:])}"
          + (f"; milestones: {', '.join(args.milestone)}" if args.milestone else ""))
    print(f"keep {len(keep)} files ({size(keep):.1f} GB), delete {len(delete)} ({size(delete):.1f} GB)")
    for name in delete:
        print(f"  delete models/{name}")
    if not args.apply:
        print("dry run; pass --apply to delete")
        return 0
    failed = 0
    for name in delete:
        try:
            volume.remove_file(f"models/{name}")
        except Exception as exc:                  # keep going; report at the end
            failed += 1
            print(f"  could not remove models/{name}: {type(exc).__name__}: {str(exc)[:120]}")
    print(f"removed {len(delete) - failed} files, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
