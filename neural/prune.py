"""Prunes checkpoints on the Volume without orphaning their sidecars.

A checkpoint is up to three files, and they go together: X.pt, its optimizer
moments X.pt.opt and its lineage record X.pt.lineage.json. The 13-09 prune
deleted *.pt only and left 93 GB of orphaned .opt files behind. This keeps
the newest checkpoints of the current model's lineage - ARENA_LAG + 1 by
default, so the next arena still finds its opponent - and any milestone
named explicitly, and deletes every other checkpoint with all of its files,
orphaned sidecars included, along with .partial files that no writer can
still be finishing. Nothing younger than RECENT_SECONDS is touched: a
learner may be publishing while this runs. A checkpoint whose lineage
records do not show that it is unrelated to the named model - a model name
copied from an older log is the usual cause - stops the plan unless it is a
--milestone or --force is given.

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

from .checkpoint_lineage import read_history, read_parent

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


def ancestors(model, parents):
    """Every checkpoint `model` descends from, as far as its records reach;
    parents maps a checkpoint to the parent its lineage record names (None
    for a checkpoint with no record)."""
    found, node = set(), model
    while (parent := parents.get(node)) is not None and parent not in found and parent != model:
        found.add(parent)
        node = parent
    return found


def ancestry_risk(model, current, parents, unreadable, files, current_ancestors):
    """Why deleting `model` might delete a descendant of `current`, or None
    when its recorded ancestry shows it does not descend from it: the chain
    reaches one of current's own ancestors (a side branch) or a root in
    models/. A chain broken by an earlier prune proves nothing either way."""
    if model in unreadable:
        return unreadable[model]
    node, seen = model, {model}
    while True:
        parent = parents.get(node)
        if parent is None:
            return None                     # a root: its record names no parent
        if parent == current:
            return f"descends from the current model {current}"
        if parent in current_ancestors:
            return None
        if parent in seen:
            return f"its lineage records form a cycle through {parent}"
        if parent in unreadable:
            return f"its ancestry passes through {parent}, whose lineage record is unreadable"
        if parent not in parents:
            if parent in files:
                return None                 # it ends at a root in models/
            return f"its ancestry cannot be traced past {parent}, which is no longer in models/"
        seen.add(parent)
        node = parent


def plan_prune(files, lineage, *, keep=DEFAULT_KEEP, milestones=(), now, recent=RECENT_SECONDS,
               parents=None, unreadable=None, force=False):
    """(delete, keep): sorted names of the models/ files to delete and to keep.

    files maps every file directly in models/ to (size, mtime). lineage is the
    current model's ancestry, oldest first and ending with the current model
    (checkpoint_lineage.read_history); its newest `keep` entries are kept.
    A checkpoint's files are kept or deleted together, and kept if any of
    them is younger than `recent` seconds. A .partial file is deleted once it
    is older than that. The current model and every milestone must exist: a
    misspelt name must not delete the checkpoint it meant to keep. Nor may a
    stale one delete the lineage's newer checkpoints - a model name copied
    from an older log line used to delete the live run's newest checkpoints
    and their moments. Given `parents` (as for `ancestors`) and `unreadable`
    (checkpoint: why its record could not be read), every checkpoint with a
    lineage record that would be deleted, other than the current model's own
    ancestors, must be shown by its records not to descend from the current
    model (see ancestry_risk); one that cannot stops the plan unless `force`.
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
    current, parents, unreadable = lineage[-1], parents or {}, unreadable or {}
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
    current_ancestors = ancestors(current, parents)
    at_risk = []
    for owner, members in groups.items():
        fresh = any(now - files[name][1] <= recent for name in members)
        doomed = owner not in kept and not fresh
        (delete if doomed else retain).extend(members)
        if (doomed and owner in files and (owner in parents or owner in unreadable)
                and owner not in current_ancestors):
            risk = ancestry_risk(owner, current, parents, unreadable, files, current_ancestors)
            if risk:
                at_risk.append(f"{owner}: {risk}")
    if at_risk and not force:
        raise ValueError(
            f"{len(at_risk)} checkpoint(s) to delete may descend from the current model {current}; name "
            "the lineage's newest checkpoint as current, keep them with --milestone, or pass --force:\n  "
            + "\n  ".join(sorted(at_risk)))
    return sorted(delete), sorted(retain)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m neural.prune", description=__doc__.split("\n\n")[0])
    parser.add_argument("current", help="the newest checkpoint of the lineage to keep, e.g. big612-abc.pt")
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP,
                        help=f"checkpoints of that lineage to keep, newest first (default {DEFAULT_KEEP})")
    parser.add_argument("--milestone", action="append", default=[], help="another checkpoint to keep")
    parser.add_argument("--apply", action="store_true", help="delete; without it nothing changes")
    parser.add_argument("--force", action="store_true",
                        help="delete even checkpoints that may descend from the current model")
    args = parser.parse_args(argv)

    import modal       # the Modal client environment; imported here so the planner needs none

    volume = modal.Volume.from_name(VOLUME_NAME)
    files = {PurePosixPath(entry.path).name: (entry.size, entry.mtime)
             for entry in volume.listdir("models") if int(entry.type) == 1}
    lineage = read_history(volume.read_file, args.current, args.keep)
    # Every lineage record, orphans included (they still link a chain). They
    # only feed the refusal that --force overrides, so --force reads none. One
    # record that cannot be read is reported by name, not fatal to the plan.
    parents, unreadable = {}, {}
    for name in [] if args.force else sorted(files):
        if not name.endswith(".pt.lineage.json"):
            continue
        model = name[:-len(".lineage.json")]
        try:
            parents[model] = read_parent(volume.read_file, model)
        except Exception as exc:
            unreadable[model] = f"models/{name}: {type(exc).__name__}: {str(exc)[:120]}"
    delete, keep = plan_prune(files, lineage, keep=args.keep, milestones=args.milestone, now=time.time(),
                              parents=parents, unreadable=unreadable, force=args.force)
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
