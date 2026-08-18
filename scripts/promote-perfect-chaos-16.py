#!/usr/bin/env python3
"""Run the 16-piece promotion implementation and expose every new file to scope checks."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


IMPLEMENTATION = Path(__file__).with_name("promote-perfect-chaos-16-impl.py")
SPEC = importlib.util.spec_from_file_location("promote_perfect_chaos_16_impl", IMPLEMENTATION)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load promotion implementation: {IMPLEMENTATION}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def argument_value(name: str) -> str:
    try:
        index = sys.argv.index(name)
        return sys.argv[index + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"Missing required wrapper argument: {name}") from error


def main() -> None:
    MODULE.main()
    release = Path(argument_value("--release")).resolve()
    # Git normally collapses a wholly new directory in short status output.  Mark
    # the provenance and 18-piece seed files intent-to-add so the subsequent
    # fail-closed scope audit sees and validates every individual path.  The final
    # ordinary git add replaces these empty index entries with the proved bytes.
    subprocess.run(
        [
            "git",
            "-C",
            str(release),
            "add",
            "-N",
            "data/perfect-chaos-prefix/provenance",
            "data/perfect-chaos-prefix-seeds-18",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
