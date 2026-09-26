#!/usr/bin/env python3
"""Exercise native size bounds under UBSan."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("perfect-classic", "perfect-classic-policy")

CLASSIC_CHECKS = r"""
  for (const int invalid : {-1, 0, 8, 63, 64,
                            std::numeric_limits<int>::min(),
                            std::numeric_limits<int>::max()}) {
    for (const bool invalid_rows : {false, true}) {
      bool rejected = false;
      try {
        const Geometry geometry(invalid_rows ? invalid : 7,
                                invalid_rows ? 7 : invalid, 4);
      } catch (const std::range_error&) { rejected = true; }
      assert(rejected);
    }
  }
  for (int rows = 1; rows <= 7; ++rows) {
    for (int columns = 1; columns <= 7; ++columns) {
      const Geometry geometry(rows, columns, 1);
      assert(geometry.stride == rows + 1);
      assert(geometry.cellCount == rows * columns);
      assert(geometry.columnOrder.size() == static_cast<std::size_t>(columns));
      assert(__builtin_popcountll(geometry.boardMask) == rows * columns);
      ExactSolver solver(geometry, 8, 1000);
      CHECK_MOVE
    }
  }
"""


def run(command):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            timeout=180)
    if result.returncode:
        raise RuntimeError(f"Command failed: {shlex.join(map(str, command))}\n"
                           f"{result.stdout}\n{result.stderr}")
    return result.stdout


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    compiler = shlex.split(os.environ.get("CXX", "g++"))
    flags = ["-std=c++20", "-O1", "-g", "-fsanitize=undefined",
             "-fno-sanitize-recover=all", "-D_GLIBCXX_ASSERTIONS", "-Wall", "-Wextra"]
    with tempfile.TemporaryDirectory(prefix="connect4-native-security-") as tmp:
        directory = Path(tmp)
        for name in SOURCES:
            move_check = ("assert(solver.root(Position{}).second == WIN);"
                          if name == "perfect-classic" else
                          "assert(solver.moveValues(Position{}).value == WIN);")
            checks = CLASSIC_CHECKS.replace("CHECK_MOVE", move_check)
            source = ROOT / "native" / f"{name}.cpp"
            wrapper = directory / f"{name}-test.cpp"
            wrapper.write_text(
                '#include <cassert>\n#define main native_entry\n'
                f'#include {json.dumps(source.as_posix())}\n#undef main\n'
                'int main(int argc, char** argv) {\n' + checks +
                '\n  return native_entry(argc, argv);\n}\n', encoding="utf-8")
            binary = directory / name
            run([*compiler, *flags, str(wrapper), "-o", str(binary)])
            actual = run([str(binary), "verify"])
            fixture_count = len(actual.splitlines())
            print(f"{name}: size-bound checks and {fixture_count} solver fixtures passed",
                  flush=True)


if __name__ == "__main__":
    main()
