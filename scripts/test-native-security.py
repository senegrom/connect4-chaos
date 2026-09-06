#!/usr/bin/env python3
"""Exercise native size bounds under UBSan; optionally compare pre-fix sources."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("perfect-classic", "perfect-classic-policy", "perfect-chaos")

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
CHAOS_CHECKS = r"""
  using namespace perfect_chaos;
  for (const std::size_t count : {0u, 1u, 255u, 256u}) {
    Graph graph;
    graph.nodes.resize(2);
    graph.nodes[0].edges.resize(count, {Action{}, kPlaying, 1});
    build_predecessors(graph);
    const auto& predecessors = graph.nodes[1].predecessors;
    assert(predecessors.size() == count);
    for (std::size_t index = 0; index < count; ++index) {
      assert(predecessors[index].node == 0);
      assert(predecessors[index].edge == index);
    }
  }
  Graph oversized;
  oversized.nodes.resize(1);
  oversized.nodes[0].edges.resize(257);
  bool rejected = false;
  try { build_predecessors(oversized); }
  catch (const std::overflow_error&) { rejected = true; }
  assert(rejected);

  Graph invalid_child;
  invalid_child.nodes.resize(1);
  invalid_child.nodes[0].edges.push_back({Action{}, kPlaying, 1});
  rejected = false;
  try { build_predecessors(invalid_child); }
  catch (const std::out_of_range&) { rejected = true; }
  assert(rejected);
"""


def run(command):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            timeout=180)
    if result.returncode:
        raise RuntimeError(f"Command failed: {shlex.join(map(str, command))}\n"
                           f"{result.stdout}\n{result.stderr}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        help="Directory containing the three pre-fix C++ sources")
    args = parser.parse_args()
    compiler = shlex.split(os.environ.get("CXX", "g++"))
    flags = ["-std=c++20", "-O1", "-g", "-fsanitize=undefined",
             "-fno-sanitize-recover=all", "-D_GLIBCXX_ASSERTIONS", "-Wall", "-Wextra"]
    with tempfile.TemporaryDirectory(prefix="connect4-native-security-") as tmp:
        directory = Path(tmp)
        for name in SOURCES:
            checks = CHAOS_CHECKS if name == "perfect-chaos" else CLASSIC_CHECKS
            move_check = ("assert(solver.root(Position{}).second == WIN);"
                          if name == "perfect-classic" else
                          "assert(solver.moveValues(Position{}).value == WIN);")
            checks = checks.replace("CHECK_MOVE", move_check)
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
            if args.baseline is None:
                continue
            baseline = directory / f"{name}-before"
            run([*compiler, "-std=c++20", "-O2",
                 str(args.baseline.resolve() / f"{name}.cpp"), "-o", str(baseline)])
            if actual != run([str(baseline), "verify"]):
                raise AssertionError(f"{name}: solver output changed")
            print(f"{name}: complete verification output matches pre-fix source", flush=True)
            if name != "perfect-classic-policy":
                continue
            for rows, columns, connect, handoff in ((2, 2, 2, 0), (3, 3, 3, 0), (4, 4, 4, 8)):
                for role in (1, 2):
                    outputs = []
                    for index, executable in enumerate((baseline, binary)):
                        output = directory / f"policy-{index}.bin"
                        run([str(executable), "generate", "--rows", str(rows),
                             "--columns", str(columns), "--connect", str(connect),
                             "--role", str(role), "--handoff-remaining", str(handoff),
                             "--table-bits", "16", "--output", str(output)])
                        outputs.append(output.read_bytes())
                    if outputs[0] != outputs[1]:
                        raise AssertionError(f"Policy changed: {rows}x{columns}, role {role}")
            print("All six generated policy files match pre-fix bytes", flush=True)


if __name__ == "__main__":
    main()
