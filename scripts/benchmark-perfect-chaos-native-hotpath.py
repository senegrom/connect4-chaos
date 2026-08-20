#!/usr/bin/env python3
"""Benchmark allocation-free Perfect Chaos native hot paths exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import NoReturn

OLD_TRANSFORM = r'''State transform(const State& s, ActionType type) {
  if (type == ActionType::Drop) throw std::runtime_error("Drop is not a transformation.");
  State out{0,0, type == ActionType::Flip ? s.rows : s.columns,
                  type == ActionType::Flip ? s.columns : s.rows, s.ai_turn};
  validate(out);
  using Piece = std::pair<int,std::uint8_t>;
  std::array<std::vector<Piece>,10> columns;
  for (int c = 0; c < s.columns; ++c) for (int r = 0; r < s.rows; ++r) {
    Mask source = bit(s,c,r); std::uint8_t owner = 0;
    if (s.mover & source) owner = 1; else if (s.opponent & source) owner = 2;
    if (!owner) continue;
    int tc = 0, tr = 0;
    if (type == ActionType::Flip) { tc = c; tr = s.rows - 1 - r; }
    else if (type == ActionType::CW) { tc = r; tr = s.columns - 1 - c; }
    else { tc = s.rows - 1 - r; tr = c; }
    columns[tc].push_back({tr,owner});
  }
  for (int c = 0; c < out.columns; ++c) {
    auto& col = columns[c]; std::sort(col.begin(),col.end());
    for (int r = 0; r < static_cast<int>(col.size()); ++r) {
      Mask target = bit(out,c,r);
      if (col[r].second == 1) out.mover |= target; else out.opponent |= target;
    }
  }
  return out;
}
'''

OPTIMIZED_TRANSFORM = r'''State transform(const State& s, ActionType type) {
  if (type == ActionType::Drop) throw std::runtime_error("Drop is not a transformation.");
  State out{0,0, type == ActionType::Flip ? s.rows : s.columns,
                  type == ActionType::Flip ? s.columns : s.rows, s.ai_turn};
  validate(out);
  auto append = [&](int target_column, int source_column, int source_row, int& target_row) {
    const Mask source = bit(s,source_column,source_row);
    if ((s.mover & source) == 0 && (s.opponent & source) == 0) return;
    const Mask target = bit(out,target_column,target_row++);
    if (s.mover & source) out.mover |= target; else out.opponent |= target;
  };
  if (type == ActionType::Flip) {
    for (int column = 0; column < s.columns; ++column) {
      int target_row = 0;
      for (int source_row = s.rows - 1; source_row >= 0; --source_row)
        append(column,column,source_row,target_row);
    }
  } else if (type == ActionType::CW) {
    for (int source_row = 0; source_row < s.rows; ++source_row) {
      int target_row = 0;
      for (int source_column = s.columns - 1; source_column >= 0; --source_column)
        append(source_row,source_column,source_row,target_row);
    }
  } else {
    for (int target_column = 0; target_column < s.rows; ++target_column) {
      const int source_row = s.rows - 1 - target_column;
      int target_row = 0;
      for (int source_column = 0; source_column < s.columns; ++source_column)
        append(target_column,source_column,source_row,target_row);
    }
  }
  return out;
}
'''

OLD_LEGAL = r'''std::vector<Action> legal(const State& s) {
  if (full(s)) return {};
  std::vector<int> order(s.columns);
  for (int c = 0; c < s.columns; ++c) order[c] = c;
  std::stable_sort(order.begin(),order.end(),[&](int a,int b){
    int da = std::abs(2*a-(s.columns-1)), db = std::abs(2*b-(s.columns-1));
    return da != db ? da < db : a < b;
  });
  std::vector<Action> actions; Mask occupied = s.mover | s.opponent;
  for (int c : order) if ((occupied & bit(s,c,s.rows-1)) == 0)
    actions.push_back({ActionType::Drop,static_cast<std::uint8_t>(c)});
  actions.push_back({ActionType::Flip,0});
  actions.push_back({ActionType::CW,0});
  actions.push_back({ActionType::CCW,0});
  return actions;
}
'''

OPTIMIZED_LEGAL = r'''struct ActionList {
  std::array<Action,10> actions{};
  std::uint8_t count = 0;
  void push_back(Action action) {
    if (count >= actions.size()) throw std::runtime_error("Too many legal actions.");
    actions[count++] = action;
  }
  const Action* begin() const { return actions.data(); }
  const Action* end() const { return actions.data() + count; }
  std::size_t size() const { return count; }
  const Action& operator[](std::size_t index) const { return actions[index]; }
};
ActionList legal(const State& s) {
  ActionList actions;
  if (full(s)) return actions;
  std::array<std::uint8_t,10> order{};
  for (int column = 0; column < s.columns; ++column)
    order[column] = static_cast<std::uint8_t>(column);
  for (int index = 1; index < s.columns; ++index) {
    const std::uint8_t value = order[index];
    const int distance = std::abs(2 * static_cast<int>(value) - (s.columns - 1));
    int target = index;
    while (target > 0) {
      const std::uint8_t previous = order[target - 1];
      const int previous_distance =
        std::abs(2 * static_cast<int>(previous) - (s.columns - 1));
      if (previous_distance < distance
          || (previous_distance == distance && previous < value)) break;
      order[target] = previous;
      --target;
    }
    order[target] = value;
  }
  const Mask occupied = s.mover | s.opponent;
  for (int index = 0; index < s.columns; ++index) {
    const int column = order[index];
    if ((occupied & bit(s,column,s.rows-1)) == 0)
      actions.push_back({ActionType::Drop,static_cast<std::uint8_t>(column)});
  }
  actions.push_back({ActionType::Flip,0});
  actions.push_back({ActionType::CW,0});
  actions.push_back({ActionType::CCW,0});
  return actions;
}
'''

VERIFY_TRANSFORM = r'''bool same_state(const State& first, const State& second) {
  return first.mover == second.mover && first.opponent == second.opponent
    && first.rows == second.rows && first.columns == second.columns
    && first.ai_turn == second.ai_turn;
}
void verify_hotpath_equivalence(){
  std::uint64_t random = 0x6a09e667f3bcc909ULL;
  auto next_random = [&]() {
    random ^= random << 7U;
    random ^= random >> 9U;
    random ^= random << 8U;
    return random;
  };
  for(const auto& [rows,columns]:std::array<std::pair<int,int>,2>{{{6,7},{7,6}}}){
    for(int sample=0;sample<20000;++sample){
      State state{0,0,static_cast<std::uint8_t>(rows),static_cast<std::uint8_t>(columns),
                  (next_random() & 1U) != 0};
      for(int column=0;column<columns;++column){
        const int height=static_cast<int>(next_random()%static_cast<std::uint64_t>(rows+1));
        for(int row=0;row<height;++row){
          const Mask target=bit(state,column,row);
          if(next_random()&1U)state.mover|=target;else state.opponent|=target;
        }
      }
      for(ActionType type:std::array<ActionType,3>{ActionType::Flip,ActionType::CW,ActionType::CCW}){
        const State expected=transform_reference(state,type);
        const State actual=transform(state,type);
        if(!same_state(expected,actual))
          throw std::runtime_error("Allocation-free transform differs from the reference transform.");
      }
    }
  }
}
'''

VERIFY_FULL = VERIFY_TRANSFORM.replace(
    "    }\n  }\n}\n",
    r'''      const auto expected_actions=legal_reference(state);
      const auto actual_actions=legal(state);
      if(expected_actions.size()!=actual_actions.size())
        throw std::runtime_error("Allocation-free legal action count differs from the reference.");
      for(std::size_t index=0;index<expected_actions.size();++index){
        if(expected_actions[index].type!=actual_actions[index].type
            ||expected_actions[index].column!=actual_actions[index].column)
          throw std::runtime_error("Allocation-free legal action order differs from the reference.");
      }
    }
  }
}
''',
)


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        fail(f"{label}: expected one exact anchor, found {count}")
    return source.replace(old, new, 1)


def make_variant(source: str, *, optimize_legal: bool) -> str:
    reference_transform = OLD_TRANSFORM.replace(
        "State transform(", "State transform_reference(", 1
    )
    source = replace_once(
        source,
        OLD_TRANSFORM,
        reference_transform + OPTIMIZED_TRANSFORM,
        "transform",
    )
    verification = VERIFY_TRANSFORM
    if optimize_legal:
        reference_legal = OLD_LEGAL.replace(
            "std::vector<Action> legal(", "std::vector<Action> legal_reference(", 1
        )
        source = replace_once(
            source,
            OLD_LEGAL,
            reference_legal + OPTIMIZED_LEGAL,
            "legal actions",
        )
        verification = VERIFY_FULL
    source = replace_once(
        source,
        "void verify_mirror(){\n",
        verification + "void verify_mirror(){\n",
        "verification helper",
    )
    source = replace_once(
        source,
        "void verify(){\n  verify_mirror();\n",
        "void verify(){\n  verify_hotpath_equivalence();\n  verify_mirror();\n",
        "verification invocation",
    )
    return source


def compile_binary(source: Path, output: Path) -> None:
    subprocess.run(
        [
            "g++",
            "-std=c++20",
            "-O3",
            "-DNDEBUG",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            str(source),
            "-o",
            str(output),
        ],
        check=True,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_generate(binary: Path, directory: Path, role: str) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=False)
    policy = directory / "0-8.policy.bin"
    frontier = directory / "0-8.frontier.bin"
    started = time.perf_counter()
    completed = subprocess.run(
        [
            str(binary),
            "generate",
            "--role",
            role,
            "--frontier-pieces",
            "8",
            "--maximum-states",
            "2000000",
            "--policy",
            str(policy),
            "--frontier",
            str(frontier),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.perf_counter() - started
    summary = json.loads(completed.stdout)
    return {
        "seconds": elapsed,
        "summary": summary,
        "policySha256": sha256(policy),
        "frontierSha256": sha256(frontier),
        "policyBytes": policy.stat().st_size,
        "frontierBytes": frontier.stat().st_size,
    }


def exact_identity(result: dict[str, object]) -> dict[str, object]:
    return {
        key: result[key]
        for key in (
            "summary",
            "policySha256",
            "frontierSha256",
            "policyBytes",
            "frontierBytes",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    source = arguments.source.read_text()
    arguments.output.mkdir(parents=True, exist_ok=False)
    build = arguments.output / "build"
    build.mkdir()
    baseline_source = build / "baseline.cpp"
    transform_source = build / "transform.cpp"
    full_source = build / "full.cpp"
    baseline_source.write_text(source)
    transform_source.write_text(make_variant(source, optimize_legal=False))
    full_source.write_text(make_variant(source, optimize_legal=True))

    binaries = {
        "baseline": build / "baseline",
        "transformOnly": build / "transform-only",
        "transformAndLegal": build / "transform-and-legal",
    }
    compile_binary(baseline_source, binaries["baseline"])
    compile_binary(transform_source, binaries["transformOnly"])
    compile_binary(full_source, binaries["transformAndLegal"])

    for name, binary in binaries.items():
        subprocess.run([str(binary), "verify"], check=True)
        print(f"verified {name}", flush=True)

    exact: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix="perfect-chaos-hotpath-exact-") as temporary:
        root = Path(temporary)
        for role in ("red", "yellow"):
            role_results: dict[str, object] = {}
            for name, binary in binaries.items():
                role_results[name] = run_generate(binary, root / role / name, role)
            reference = exact_identity(role_results["baseline"])
            for name in ("transformOnly", "transformAndLegal"):
                if exact_identity(role_results[name]) != reference:
                    fail(f"{role} exact artifacts changed in {name}")
            exact[role] = role_results

    samples: dict[str, list[float]] = {name: [] for name in binaries}
    benchmark_identity: dict[str, object] | None = None
    orders = [
        ("baseline", "transformOnly", "transformAndLegal"),
        ("transformAndLegal", "baseline", "transformOnly"),
        ("transformOnly", "transformAndLegal", "baseline"),
    ]
    with tempfile.TemporaryDirectory(prefix="perfect-chaos-hotpath-benchmark-") as temporary:
        root = Path(temporary)
        for round_index, order in enumerate(orders):
            for name in order:
                result = run_generate(
                    binaries[name], root / f"round-{round_index}-{name}", "red"
                )
                identity = exact_identity(result)
                if benchmark_identity is None:
                    benchmark_identity = identity
                elif identity != benchmark_identity:
                    fail(f"benchmark artifact identity changed in {name}")
                samples[name].append(float(result["seconds"]))
                print(
                    f"benchmark round={round_index + 1} variant={name} "
                    f"seconds={result['seconds']:.6f}",
                    flush=True,
                )

    medians = {name: statistics.median(values) for name, values in samples.items()}
    baseline = medians["baseline"]
    report = {
        "format": "connect4-chaos-native-hotpath-benchmark-v1",
        "sourceSha256": hashlib.sha256(source.encode()).hexdigest(),
        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "workload": {
            "command": "generate",
            "role": "red",
            "fromPieces": 0,
            "frontierPieces": 8,
            "expectedGraphStates": 1292938,
            "rounds": len(orders),
            "rotatedOrder": True,
        },
        "equivalence": {
            "randomGravityStatesPerGeometry": 20000,
            "geometries": ["6x7", "7x6"],
            "exactRoles": ["red", "yellow"],
            "artifactsByteIdentical": True,
            "exact": exact,
        },
        "benchmark": {
            name: {
                "samplesSeconds": values,
                "medianSeconds": medians[name],
                "speedupVsBaseline": baseline / medians[name],
            }
            for name, values in samples.items()
        },
    }
    (arguments.output / "benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", newline="\n"
    )
    markdown = [
        "# Native Perfect Chaos hot-path benchmark",
        "",
        "Every optimized binary passed randomized transform/action equivalence and emitted",
        "byte-identical Red and Yellow 0→8 exact policy/frontier artifacts.",
        "",
        "| Variant | Median seconds | Speedup |",
        "|---|---:|---:|",
    ]
    for name in ("baseline", "transformOnly", "transformAndLegal"):
        markdown.append(
            f"| {name} | {medians[name]:.6f} | {baseline / medians[name]:.4f}× |"
        )
    markdown.extend(["", "Raw samples and artifact hashes are in `benchmark.json`.", ""])
    (arguments.output / "README.md").write_text("\n".join(markdown), newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
