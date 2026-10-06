// Writes the optimal-policy closure of one starting role on a classic board up
// to 7x7 (`generate`) and checks small boards with known values (`verify`).
// The exact search is native/classic-exact.hpp; every policy is replayed in
// JavaScript before it is accepted (docs/PERFECT_CLASSIC_VARIANTS.md).
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "classic-exact.hpp"

namespace {

using namespace connect4::classic;

constexpr std::uint8_t ROLE_FIRST = 1;
constexpr std::uint8_t ROLE_SECOND = 2;
constexpr std::uint8_t FORMAT_VERSION = 1;
constexpr std::uint8_t RECORD_SIZE = 10;
constexpr std::size_t HEADER_SIZE = 24;
// A canonical key spans at most 56 bits (classic-exact.hpp), so bit 63 is
// free to mark the AI's turn in a closure key.
constexpr std::uint64_t AI_TURN_BIT = std::uint64_t{1} << 63;

struct PolicyRecord {
  std::uint64_t key;
  std::uint8_t moveMask;
  std::int8_t outcome;
};

struct ClosureState {
  Position position;
  bool aiTurn;
};

struct GenerationStats {
  std::uint64_t closureStates = 0;
  std::uint64_t aiStates = 0;
  std::uint64_t opponentStates = 0;
  std::uint64_t handoffStates = 0;
  std::uint64_t terminalAiWins = 0;
  std::uint64_t terminalAiLosses = 0;
  std::uint64_t terminalDraws = 0;
  std::uint64_t revisitedStates = 0;
};

std::uint64_t closureKey(const CanonicalPosition& canonical, bool aiTurn) {
  return canonical.key | (aiTurn ? AI_TURN_BIT : 0);
}

int continuationCount(const Geometry& geometry, const Position& position, int column) {
  const std::uint64_t move = moveForColumn(geometry, position.mask, column);
  if (move == 0 || hasAlignment(geometry, position.current | move)) return 0;
  const Position opponent = play(position, move);
  const std::uint64_t possible = possibleMoves(geometry, opponent.mask);
  if (possible == 0) return 0;

  std::unordered_set<std::uint64_t> replies;
  for (const int replyColumn : geometry.columnOrder) {
    const std::uint64_t reply = moveForColumn(geometry, opponent.mask, replyColumn);
    if ((possible & reply) == 0) continue;
    if (hasAlignment(geometry, opponent.current | reply)) continue;
    const Position child = play(opponent, reply);
    if (possibleMoves(geometry, child.mask) == 0) continue;
    replies.insert(canonicalize(geometry, child).key);
  }
  return static_cast<int>(replies.size());
}

int choosePolicyColumn(
    const Geometry& geometry,
    const Position& position,
    std::uint8_t optimalMask) {
  int selected = -1;
  int selectedContinuations = std::numeric_limits<int>::max();
  for (const int column : geometry.columnOrder) {
    if ((optimalMask & (1u << column)) == 0) continue;
    const int continuations = continuationCount(geometry, position, column);
    if (selected < 0 || continuations < selectedContinuations) {
      selected = column;
      selectedContinuations = continuations;
    }
  }
  if (selected < 0) throw std::runtime_error("exact solver returned no optimal policy move");
  return selected;
}

void writeByte(std::ofstream& output, std::uint8_t value) {
  output.put(static_cast<char>(value));
}

void writeUint32(std::ofstream& output, std::uint32_t value) {
  for (int index = 0; index < 4; ++index) writeByte(output, (value >> (index * 8)) & 0xff);
}

void writeUint64(std::ofstream& output, std::uint64_t value) {
  for (int index = 0; index < 8; ++index) writeByte(output, (value >> (index * 8)) & 0xff);
}

void writePolicy(
    const std::string& path,
    const Geometry& geometry,
    std::uint8_t role,
    int handoffRemaining,
    int rootValue,
    const std::vector<PolicyRecord>& records,
    const GenerationStats& stats) {
  if (records.size() > std::numeric_limits<std::uint32_t>::max()
      || stats.closureStates > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("policy is too large for format version 1");
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("could not open policy output");
  const std::array<char, 8> magic{{'C', '4', 'V', 'P', 'O', 'L', '1', '\0'}};
  output.write(magic.data(), magic.size());
  writeByte(output, FORMAT_VERSION);
  writeByte(output, static_cast<std::uint8_t>(geometry.rows));
  writeByte(output, static_cast<std::uint8_t>(geometry.columns));
  writeByte(output, static_cast<std::uint8_t>(geometry.connect));
  writeByte(output, role);
  writeByte(output, static_cast<std::uint8_t>(handoffRemaining));
  writeByte(output, RECORD_SIZE);
  writeByte(output, static_cast<std::uint8_t>(static_cast<std::int8_t>(rootValue)));
  writeUint32(output, static_cast<std::uint32_t>(records.size()));
  writeUint32(output, static_cast<std::uint32_t>(stats.closureStates));
  for (const PolicyRecord& record : records) {
    writeUint64(output, record.key);
    writeByte(output, record.moveMask);
    writeByte(output, static_cast<std::uint8_t>(record.outcome));
  }
  // The last block is written by the flush on close, so a full disk shows up
  // only there; the policy is not reported written before it is.
  output.close();
  if (output.fail()) throw std::runtime_error("could not write complete policy output");
}

struct Arguments {
  std::string command = "verify";
  int rows = 6;
  int columns = 7;
  int connect = 4;
  int role = ROLE_FIRST;
  int handoffRemaining = 24;
  int tableBits = 24;
  std::uint64_t maximumNodes = 0;
  std::uint64_t maximumStates = 100'000'000;
  std::string output;
};

int parseInt(std::string_view value, int minimum, int maximum, std::string_view label) {
  std::size_t parsed = 0;
  const int result = std::stoi(std::string(value), &parsed);
  if (parsed != value.size() || result < minimum || result > maximum) {
    throw std::range_error(std::string(label) + " is outside its supported range");
  }
  return result;
}

std::uint64_t parseUint64(std::string_view value, std::string_view label) {
  std::size_t parsed = 0;
  const std::uint64_t result = std::stoull(std::string(value), &parsed);
  if (parsed != value.size()) throw std::range_error(std::string(label) + " is invalid");
  return result;
}

Arguments parseArguments(int argc, char** argv) {
  Arguments arguments;
  int index = 1;
  if (index < argc && !std::string_view(argv[index]).starts_with("--")) {
    arguments.command = argv[index++];
  }
  while (index < argc) {
    const std::string name = argv[index++];
    if (index >= argc) throw std::range_error(name + " requires a value");
    const std::string value = argv[index++];
    if (name == "--rows") arguments.rows = parseInt(value, 1, 7, "rows");
    else if (name == "--columns") arguments.columns = parseInt(value, 1, 7, "columns");
    else if (name == "--connect") arguments.connect = parseInt(value, 1, 7, "connect");
    else if (name == "--role") arguments.role = parseInt(value, 1, 2, "role");
    else if (name == "--handoff-remaining") {
      arguments.handoffRemaining = parseInt(value, 0, 49, "handoff-remaining");
    } else if (name == "--table-bits") arguments.tableBits = parseInt(value, 8, 27, "table-bits");
    else if (name == "--maximum-nodes") arguments.maximumNodes = parseUint64(value, "maximum-nodes");
    else if (name == "--maximum-states") arguments.maximumStates = parseUint64(value, "maximum-states");
    else if (name == "--output") arguments.output = value;
    else throw std::range_error("unknown argument: " + name);
  }
  return arguments;
}

struct GeneratedPolicy {
  int rootValue;
  std::vector<PolicyRecord> records;
  GenerationStats stats;
  // The exact search's own counters, over every AI decision of the closure.
  std::uint64_t nodes;
  std::uint64_t tableHits;
  std::uint64_t tableStores;
  std::uint64_t tableCollisions;
  std::uint64_t cutoffs;
};

GeneratedPolicy generatePolicy(
    const Geometry& geometry,
    std::uint8_t role,
    int handoffRemaining,
    int tableBits,
    std::uint64_t maximumNodes,
    std::uint64_t maximumStates) {
  if (handoffRemaining < 0 || handoffRemaining > geometry.cellCount) {
    throw std::range_error("handoff-remaining must fit the board");
  }
  ExactSolver solver(geometry, tableBits, maximumNodes);
  const MoveValues root = solver.moveValues(Position{});
  const int rootValue = role == ROLE_FIRST ? root.value : -root.value;

  std::vector<ClosureState> queue;
  std::unordered_set<std::uint64_t> seen;
  GenerationStats stats;
  auto enqueue = [&](const Position& raw, bool aiTurn) {
    const CanonicalPosition canonical = canonicalize(geometry, raw);
    const std::uint64_t key = closureKey(canonical, aiTurn);
    if (!seen.insert(key).second) {
      ++stats.revisitedStates;
      return;
    }
    if (seen.size() > maximumStates) throw std::runtime_error("closure-limit");
    queue.push_back({canonical.position, aiTurn});
  };

  enqueue(Position{}, role == ROLE_FIRST);
  std::vector<PolicyRecord> records;
  for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
    const ClosureState state = queue[cursor];
    ++stats.closureStates;
    const int remaining = geometry.cellCount - state.position.moves;

    if (state.aiTurn && remaining <= handoffRemaining) {
      ++stats.handoffStates;
      continue;
    }

    const std::uint64_t possible = possibleMoves(geometry, state.position.mask);
    if (possible == 0) {
      ++stats.terminalDraws;
      continue;
    }

    if (state.aiTurn) {
      ++stats.aiStates;
      const MoveValues values = solver.moveValues(state.position);
      const int column = choosePolicyColumn(geometry, state.position, values.optimalMask);
      records.push_back({
          canonicalize(geometry, state.position).key,
          static_cast<std::uint8_t>(1u << column),
          static_cast<std::int8_t>(values.value),
      });
      const std::uint64_t move = moveForColumn(geometry, state.position.mask, column);
      if (hasAlignment(geometry, state.position.current | move)) {
        ++stats.terminalAiWins;
        continue;
      }
      const Position child = play(state.position, move);
      if (possibleMoves(geometry, child.mask) == 0) {
        ++stats.terminalDraws;
        continue;
      }
      enqueue(child, false);
      continue;
    }

    ++stats.opponentStates;
    for (const int column : geometry.columnOrder) {
      const std::uint64_t move = moveForColumn(geometry, state.position.mask, column);
      if ((possible & move) == 0) continue;
      if (hasAlignment(geometry, state.position.current | move)) {
        ++stats.terminalAiLosses;
        continue;
      }
      const Position child = play(state.position, move);
      if (possibleMoves(geometry, child.mask) == 0) {
        ++stats.terminalDraws;
        continue;
      }
      enqueue(child, true);
    }
  }

  std::sort(records.begin(), records.end(), [](const PolicyRecord& first, const PolicyRecord& second) {
    return first.key < second.key;
  });
  for (std::size_t index = 1; index < records.size(); ++index) {
    if (records[index - 1].key == records[index].key) {
      throw std::runtime_error("duplicate policy key");
    }
  }
  return {rootValue, std::move(records), stats, solver.nodes, solver.table().hits,
          solver.table().stores, solver.table().collisions, solver.cutoffs};
}

void printSummary(
    const Geometry& geometry,
    const Arguments& arguments,
    const GeneratedPolicy& generated,
    std::int64_t elapsedMs) {
  const GenerationStats& stats = generated.stats;
  std::cout << "{\"format\":\"connect4-perfect-classic-policy-summary-v1\""
            << ",\"rows\":" << geometry.rows
            << ",\"columns\":" << geometry.columns
            << ",\"connect\":" << geometry.connect
            << ",\"role\":" << arguments.role
            << ",\"handoffRemaining\":" << arguments.handoffRemaining
            << ",\"rootValue\":" << generated.rootValue
            << ",\"entryCount\":" << generated.records.size()
            << ",\"closureStates\":" << stats.closureStates
            << ",\"aiStates\":" << stats.aiStates
            << ",\"opponentStates\":" << stats.opponentStates
            << ",\"handoffStates\":" << stats.handoffStates
            << ",\"terminalAiWins\":" << stats.terminalAiWins
            << ",\"terminalAiLosses\":" << stats.terminalAiLosses
            << ",\"terminalDraws\":" << stats.terminalDraws
            << ",\"revisitedStates\":" << stats.revisitedStates
            << ",\"nodes\":" << generated.nodes
            << ",\"tableHits\":" << generated.tableHits
            << ",\"tableStores\":" << generated.tableStores
            << ",\"tableCollisions\":" << generated.tableCollisions
            << ",\"cutoffs\":" << generated.cutoffs
            << ",\"elapsedMs\":" << elapsedMs << "}\n";
}

void generate(const Arguments& arguments) {
  if (arguments.output.empty()) throw std::range_error("--output is required");
  const Geometry geometry(arguments.rows, arguments.columns, arguments.connect);
  const auto start = std::chrono::steady_clock::now();

  // Keep the policy generator and its exact solver in one deterministic process.
  // generatePolicy owns the shared transposition table for the complete closure.
  GeneratedPolicy generated = generatePolicy(
      geometry,
      static_cast<std::uint8_t>(arguments.role),
      arguments.handoffRemaining,
      arguments.tableBits,
      arguments.maximumNodes,
      arguments.maximumStates);

  writePolicy(
      arguments.output,
      geometry,
      static_cast<std::uint8_t>(arguments.role),
      arguments.handoffRemaining,
      generated.rootValue,
      generated.records,
      generated.stats);
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start).count();
  printSummary(geometry, arguments, generated, elapsed);
}

void verify() {
  struct Case { int rows; int columns; int connect; int role; int handoff; int value; };
  const std::array<Case, 6> cases{{
      {2, 2, 2, ROLE_FIRST, 0, WIN},
      {2, 2, 2, ROLE_SECOND, 0, LOSS},
      {3, 3, 3, ROLE_FIRST, 0, DRAW},
      {3, 3, 3, ROLE_SECOND, 0, DRAW},
      {4, 4, 4, ROLE_FIRST, 8, DRAW},
      {4, 6, 4, ROLE_FIRST, 24, LOSS},
  }};
  for (const Case& test : cases) {
    const Geometry geometry(test.rows, test.columns, test.connect);
    const GeneratedPolicy generated = generatePolicy(
        geometry,
        static_cast<std::uint8_t>(test.role),
        test.handoff,
        20,
        100'000'000,
        10'000'000);
    if (generated.rootValue != test.value) {
      throw std::runtime_error("policy verification root-value mismatch");
    }
    std::cout << "{\"format\":\"connect4-perfect-classic-policy-verification-v1\""
              << ",\"rows\":" << test.rows
              << ",\"columns\":" << test.columns
              << ",\"connect\":" << test.connect
              << ",\"role\":" << test.role
              << ",\"handoffRemaining\":" << test.handoff
              << ",\"rootValue\":" << generated.rootValue
              << ",\"entryCount\":" << generated.records.size()
              << ",\"closureStates\":" << generated.stats.closureStates << "}\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments = parseArguments(argc, argv);
    if (arguments.command == "verify") verify();
    else if (arguments.command == "generate") generate(arguments);
    else throw std::range_error("unknown command: " + arguments.command);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
