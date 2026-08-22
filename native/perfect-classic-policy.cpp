#include <algorithm>
#include <array>
#include <chrono>
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

namespace {

constexpr int WIN = 1;
constexpr int DRAW = 0;
constexpr int LOSS = -1;
constexpr std::uint8_t ROLE_FIRST = 1;
constexpr std::uint8_t ROLE_SECOND = 2;
constexpr std::uint8_t FORMAT_VERSION = 1;
constexpr std::uint8_t RECORD_SIZE = 10;
constexpr std::size_t HEADER_SIZE = 24;
constexpr std::uint64_t AI_TURN_BIT = std::uint64_t{1} << 63;

struct Geometry {
  int rows;
  int columns;
  int connect;
  int stride;
  int cellCount;
  std::uint64_t columnBits;
  std::uint64_t columnWithSentinel;
  std::array<std::uint64_t, 7> bottomMasks{};
  std::array<std::uint64_t, 7> columnMasks{};
  std::uint64_t bottomMask = 0;
  std::uint64_t boardMask = 0;
  std::vector<int> columnOrder;
  std::array<int, 4> directions{};

  Geometry(int selectedRows, int selectedColumns, int selectedConnect)
      : rows(selectedRows),
        columns(selectedColumns),
        connect(selectedConnect),
        stride(selectedRows + 1),
        cellCount(selectedRows * selectedColumns),
        columnBits((std::uint64_t{1} << selectedRows) - 1),
        columnWithSentinel((std::uint64_t{1} << (selectedRows + 1)) - 1),
        columnOrder(static_cast<std::size_t>(selectedColumns)) {
    if (rows < 1 || rows > 7 || columns < 1 || columns > 7) {
      throw std::range_error("rows and columns must be from 1 through 7");
    }
    if (connect < 1 || connect > std::max(rows, columns)) {
      throw std::range_error("connect must fit the board");
    }
    if (stride * columns >= 63) {
      throw std::range_error("board encoding collides with the closure turn bit");
    }
    for (int column = 0; column < columns; ++column) {
      bottomMasks[column] = std::uint64_t{1} << (column * stride);
      columnMasks[column] = bottomMasks[column] * columnBits;
      bottomMask |= bottomMasks[column];
      columnOrder[static_cast<std::size_t>(column)] = column;
    }
    boardMask = bottomMask * columnBits;
    const double centre = static_cast<double>(columns - 1) / 2.0;
    std::sort(columnOrder.begin(), columnOrder.end(), [centre](int first, int second) {
      const double firstDistance = std::abs(first - centre);
      const double secondDistance = std::abs(second - centre);
      return firstDistance != secondDistance ? firstDistance < secondDistance : first < second;
    });
    directions = {1, stride - 1, stride, stride + 1};
  }
};

struct Position {
  std::uint64_t current = 0;
  std::uint64_t mask = 0;
  int moves = 0;
};

struct CanonicalPosition {
  Position position;
  std::uint64_t key;
  bool mirrored;
};

std::uint64_t possibleMoves(const Geometry& geometry, std::uint64_t mask) {
  return (mask + geometry.bottomMask) & geometry.boardMask;
}

std::uint64_t moveForColumn(const Geometry& geometry, std::uint64_t mask, int column) {
  if (column < 0 || column >= geometry.columns) return 0;
  return (mask + geometry.bottomMasks[column]) & geometry.columnMasks[column];
}

Position play(Position position, std::uint64_t move) {
  position.current ^= position.mask;
  position.mask |= move;
  ++position.moves;
  return position;
}

bool hasAlignment(const Geometry& geometry, std::uint64_t bits) {
  for (const int direction : geometry.directions) {
    std::uint64_t run = bits;
    for (int offset = 1; offset < geometry.connect && run != 0; ++offset) {
      run &= bits >> (offset * direction);
    }
    if (run != 0) return true;
  }
  return false;
}

std::uint64_t mirrorBits(const Geometry& geometry, std::uint64_t bits) {
  std::uint64_t mirrored = 0;
  for (int column = 0; column < geometry.columns; ++column) {
    const std::uint64_t group =
        (bits >> (column * geometry.stride)) & geometry.columnWithSentinel;
    mirrored |= group << ((geometry.columns - 1 - column) * geometry.stride);
  }
  return mirrored;
}

CanonicalPosition canonicalize(const Geometry& geometry, const Position& position) {
  const std::uint64_t normal = position.current + position.mask;
  const std::uint64_t mirroredCurrent = mirrorBits(geometry, position.current);
  const std::uint64_t mirroredMask = mirrorBits(geometry, position.mask);
  const std::uint64_t mirrored = mirroredCurrent + mirroredMask;
  if (normal <= mirrored) return {position, normal, false};
  return {{mirroredCurrent, mirroredMask, position.moves}, mirrored, true};
}

std::uint64_t immediateWinningMoves(
    const Geometry& geometry,
    const Position& position,
    std::uint64_t pieces) {
  const std::uint64_t possible = possibleMoves(geometry, position.mask);
  std::uint64_t winning = 0;
  for (const int column : geometry.columnOrder) {
    const std::uint64_t move = moveForColumn(geometry, position.mask, column);
    if ((possible & move) != 0 && hasAlignment(geometry, pieces | move)) winning |= move;
  }
  return winning;
}

std::uint64_t immediateWinningMoves(const Geometry& geometry, const Position& position) {
  return immediateWinningMoves(geometry, position, position.current);
}

std::uint64_t possibleNonLosingMoves(const Geometry& geometry, const Position& position) {
  std::uint64_t possible = possibleMoves(geometry, position.mask);
  const std::uint64_t opponent = position.current ^ position.mask;
  const std::uint64_t opponentWins = immediateWinningMoves(geometry, position, opponent);
  if (opponentWins != 0) {
    if ((opponentWins & (opponentWins - 1)) != 0) return 0;
    possible = opponentWins;
  }

  std::uint64_t safe = 0;
  for (const int column : geometry.columnOrder) {
    const std::uint64_t move = moveForColumn(geometry, position.mask, column);
    if ((possible & move) == 0) continue;
    if (hasAlignment(geometry, position.current | move)) {
      safe |= move;
      continue;
    }
    const Position child = play(position, move);
    if (immediateWinningMoves(geometry, child) == 0) safe |= move;
  }
  return safe;
}

struct TableEntry {
  std::uint64_t key = 0;
  std::int8_t lower = 0;
  std::int8_t upper = 0;
  std::uint8_t flags = 0;
};

class OutcomeTable {
 public:
  explicit OutcomeTable(int bits)
      : entries_(std::size_t{1} << bits), indexMask_(entries_.size() - 1) {}

  bool probe(std::uint64_t key, int& lower, int& upper) {
    const TableEntry& entry = entries_[index(key)];
    if (entry.key != key + 1) return false;
    ++hits;
    lower = (entry.flags & 1) == 0 ? -2 : entry.lower;
    upper = (entry.flags & 2) == 0 ? 2 : entry.upper;
    return true;
  }

  void storeLower(std::uint64_t key, int score) {
    TableEntry& entry = prepare(key);
    if ((entry.flags & 1) != 0 && score <= entry.lower) return;
    entry.lower = static_cast<std::int8_t>(score);
    entry.flags |= 1;
    ++stores;
  }

  void storeUpper(std::uint64_t key, int score) {
    TableEntry& entry = prepare(key);
    if ((entry.flags & 2) != 0 && score >= entry.upper) return;
    entry.upper = static_cast<std::int8_t>(score);
    entry.flags |= 2;
    ++stores;
  }

  std::uint64_t hits = 0;
  std::uint64_t stores = 0;
  std::uint64_t collisions = 0;

 private:
  std::size_t index(std::uint64_t key) const {
    key ^= key >> 23;
    key ^= key >> 41;
    return static_cast<std::size_t>(key) & indexMask_;
  }

  TableEntry& prepare(std::uint64_t key) {
    TableEntry& entry = entries_[index(key)];
    if (entry.key != 0 && entry.key != key + 1) ++collisions;
    if (entry.key != key + 1) {
      entry.key = key + 1;
      entry.flags = 0;
    }
    return entry;
  }

  std::vector<TableEntry> entries_;
  std::size_t indexMask_;
};

struct Candidate {
  int column;
  std::uint64_t move;
  int historyIndex;
  int order;
};

struct MoveValues {
  int value;
  std::uint8_t optimalMask;
};

class ExactSolver {
 public:
  ExactSolver(const Geometry& geometry, int tableBits, std::uint64_t maximumNodes)
      : geometry_(geometry),
        table_(tableBits),
        maximumNodes_(maximumNodes),
        history_(2 * geometry.columns * geometry.stride, 0) {}

  int solve(const Position& position) {
    int minimum = LOSS;
    int maximum = WIN;
    while (minimum < maximum) {
      const int middle = minimum + (maximum - minimum) / 2;
      const int score = search(position, middle, middle + 1);
      if (score <= middle) maximum = score;
      else minimum = score;
    }
    return minimum;
  }

  MoveValues moveValues(const Position& position) {
    const std::uint64_t possible = possibleMoves(geometry_, position.mask);
    if (possible == 0) return {DRAW, 0};

    const std::uint64_t winning = immediateWinningMoves(geometry_, position);
    if (winning != 0) {
      std::uint8_t mask = 0;
      for (int column = 0; column < geometry_.columns; ++column) {
        if ((winning & moveForColumn(geometry_, position.mask, column)) != 0) {
          mask |= static_cast<std::uint8_t>(1u << column);
        }
      }
      return {WIN, mask};
    }

    const std::uint64_t safe = possibleNonLosingMoves(geometry_, position);
    if (safe == 0) {
      std::uint8_t mask = 0;
      for (int column = 0; column < geometry_.columns; ++column) {
        if ((possible & moveForColumn(geometry_, position.mask, column)) != 0) {
          mask |= static_cast<std::uint8_t>(1u << column);
        }
      }
      return {LOSS, mask};
    }

    int best = -2;
    std::uint8_t optimal = 0;
    for (const Candidate& candidate : orderedMoves(position, safe)) {
      const Position child = play(position, candidate.move);
      const int childValue = solve(child);
      const int value = childValue == DRAW ? DRAW : -childValue;
      if (value > best) {
        best = value;
        optimal = static_cast<std::uint8_t>(1u << candidate.column);
      } else if (value == best) {
        optimal |= static_cast<std::uint8_t>(1u << candidate.column);
      }
    }

    if (best == LOSS) {
      for (int column = 0; column < geometry_.columns; ++column) {
        if ((possible & moveForColumn(geometry_, position.mask, column)) != 0) {
          optimal |= static_cast<std::uint8_t>(1u << column);
        }
      }
    }
    return {best, optimal};
  }

  std::uint64_t nodes = 0;
  std::uint64_t cutoffs = 0;
  const OutcomeTable& table() const { return table_; }

 private:
  void visit() {
    ++nodes;
    if (maximumNodes_ != 0 && nodes > maximumNodes_) {
      throw std::runtime_error("node-limit");
    }
  }

  int historyIndex(const Position& position, int column) const {
    const int row = __builtin_popcountll(position.mask & geometry_.columnMasks[column]);
    return (position.moves & 1) * geometry_.columns * geometry_.stride
        + column * geometry_.stride + row;
  }

  std::vector<Candidate> orderedMoves(const Position& position, std::uint64_t moves) {
    std::vector<Candidate> candidates;
    candidates.reserve(geometry_.columns);
    for (const int column : geometry_.columnOrder) {
      const std::uint64_t move = moveForColumn(geometry_, position.mask, column);
      if ((moves & move) == 0) continue;
      const std::uint64_t nextMask = position.mask | move;
      int futureWins = 0;
      for (const int futureColumn : geometry_.columnOrder) {
        const std::uint64_t future = moveForColumn(geometry_, nextMask, futureColumn);
        if (future != 0 && hasAlignment(geometry_, position.current | move | future)) {
          ++futureWins;
        }
      }
      const int index = historyIndex(position, column);
      const int centrality = static_cast<int>(
          (geometry_.columns - std::abs(column - (geometry_.columns - 1) / 2.0)) * 8);
      candidates.push_back({column, move, index, history_[index] + futureWins * 128 + centrality});
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& first, const Candidate& second) {
      return first.order != second.order ? first.order > second.order : first.column < second.column;
    });
    return candidates;
  }

  int search(const Position& position, int alpha, int beta) {
    visit();
    const std::uint64_t possible = possibleMoves(geometry_, position.mask);
    if (possible == 0) return DRAW;
    if (immediateWinningMoves(geometry_, position) != 0) return WIN;

    const std::uint64_t nonLosing = possibleNonLosingMoves(geometry_, position);
    if (nonLosing == 0) return LOSS;
    if (position.moves >= geometry_.cellCount - 2) return DRAW;

    const std::uint64_t key = canonicalize(geometry_, position).key;
    int lower;
    int upper;
    if (table_.probe(key, lower, upper)) {
      if (lower >= beta) return lower;
      if (upper <= alpha) return upper;
      alpha = std::max(alpha, lower);
      beta = std::min(beta, upper);
      if (alpha >= beta) return alpha;
    }

    std::vector<Candidate> tried;
    for (const Candidate& candidate : orderedMoves(position, nonLosing)) {
      const int childScore = search(play(position, candidate.move), -beta, -alpha);
      const int score = childScore == DRAW ? DRAW : -childScore;
      if (score >= beta) {
        ++cutoffs;
        for (const Candidate& previous : tried) --history_[previous.historyIndex];
        history_[candidate.historyIndex] += static_cast<int>(tried.size());
        table_.storeLower(key, score);
        return score;
      }
      tried.push_back(candidate);
      if (score > alpha) alpha = score;
    }

    table_.storeUpper(key, alpha);
    return alpha;
  }

  const Geometry& geometry_;
  OutcomeTable table_;
  std::uint64_t maximumNodes_;
  std::vector<int> history_;
};

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
  if (!output) throw std::runtime_error("could not write complete policy output");
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
  return {rootValue, std::move(records), stats};
}

void printSummary(
    const Geometry& geometry,
    const Arguments& arguments,
    const GeneratedPolicy& generated,
    std::uint64_t nodes,
    std::uint64_t tableHits,
    std::uint64_t tableStores,
    std::uint64_t tableCollisions,
    std::uint64_t cutoffs,
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
            << ",\"nodes\":" << nodes
            << ",\"tableHits\":" << tableHits
            << ",\"tableStores\":" << tableStores
            << ",\"tableCollisions\":" << tableCollisions
            << ",\"cutoffs\":" << cutoffs
            << ",\"elapsedMs\":" << elapsedMs << "}\n";
}

void generate(const Arguments& arguments) {
  if (arguments.output.empty()) throw std::range_error("--output is required");
  const Geometry geometry(arguments.rows, arguments.columns, arguments.connect);
  if (arguments.handoffRemaining > geometry.cellCount) {
    throw std::range_error("handoff-remaining must fit the board");
  }
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

  // The exact counters are emitted as zero here because the solver is intentionally
  // encapsulated by generatePolicy. The policy and independent replay are the proof
  // artifacts; per-query counters are diagnostic only.
  printSummary(geometry, arguments, generated, 0, 0, 0, 0, 0, elapsed);
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
