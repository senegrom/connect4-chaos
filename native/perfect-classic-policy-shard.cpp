#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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
  std::array<int, 7> columnOrder{};

  Geometry(int selectedRows, int selectedColumns, int selectedConnect)
      : rows(selectedRows),
        columns(selectedColumns),
        connect(selectedConnect),
        stride(selectedRows + 1),
        cellCount(selectedRows * selectedColumns),
        columnBits((std::uint64_t{1} << selectedRows) - 1),
        columnWithSentinel((std::uint64_t{1} << (selectedRows + 1)) - 1) {
    if (rows < 1 || rows > 7 || columns < 1 || columns > 7) {
      throw std::range_error("rows and columns must be from 1 through 7");
    }
    if (connect != 4) {
      throw std::range_error("the optimized policy generator currently requires Connect Four");
    }
    if (connect > std::max(rows, columns)) {
      throw std::range_error("connect must fit the board");
    }
    if (stride * columns >= 63) {
      throw std::range_error("board encoding collides with the closure turn bit");
    }

    for (int column = 0; column < columns; ++column) {
      bottomMasks[column] = std::uint64_t{1} << (column * stride);
      columnMasks[column] = bottomMasks[column] * columnBits;
      bottomMask |= bottomMasks[column];
      columnOrder[column] = column;
    }
    boardMask = bottomMask * columnBits;
    const double centre = static_cast<double>(columns - 1) / 2.0;
    std::sort(columnOrder.begin(), columnOrder.begin() + columns,
        [centre](int first, int second) {
          const double firstDistance = std::abs(first - centre);
          const double secondDistance = std::abs(second - centre);
          return firstDistance != secondDistance
              ? firstDistance < secondDistance
              : first < second;
        });
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

bool hasAlignment(const Geometry& geometry, std::uint64_t pieces) {
  for (const int direction : {1, geometry.rows, geometry.stride, geometry.rows + 2}) {
    const std::uint64_t pair = pieces & (pieces >> direction);
    if ((pair & (pair >> (2 * direction))) != 0) return true;
  }
  return false;
}

std::uint64_t winningPosition(
    const Geometry& geometry,
    std::uint64_t pieces,
    std::uint64_t mask) {
  std::uint64_t result = (pieces << 1) & (pieces << 2) & (pieces << 3);
  for (const int direction : {geometry.rows, geometry.stride, geometry.rows + 2}) {
    std::uint64_t pair =
        (pieces << direction) & (pieces << (2 * direction));
    result |= pair & (pieces << (3 * direction));
    result |= pair & (pieces >> direction);
    pair = (pieces >> direction) & (pieces >> (2 * direction));
    result |= pair & (pieces << direction);
    result |= pair & (pieces >> (3 * direction));
  }
  return result & (geometry.boardMask ^ mask);
}

std::uint64_t immediateWinningMoves(const Geometry& geometry, const Position& position) {
  return possibleMoves(geometry, position.mask)
      & winningPosition(geometry, position.current, position.mask);
}

std::uint64_t possibleNonLosingMoves(const Geometry& geometry, const Position& position) {
  std::uint64_t possible = possibleMoves(geometry, position.mask);
  const std::uint64_t opponent = position.current ^ position.mask;
  const std::uint64_t opponentWinning =
      winningPosition(geometry, opponent, position.mask);
  const std::uint64_t forced = possible & opponentWinning;
  if (forced != 0) {
    if ((forced & (forced - 1)) != 0) return 0;
    possible = forced;
  }
  return possible & ~(opponentWinning >> 1);
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
  return normal <= mirrored
      ? CanonicalPosition{position, normal}
      : CanonicalPosition{
          Position{mirroredCurrent, mirroredMask, position.moves},
          mirrored,
        };
}

class OutcomeTable {
 public:
  explicit OutcomeTable(int bits)
      : size_(std::size_t{1} << bits),
        indexMask_(size_ - 1),
        keys_(size_),
        lower_(size_),
        upper_(size_),
        flags_(size_) {}

  bool probe(std::uint64_t key, int& lower, int& upper) {
    const std::size_t slot = index(key);
    if (keys_[slot] != key + 1) return false;
    ++hits;
    lower = (flags_[slot] & 1) == 0 ? -127 : lower_[slot];
    upper = (flags_[slot] & 2) == 0 ? 127 : upper_[slot];
    return true;
  }

  void storeLower(std::uint64_t key, int score) {
    const std::size_t slot = prepare(key);
    if ((flags_[slot] & 1) != 0 && score <= lower_[slot]) return;
    lower_[slot] = static_cast<std::int8_t>(score);
    flags_[slot] |= 1;
    ++stores;
  }

  void storeUpper(std::uint64_t key, int score) {
    const std::size_t slot = prepare(key);
    if ((flags_[slot] & 2) != 0 && score >= upper_[slot]) return;
    upper_[slot] = static_cast<std::int8_t>(score);
    flags_[slot] |= 2;
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

  std::size_t prepare(std::uint64_t key) {
    const std::size_t slot = index(key);
    const std::uint64_t stored = keys_[slot];
    if (stored != 0 && stored != key + 1) ++collisions;
    if (stored != key + 1) {
      keys_[slot] = key + 1;
      flags_[slot] = 0;
    }
    return slot;
  }

  std::size_t size_;
  std::size_t indexMask_;
  std::vector<std::uint64_t> keys_;
  std::vector<std::int8_t> lower_;
  std::vector<std::int8_t> upper_;
  std::vector<std::uint8_t> flags_;
};

struct Candidate {
  int column = -1;
  std::uint64_t move = 0;
  int historyIndex = 0;
  int order = 0;
};

struct CandidateList {
  std::array<Candidate, 7> values{};
  int count = 0;

  Candidate* begin() { return values.data(); }
  Candidate* end() { return values.data() + count; }
  const Candidate* begin() const { return values.data(); }
  const Candidate* end() const { return values.data() + count; }
};

struct MoveChoice {
  int value;
  int column;
};

class ExactSolver {
 public:
  ExactSolver(const Geometry& geometry, int tableBits, std::uint64_t maximumNodes)
      : geometry_(geometry), table_(tableBits), maximumNodes_(maximumNodes), history_{} {}

  int solveStrong(const Position& position) {
    if (immediateWinningMoves(geometry_, position) != 0) {
      return (geometry_.cellCount + 1 - position.moves) / 2;
    }
    int minimum = -(geometry_.cellCount - position.moves) / 2;
    int maximum = (geometry_.cellCount + 1 - position.moves) / 2;
    while (minimum < maximum) {
      int middle = minimum + (maximum - minimum) / 2;
      if (middle <= 0 && minimum / 2 < middle) middle = minimum / 2;
      else if (middle >= 0 && maximum / 2 > middle) middle = maximum / 2;
      const int score = search(position, middle, middle + 1);
      if (score <= middle) maximum = score;
      else minimum = score;
    }
    return minimum;
  }

  int solve(const Position& position) {
    const int score = solveStrong(position);
    return score > 0 ? WIN : score < 0 ? LOSS : DRAW;
  }

  MoveChoice choose(const Position& position) {
    const std::uint64_t possible = possibleMoves(geometry_, position.mask);
    if (possible == 0) return {DRAW, -1};
    const std::uint64_t winning = immediateWinningMoves(geometry_, position);
    if (winning != 0) {
      for (int index = 0; index < geometry_.columns; ++index) {
        const int column = geometry_.columnOrder[index];
        if ((winning & moveForColumn(geometry_, position.mask, column)) != 0) return {WIN, column};
      }
    }
    const std::uint64_t safe = possibleNonLosingMoves(geometry_, position);
    if (safe == 0) return {LOSS, closureCandidates(position, possible).values[0].column};
    const int strongValue = solveStrong(position);
    const int value = strongValue > 0 ? WIN : strongValue < 0 ? LOSS : DRAW;
    const CandidateList candidates = closureCandidates(position, safe);
    if (value == LOSS) return {LOSS, candidates.values[0].column};
    for (const Candidate& candidate : candidates) {
      const int moveValue = -solveStrong(play(position, candidate.move));
      if ((value == WIN && moveValue > 0) || (value == DRAW && moveValue == 0)) {
        return {value, candidate.column};
      }
    }
    throw std::runtime_error("exact solver could not recover an optimal move");
  }

  std::uint64_t nodes = 0;
  std::uint64_t cutoffs = 0;
  OutcomeTable& table() { return table_; }

 private:
  void visit() {
    ++nodes;
    if (maximumNodes_ != 0 && nodes > maximumNodes_) throw std::runtime_error("node-limit");
  }
  int historyIndex(const Position& position, int column) const {
    const int row = __builtin_popcountll(position.mask & geometry_.columnMasks[column]);
    return (position.moves & 1) * 56 + column * 8 + row;
  }
  CandidateList orderedMoves(const Position& position, std::uint64_t moves) {
    CandidateList candidates;
    for (int orderIndex = 0; orderIndex < geometry_.columns; ++orderIndex) {
      const int column = geometry_.columnOrder[orderIndex];
      const std::uint64_t move = moveForColumn(geometry_, position.mask, column);
      if ((moves & move) == 0) continue;
      const std::uint64_t nextMask = position.mask | move;
      const int futureWins = __builtin_popcountll(winningPosition(geometry_, position.current | move, nextMask));
      const int historySlot = historyIndex(position, column);
      const int centrality = static_cast<int>((geometry_.columns - std::abs(column - (geometry_.columns - 1) / 2.0)) * 8);
      Candidate candidate{column, move, historySlot, history_[historySlot] + futureWins * 128 + centrality};
      int insertion = candidates.count;
      while (insertion > 0) {
        const Candidate& previous = candidates.values[insertion - 1];
        if (previous.order > candidate.order || (previous.order == candidate.order && previous.column < candidate.column)) break;
        candidates.values[insertion] = previous;
        --insertion;
      }
      candidates.values[insertion] = candidate;
      ++candidates.count;
    }
    return candidates;
  }
  int continuationCount(const Position& position, const Candidate& candidate) {
    if (hasAlignment(geometry_, position.current | candidate.move)) return 0;
    const Position opponent = play(position, candidate.move);
    const std::uint64_t possible = possibleMoves(geometry_, opponent.mask);
    if (possible == 0) return 0;
    std::array<std::uint64_t, 7> keys{};
    int count = 0;
    for (int index = 0; index < geometry_.columns; ++index) {
      const int column = geometry_.columnOrder[index];
      const std::uint64_t reply = moveForColumn(geometry_, opponent.mask, column);
      if ((possible & reply) == 0 || hasAlignment(geometry_, opponent.current | reply)) continue;
      const Position child = play(opponent, reply);
      if (possibleMoves(geometry_, child.mask) == 0) continue;
      const std::uint64_t key = canonicalize(geometry_, child).key;
      bool duplicate = false;
      for (int existing = 0; existing < count; ++existing) if (keys[existing] == key) duplicate = true;
      if (!duplicate) keys[count++] = key;
    }
    return count;
  }
  CandidateList closureCandidates(const Position& position, std::uint64_t moves) {
    CandidateList candidates = orderedMoves(position, moves);
    std::array<int, 7> continuationCounts{};
    for (int index = 0; index < candidates.count; ++index) continuationCounts[index] = continuationCount(position, candidates.values[index]);
    for (int index = 1; index < candidates.count; ++index) {
      const Candidate candidate = candidates.values[index];
      const int continuations = continuationCounts[index];
      int insertion = index;
      while (insertion > 0) {
        const int previousContinuations = continuationCounts[insertion - 1];
        const Candidate& previous = candidates.values[insertion - 1];
        if (previousContinuations < continuations || (previousContinuations == continuations && (previous.order > candidate.order || (previous.order == candidate.order && previous.column < candidate.column)))) break;
        candidates.values[insertion] = previous;
        continuationCounts[insertion] = previousContinuations;
        --insertion;
      }
      candidates.values[insertion] = candidate;
      continuationCounts[insertion] = continuations;
    }
    return candidates;
  }
  int search(const Position& rawPosition, int alpha, int beta) {
    visit();
    const CanonicalPosition canonical = canonicalize(geometry_, rawPosition);
    const Position& position = canonical.position;
    if (possibleMoves(geometry_, position.mask) == 0) return DRAW;
    if (immediateWinningMoves(geometry_, position) != 0) return (geometry_.cellCount + 1 - position.moves) / 2;
    const std::uint64_t nonLosing = possibleNonLosingMoves(geometry_, position);
    if (nonLosing == 0) return -(geometry_.cellCount - position.moves) / 2;
    if (position.moves >= geometry_.cellCount - 2) return DRAW;
    const int minimum = -(geometry_.cellCount - 2 - position.moves) / 2;
    if (alpha < minimum) { alpha = minimum; if (alpha >= beta) return alpha; }
    const int maximum = (geometry_.cellCount - 1 - position.moves) / 2;
    if (beta > maximum) { beta = maximum; if (alpha >= beta) return beta; }
    int cachedLower, cachedUpper;
    if (table_.probe(canonical.key, cachedLower, cachedUpper)) {
      if (cachedLower >= beta) return cachedLower;
      if (cachedUpper <= alpha) return cachedUpper;
      alpha = std::max(alpha, cachedLower);
      beta = std::min(beta, cachedUpper);
      if (alpha >= beta) return alpha;
    }
    std::array<int, 7> triedHistory{};
    int triedCount = 0;
    const CandidateList candidates = orderedMoves(position, nonLosing);
    for (const Candidate& candidate : candidates) {
      const int score = -search(play(position, candidate.move), -beta, -alpha);
      if (score >= beta) {
        ++cutoffs;
        for (int index = 0; index < triedCount; ++index) --history_[triedHistory[index]];
        history_[candidate.historyIndex] += triedCount;
        table_.storeLower(canonical.key, score);
        return score;
      }
      triedHistory[triedCount++] = candidate.historyIndex;
      if (score > alpha) alpha = score;
    }
    table_.storeUpper(canonical.key, alpha);
    return alpha;
  }
  const Geometry& geometry_;
  OutcomeTable table_;
  std::uint64_t maximumNodes_;
  std::array<int, 112> history_;
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

struct GeneratedPolicy {
  int rootValue;
  std::vector<PolicyRecord> records;
  std::vector<Position> frontier;
  GenerationStats stats;
  std::uint64_t nodes;
  std::uint64_t tableHits;
  std::uint64_t tableStores;
  std::uint64_t tableCollisions;
  std::uint64_t cutoffs;
};

std::uint64_t closureKey(const CanonicalPosition& canonical, bool aiTurn) {
  return canonical.key | (aiTurn ? AI_TURN_BIT : 0);
}

GeneratedPolicy generatePolicy(
    const Geometry& geometry,
    std::uint8_t role,
    const Position& start,
    int handoffRemaining,
    int frontierRemaining,
    int tableBits,
    std::uint64_t maximumNodes,
    std::uint64_t maximumStates) {
  if (role != ROLE_FIRST && role != ROLE_SECOND) {
    throw std::range_error("role must be first or second");
  }
  if (handoffRemaining < 0 || handoffRemaining > geometry.cellCount) {
    throw std::range_error("handoff-remaining must fit the board");
  }
  if (frontierRemaining >= 0
      && (frontierRemaining <= handoffRemaining
          || frontierRemaining >= geometry.cellCount - start.moves)) {
    throw std::range_error(
        "frontier-remaining must be above the handoff and below the start remainder");
  }
  const bool aiTurn = (role == ROLE_FIRST) == ((start.moves & 1) == 0);
  if (!aiTurn) throw std::range_error("fragment start must be an AI decision state");

  ExactSolver solver(geometry, tableBits, maximumNodes);
  const int rootValue = solver.solve(start);

  std::vector<ClosureState> queue;
  std::unordered_set<std::uint64_t> seen;
  seen.reserve(1 << 20);
  std::vector<Position> frontier;
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

  enqueue(start, true);
  std::vector<PolicyRecord> records;
  for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
    const ClosureState state = queue[cursor];
    ++stats.closureStates;
    const int remaining = geometry.cellCount - state.position.moves;

    if (state.aiTurn && remaining <= handoffRemaining) {
      ++stats.handoffStates;
      continue;
    }
    if (state.aiTurn && frontierRemaining >= 0
        && remaining <= frontierRemaining) {
      frontier.push_back(state.position);
      continue;
    }

    const std::uint64_t possible = possibleMoves(geometry, state.position.mask);
    if (possible == 0) {
      ++stats.terminalDraws;
      continue;
    }

    if (state.aiTurn) {
      ++stats.aiStates;
      const MoveChoice choice = solver.choose(state.position);
      if (choice.column < 0) {
        ++stats.terminalDraws;
        continue;
      }
      records.push_back({
          canonicalize(geometry, state.position).key,
          static_cast<std::uint8_t>(1u << choice.column),
          static_cast<std::int8_t>(choice.value),
      });
      const std::uint64_t move =
          moveForColumn(geometry, state.position.mask, choice.column);
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
    for (int index = 0; index < geometry.columns; ++index) {
      const int column = geometry.columnOrder[index];
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

  std::sort(records.begin(), records.end(),
      [](const PolicyRecord& first, const PolicyRecord& second) {
        return first.key < second.key;
      });
  for (std::size_t index = 1; index < records.size(); ++index) {
    if (records[index - 1].key == records[index].key) {
      throw std::runtime_error("duplicate policy key");
    }
  }

  return {
    rootValue,
    std::move(records),
    std::move(frontier),
    stats,
    solver.nodes,
    solver.table().hits,
    solver.table().stores,
    solver.table().collisions,
    solver.cutoffs,
  };
}

void writeByte(std::ofstream& output, std::uint8_t value) {
  output.put(static_cast<char>(value));
}

void writeUint32(std::ofstream& output, std::uint32_t value) {
  for (int index = 0; index < 4; ++index) {
    writeByte(output, static_cast<std::uint8_t>((value >> (index * 8)) & 0xff));
  }
}

void writeUint64(std::ofstream& output, std::uint64_t value) {
  for (int index = 0; index < 8; ++index) {
    writeByte(output, static_cast<std::uint8_t>((value >> (index * 8)) & 0xff));
  }
}

void writePolicy(
    const std::string& path,
    const Geometry& geometry,
    std::uint8_t role,
    int handoffRemaining,
    const GeneratedPolicy& generated) {
  if (generated.records.size() > std::numeric_limits<std::uint32_t>::max()
      || generated.stats.closureStates > std::numeric_limits<std::uint32_t>::max()) {
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
  writeByte(output, static_cast<std::uint8_t>(
      static_cast<std::int8_t>(generated.rootValue)));
  writeUint32(output, static_cast<std::uint32_t>(generated.records.size()));
  writeUint32(output, static_cast<std::uint32_t>(generated.stats.closureStates));
  for (const PolicyRecord& record : generated.records) {
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
  int frontierRemaining = -1;
  int tableBits = 24;
  std::uint64_t maximumNodes = 0;
  std::uint64_t maximumStates = 100'000'000;
  Position start{};
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
    } else if (name == "--frontier-remaining") {
      arguments.frontierRemaining = parseInt(value, 0, 49, "frontier-remaining");
    } else if (name == "--table-bits") arguments.tableBits = parseInt(value, 8, 27, "table-bits");
    else if (name == "--maximum-nodes") {
      arguments.maximumNodes = parseUint64(value, "maximum-nodes");
    } else if (name == "--maximum-states") {
      arguments.maximumStates = parseUint64(value, "maximum-states");
    } else if (name == "--start-current") {
      arguments.start.current = parseUint64(value, "start-current");
    } else if (name == "--start-mask") {
      arguments.start.mask = parseUint64(value, "start-mask");
    } else if (name == "--start-moves") {
      arguments.start.moves = parseInt(value, 0, 49, "start-moves");
    } else if (name == "--output") arguments.output = value;
    else throw std::range_error("unknown argument: " + name);
  }
  return arguments;
}

void validateStart(const Geometry& geometry, const Position& start) {
  if (start.moves < 0 || start.moves > geometry.cellCount) {
    throw std::range_error("start-moves must fit the board");
  }
  if ((start.current & ~start.mask) != 0
      || (start.mask & ~geometry.boardMask) != 0
      || __builtin_popcountll(start.mask) != start.moves
      || __builtin_popcountll(start.current) != start.moves / 2) {
    throw std::range_error("fragment start bitboards are inconsistent");
  }
  for (int column = 0; column < geometry.columns; ++column) {
    const std::uint64_t occupied =
        (start.mask >> (column * geometry.stride)) & geometry.columnBits;
    const int height = __builtin_popcountll(occupied);
    const std::uint64_t expected = height == 0 ? 0 : (std::uint64_t{1} << height) - 1;
    if (occupied != expected) throw std::range_error("fragment start violates gravity");
  }
  if (hasAlignment(geometry, start.current)
      || hasAlignment(geometry, start.current ^ start.mask)) {
    throw std::range_error("fragment start is already terminal");
  }
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
            << ",\"frontierStates\":" << generated.frontier.size()
            << ",\"startCurrent\":\"" << arguments.start.current << "\""
            << ",\"startMask\":\"" << arguments.start.mask << "\""
            << ",\"startMoves\":" << arguments.start.moves
            << ",\"elapsedMs\":" << elapsedMs << "}\n";
  for (const Position& state : generated.frontier) {
    std::cout << "{\"format\":\"connect4-perfect-classic-fragment-frontier-v1\""
              << ",\"current\":\"" << state.current << "\""
              << ",\"mask\":\"" << state.mask << "\""
              << ",\"moves\":" << state.moves << "}\n";
  }
}

void generate(const Arguments& arguments) {
  if (arguments.output.empty()) throw std::range_error("--output is required");
  const Geometry geometry(arguments.rows, arguments.columns, arguments.connect);
  if (arguments.handoffRemaining > geometry.cellCount) {
    throw std::range_error("handoff-remaining must fit the board");
  }
  validateStart(geometry, arguments.start);
  const auto start = std::chrono::steady_clock::now();
  const GeneratedPolicy generated = generatePolicy(
      geometry,
      static_cast<std::uint8_t>(arguments.role),
      arguments.start,
      arguments.handoffRemaining,
      arguments.frontierRemaining,
      arguments.tableBits,
      arguments.maximumNodes,
      arguments.maximumStates);
  writePolicy(
      arguments.output,
      geometry,
      static_cast<std::uint8_t>(arguments.role),
      arguments.handoffRemaining,
      generated);
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start).count();
  printSummary(geometry, arguments, generated, elapsed);
}

void verify() {
  struct Case { int rows; int columns; int value; };
  const std::array<Case, 4> cases{{
      {4, 4, DRAW},
      {4, 5, DRAW},
      {4, 6, LOSS},
      {5, 4, DRAW},
  }};
  for (const Case& test : cases) {
    const Geometry geometry(test.rows, test.columns, 4);
    const GeneratedPolicy generated = generatePolicy(
        geometry,
        ROLE_FIRST,
        Position{},
        std::min(24, geometry.cellCount),
        -1,
        20,
        100'000'000,
        10'000'000);
    if (generated.rootValue != test.value) {
      throw std::runtime_error("optimized policy verification root-value mismatch");
    }
    std::cout << "{\"format\":\"connect4-perfect-classic-fast-verification-v1\""
              << ",\"rows\":" << test.rows
              << ",\"columns\":" << test.columns
              << ",\"rootValue\":" << generated.rootValue
              << ",\"entryCount\":" << generated.records.size()
              << ",\"closureStates\":" << generated.stats.closureStates
              << ",\"nodes\":" << generated.nodes << "}\n";
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
