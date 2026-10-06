// The exact W/D/L search behind the two classic programs: perfect-classic.cpp
// solves single positions with it, and perfect-classic-policy.cpp every AI
// decision of a policy closure.
//
// A board of up to 7x7 is a bitboard with one sentinel bit per column (bit
// column * (rows + 1) + row), so line runs cannot wrap between columns. A
// position holds the stones of the side to move and the occupied mask, and
// its table key is the smaller of its own and its mirror image's encoding.
//
// One copy serves both programs. Nothing compares their answers with each
// other, so separate copies were no independent check; the independent checks
// are scripts/perfect-classic.mjs, which compares root values with the
// published table, and the JavaScript replay every generated policy must pass.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace connect4::classic {

constexpr int WIN = 1;
constexpr int DRAW = 0;
constexpr int LOSS = -1;

// Validate before member initializers perform arithmetic, shifts or allocations.
inline int checkedDimension(int dimension) {
  if (dimension < 1 || dimension > 7) {
    throw std::range_error("rows and columns must be from 1 through 7");
  }
  return dimension;
}

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
      : rows(checkedDimension(selectedRows)),
        columns(checkedDimension(selectedColumns)),
        connect(selectedConnect),
        stride(rows + 1),
        cellCount(rows * columns),
        columnBits((std::uint64_t{1} << rows) - 1),
        columnWithSentinel((std::uint64_t{1} << stride) - 1),
        columnOrder(static_cast<std::size_t>(columns)) {
    if (connect < 1 || connect > std::max(rows, columns)) {
      throw std::range_error("connect must fit the board");
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

inline std::uint64_t possibleMoves(const Geometry& geometry, std::uint64_t mask) {
  return (mask + geometry.bottomMask) & geometry.boardMask;
}

inline std::uint64_t moveForColumn(const Geometry& geometry, std::uint64_t mask, int column) {
  if (column < 0 || column >= geometry.columns) return 0;
  return (mask + geometry.bottomMasks[column]) & geometry.columnMasks[column];
}

inline Position play(Position position, std::uint64_t move) {
  position.current ^= position.mask;
  position.mask |= move;
  ++position.moves;
  return position;
}

inline bool hasAlignment(const Geometry& geometry, std::uint64_t bits) {
  for (const int direction : geometry.directions) {
    std::uint64_t run = bits;
    for (int offset = 1; offset < geometry.connect && run != 0; ++offset) {
      run &= bits >> (offset * direction);
    }
    if (run != 0) return true;
  }
  return false;
}

inline std::uint64_t mirrorBits(const Geometry& geometry, std::uint64_t bits) {
  std::uint64_t mirrored = 0;
  for (int column = 0; column < geometry.columns; ++column) {
    const std::uint64_t group =
        (bits >> (column * geometry.stride)) & geometry.columnWithSentinel;
    mirrored |= group << ((geometry.columns - 1 - column) * geometry.stride);
  }
  return mirrored;
}

// The orientation with the smaller key, and that key. A key spans at most
// (7 + 1) * 7 = 56 bits.
inline CanonicalPosition canonicalize(const Geometry& geometry, const Position& position) {
  const std::uint64_t normal = position.current + position.mask;
  const std::uint64_t mirroredCurrent = mirrorBits(geometry, position.current);
  const std::uint64_t mirroredMask = mirrorBits(geometry, position.mask);
  const std::uint64_t mirrored = mirroredCurrent + mirroredMask;
  if (normal <= mirrored) return {position, normal, false};
  return {{mirroredCurrent, mirroredMask, position.moves}, mirrored, true};
}

inline std::uint64_t immediateWinningMoves(
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

inline std::uint64_t immediateWinningMoves(const Geometry& geometry, const Position& position) {
  return immediateWinningMoves(geometry, position, position.current);
}

inline std::uint64_t possibleNonLosingMoves(const Geometry& geometry, const Position& position) {
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
        history_(std::size_t{2} * static_cast<std::size_t>(geometry.columns)
                     * static_cast<std::size_t>(geometry.stride), 0) {}

  // One best column and the position's value, stopping at the first win.
  std::pair<int, int> root(const Position& position) {
    const std::uint64_t possible = possibleMoves(geometry_, position.mask);
    if (possible == 0) return {-1, DRAW};

    const std::uint64_t winning = immediateWinningMoves(geometry_, position);
    if (winning != 0) {
      for (const int column : geometry_.columnOrder) {
        if ((winning & moveForColumn(geometry_, position.mask, column)) != 0) {
          return {column, WIN};
        }
      }
    }

    const std::uint64_t nonLosing = possibleNonLosingMoves(geometry_, position);
    if (nonLosing == 0) {
      for (const int column : geometry_.columnOrder) {
        if ((possible & moveForColumn(geometry_, position.mask, column)) != 0) {
          return {column, LOSS};
        }
      }
    }

    int bestColumn = -1;
    int bestOutcome = -2;
    for (const Candidate& candidate : orderedMoves(position, nonLosing)) {
      const int childOutcome = solve(play(position, candidate.move));
      const int outcome = childOutcome == DRAW ? DRAW : -childOutcome;
      if (outcome > bestOutcome) {
        bestOutcome = outcome;
        bestColumn = candidate.column;
      }
      if (bestOutcome == WIN) break;
    }
    return {bestColumn, bestOutcome};
  }

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

  // The position's value and every optimal column, each child solved exactly.
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

// Command-line numbers, refused unless the whole value parses.
inline int parseInt(std::string_view value, int minimum, int maximum, std::string_view label) {
  std::size_t parsed = 0;
  const int result = std::stoi(std::string(value), &parsed);
  if (parsed != value.size() || result < minimum || result > maximum) {
    throw std::range_error(std::string(label) + " is outside its supported range");
  }
  return result;
}

inline std::uint64_t parseUint64(std::string_view value, std::string_view label) {
  std::size_t parsed = 0;
  const std::uint64_t result = std::stoull(std::string(value), &parsed);
  if (parsed != value.size()) throw std::range_error(std::string(label) + " is invalid");
  return result;
}

}  // namespace connect4::classic
