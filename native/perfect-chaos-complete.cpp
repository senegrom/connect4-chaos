// Complete exact solver for small Chaos Mode boards.
//
// Chaos Mode is a directed graph rather than a tree: flips and rotations can
// revisit earlier positions, and a rotation transposes the board. This solves
// the whole reachable graph from the empty board by ranked retrograde
// analysis, using the same game model as src/chaos-solver.js:
//
//   * colours are normalised so the side to move is always "mover";
//   * horizontal reflection is canonicalised;
//   * terminal outcomes are edge labels, not nodes;
//   * a transformation that completes lines for both players loses for the
//     player who made it;
//   * closed unresolved cycles are draws, which is what the automatic
//     threefold-repetition rule produces from a fresh root.
//
// Memory is sized by the number of REACHABLE states, not by the index space.
// A dense mixed-radix index over every gravity-valid arrangement is used only
// as a key; a rank/select bitset maps it to a compact ordinal. That is what
// lets 10^9-state boards fit where the naive per-index arrays would not.
// Value resolution proceeds by rank iteration with successor lists
// regenerated on demand each round, so neither a forward-edge nor a
// reverse-edge list is ever materialised.
//
// Index: each column of an R-row board is a stack of height h with h colour
// bits, encoded as (2^h - 1) + colourBits, so a column takes 2^(R+1)-1 values
// and a board is a mixed-radix number over its columns. A board and its
// transpose share one index space laid out as two consecutive blocks.
//
// Certificates (C4CFUL1) are the closure a starting role actually reaches: one
// stored action per AI position, every legal opponent reply explored, with the
// mover-relative value of every AI position recorded. A drawn position prefers
// an action back into a state the closure already contains, which roughly
// halves the file; a won position keeps the rank-reducing action that makes
// the win finite. scripts/perfect-chaos-complete.mjs replays every certificate
// through the game's own engine.js before it is accepted.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

constexpr int WIN = 1;
constexpr int DRAW = 0;
constexpr int LOSS = -1;
constexpr int NOT_TERMINAL = 2;
constexpr std::uint8_t VALUE_UNKNOWN = 3;
constexpr std::uint8_t NO_ACTION = 0xffu;
constexpr int MAX_SIDE = 8;
constexpr int MAX_EDGES = MAX_SIDE + 3;   // every column plus flip and two rotations

constexpr int ACTION_DROP = 0;
constexpr int ACTION_FLIP = 1;
constexpr int ACTION_ROTATE_CW = 2;
constexpr int ACTION_ROTATE_CCW = 3;

// ---------------------------------------------------------------------------
// Board model
// ---------------------------------------------------------------------------

struct Board {
  int rows = 0;
  int columns = 0;
  // cells[row][column], row 0 at the bottom. 0 empty, 1 mover, 2 opponent.
  std::array<std::array<std::uint8_t, MAX_SIDE>, MAX_SIDE> cells{};

  void clear(int selectedRows, int selectedColumns) {
    rows = selectedRows;
    columns = selectedColumns;
    for (auto& row : cells) row.fill(0);
  }

  int height(int column) const {
    int count = 0;
    while (count < rows && cells[count][column] != 0) ++count;
    return count;
  }

  bool full() const {
    for (int column = 0; column < columns; ++column) {
      if (cells[rows - 1][column] == 0) return false;
    }
    return true;
  }
};

struct Layout {
  int rows = 0;
  int columns = 0;
  std::uint64_t radix = 0;
  std::uint64_t size = 0;
  std::uint64_t offset = 0;
};

struct Geometry {
  int connect = 0;
  std::array<Layout, 2> shapes{};
  int shapeCount = 0;
  std::uint64_t total = 0;

  const Layout& layoutFor(int rows, int columns) const {
    for (int index = 0; index < shapeCount; ++index) {
      if (shapes[index].rows == rows && shapes[index].columns == columns) return shapes[index];
    }
    throw std::runtime_error("board shape is outside the solved index space");
  }
};

Geometry makeGeometry(int rows, int columns, int connect) {
  if (rows < 1 || rows >= MAX_SIDE || columns < 1 || columns >= MAX_SIDE) {
    throw std::range_error("board dimensions are out of range");
  }
  if (connect < 1 || connect > std::max(rows, columns)) {
    throw std::range_error("connect length does not fit the board");
  }
  Geometry geometry;
  geometry.connect = connect;
  auto addShape = [&geometry](int shapeRows, int shapeColumns) {
    Layout layout;
    layout.rows = shapeRows;
    layout.columns = shapeColumns;
    layout.radix = (std::uint64_t{1} << (shapeRows + 1)) - 1;
    layout.size = 1;
    for (int index = 0; index < shapeColumns; ++index) {
      layout.size *= layout.radix;
      if (layout.size > (std::uint64_t{1} << 36)) {
        throw std::runtime_error("index space is too large for this solver");
      }
    }
    layout.offset = geometry.total;
    geometry.total += layout.size;
    geometry.shapes[geometry.shapeCount++] = layout;
  };
  addShape(rows, columns);
  if (rows != columns) addShape(columns, rows);
  return geometry;
}

std::uint64_t encode(const Geometry& geometry, const Board& board) {
  const Layout& layout = geometry.layoutFor(board.rows, board.columns);
  std::uint64_t index = 0;
  for (int column = board.columns - 1; column >= 0; --column) {
    const int height = board.height(column);
    std::uint64_t colour = 0;
    for (int row = 0; row < height; ++row) {
      if (board.cells[row][column] == 1) colour |= std::uint64_t{1} << row;
    }
    index = index * layout.radix + ((std::uint64_t{1} << height) - 1) + colour;
  }
  return layout.offset + index;
}

void decode(const Geometry& geometry, std::uint64_t global, Board& board) {
  int shape = 0;
  for (int index = geometry.shapeCount - 1; index >= 0; --index) {
    if (global >= geometry.shapes[index].offset) {
      shape = index;
      break;
    }
  }
  const Layout& layout = geometry.shapes[shape];
  std::uint64_t index = global - layout.offset;
  board.clear(layout.rows, layout.columns);
  for (int column = 0; column < layout.columns; ++column) {
    const std::uint64_t value = index % layout.radix;
    index /= layout.radix;
    int height = 0;
    while (((std::uint64_t{1} << (height + 1)) - 1) <= value) ++height;
    const std::uint64_t colour = value - ((std::uint64_t{1} << height) - 1);
    for (int row = 0; row < height; ++row) {
      board.cells[row][column] = ((colour >> row) & 1) != 0 ? 1 : 2;
    }
  }
}

Board mirror(const Board& board) {
  Board result;
  result.clear(board.rows, board.columns);
  for (int row = 0; row < board.rows; ++row) {
    for (int column = 0; column < board.columns; ++column) {
      result.cells[row][board.columns - 1 - column] = board.cells[row][column];
    }
  }
  return result;
}

std::uint64_t canonicalIndex(const Geometry& geometry, const Board& board) {
  return std::min(encode(geometry, board), encode(geometry, mirror(board)));
}

void applyGravity(Board& board) {
  for (int column = 0; column < board.columns; ++column) {
    std::array<std::uint8_t, MAX_SIDE> stack{};
    int count = 0;
    for (int row = 0; row < board.rows; ++row) {
      if (board.cells[row][column] != 0) stack[count++] = board.cells[row][column];
      board.cells[row][column] = 0;
    }
    for (int row = 0; row < count; ++row) board.cells[row][column] = stack[row];
  }
}

Board flipBoard(const Board& board) {
  Board result;
  result.clear(board.rows, board.columns);
  for (int row = 0; row < board.rows; ++row) {
    for (int column = 0; column < board.columns; ++column) {
      result.cells[board.rows - 1 - row][column] = board.cells[row][column];
    }
  }
  applyGravity(result);
  return result;
}

// Same orientation convention as engine.js rotateBoard; row 0 is the bottom
// here and the top there, hence the displayRow conversions.
Board rotateBoard(const Board& board, int direction) {
  Board result;
  result.clear(board.columns, board.rows);
  for (int row = 0; row < board.rows; ++row) {
    for (int column = 0; column < board.columns; ++column) {
      const int displayRow = board.rows - 1 - row;
      int targetDisplayRow;
      int targetColumn;
      if (direction == 1) {
        targetDisplayRow = column;
        targetColumn = board.rows - 1 - displayRow;
      } else {
        targetDisplayRow = board.columns - 1 - column;
        targetColumn = displayRow;
      }
      result.cells[result.rows - 1 - targetDisplayRow][targetColumn] = board.cells[row][column];
    }
  }
  applyGravity(result);
  return result;
}

bool winsThrough(const Board& board, int row, int column, int player, int connect) {
  static constexpr int directions[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
  for (const auto& direction : directions) {
    int count = 1;
    for (const int sign : {1, -1}) {
      int nextRow = row + sign * direction[0];
      int nextColumn = column + sign * direction[1];
      while (nextRow >= 0 && nextRow < board.rows && nextColumn >= 0 && nextColumn < board.columns
             && board.cells[nextRow][nextColumn] == player) {
        ++count;
        nextRow += sign * direction[0];
        nextColumn += sign * direction[1];
      }
    }
    if (count >= connect) return true;
  }
  return false;
}

bool hasLine(const Board& board, int player, int connect) {
  for (int row = 0; row < board.rows; ++row) {
    for (int column = 0; column < board.columns; ++column) {
      if (board.cells[row][column] == player && winsThrough(board, row, column, player, connect)) {
        return true;
      }
    }
  }
  return false;
}

struct Edge {
  std::uint8_t action;
  std::uint8_t column;
  std::int8_t terminal;      // WIN / DRAW / LOSS for the mover, or NOT_TERMINAL
  std::uint64_t next;        // canonical index of the child when non-terminal
};

struct EdgeList {
  std::array<Edge, MAX_EDGES> values{};
  int count = 0;
};

// Every legal action from a mover-relative position; children are
// re-normalised so the next player becomes the mover.
void successors(const Geometry& geometry, const Board& board, EdgeList& edges) {
  edges.count = 0;
  if (board.full()) return;

  auto settle = [&](Board next, int action, int column, bool dropWin) {
    int terminal = NOT_TERMINAL;
    if (action == ACTION_DROP) {
      if (dropWin) terminal = WIN;
      else if (next.full()) terminal = DRAW;
    } else {
      const bool moverLine = hasLine(next, 1, geometry.connect);
      const bool opponentLine = hasLine(next, 2, geometry.connect);
      if (moverLine && opponentLine) terminal = LOSS;
      else if (moverLine) terminal = WIN;
      else if (opponentLine) terminal = LOSS;
      else if (next.full()) terminal = DRAW;
    }
    std::uint64_t child = 0;
    if (terminal == NOT_TERMINAL) {
      for (int row = 0; row < next.rows; ++row) {
        for (int c = 0; c < next.columns; ++c) {
          const std::uint8_t cell = next.cells[row][c];
          if (cell != 0) next.cells[row][c] = cell == 1 ? 2 : 1;
        }
      }
      child = canonicalIndex(geometry, next);
    }
    edges.values[edges.count++] = Edge{
        static_cast<std::uint8_t>(action), static_cast<std::uint8_t>(column),
        static_cast<std::int8_t>(terminal), child};
  };

  for (int column = 0; column < board.columns; ++column) {
    const int height = board.height(column);
    if (height >= board.rows) continue;
    Board next = board;
    next.cells[height][column] = 1;
    settle(next, ACTION_DROP, column, winsThrough(next, height, column, 1, geometry.connect));
  }
  settle(flipBoard(board), ACTION_FLIP, 0, false);
  settle(rotateBoard(board, 1), ACTION_ROTATE_CW, 0, false);
  settle(rotateBoard(board, -1), ACTION_ROTATE_CCW, 0, false);
}

// ---------------------------------------------------------------------------
// Rank/select bitset: canonical index -> compact ordinal
// ---------------------------------------------------------------------------

class ReachableIndex {
 public:
  explicit ReachableIndex(std::uint64_t universe)
      : words_((universe + 63) / 64, 0), universe_(universe) {}

  bool test(std::uint64_t index) const {
    return (words_[index >> 6] >> (index & 63)) & 1;
  }

  void set(std::uint64_t index) {
    words_[index >> 6] |= std::uint64_t{1} << (index & 63);
  }

  // Must be called once all bits are set. Builds the rank superblocks and a
  // sampled select directory: one word position per 512 set bits. Cumulative
  // counts are stored in 32 bits, which caps a solve at 2^32 - 1 reachable
  // states; anything near that could not hold its value arrays in memory
  // anyway, and overflow is refused rather than wrapped.
  void finalize() {
    ranks_.assign(words_.size() + 1, 0);
    selectSample_.clear();
    std::uint64_t running = 0;
    for (std::size_t word = 0; word < words_.size(); ++word) {
      ranks_[word] = static_cast<std::uint32_t>(running);
      const std::uint64_t next =
          running + static_cast<std::uint64_t>(__builtin_popcountll(words_[word]));
      while ((static_cast<std::uint64_t>(selectSample_.size()) << 9) < next) {
        selectSample_.push_back(word);
      }
      running = next;
    }
    if (running > 0xffffffffull) {
      throw std::runtime_error("reachable states exceed the 32-bit rank directory");
    }
    ranks_[words_.size()] = static_cast<std::uint32_t>(running);
    count_ = running;
  }

  // Ordinal of a set bit: number of set bits strictly before it.
  std::uint64_t rank(std::uint64_t index) const {
    const std::uint64_t word = index >> 6;
    const std::uint64_t below = words_[word] & ((std::uint64_t{1} << (index & 63)) - 1);
    return static_cast<std::uint64_t>(ranks_[word])
        + static_cast<std::uint64_t>(__builtin_popcountll(below));
  }

  // Canonical index of the ordinal-th set bit; the inverse of rank().
  std::uint64_t select(std::uint64_t ordinal) const {
    std::uint64_t word = selectSample_[ordinal >> 9];
    std::uint64_t before = ranks_[word];
    for (;;) {
      const std::uint64_t pop =
          static_cast<std::uint64_t>(__builtin_popcountll(words_[word]));
      if (ordinal < before + pop) break;
      before += pop;
      ++word;
    }
    std::uint64_t bits = words_[word];
    for (std::uint64_t skip = ordinal - before; skip != 0; --skip) bits &= bits - 1;
    return word * 64 + static_cast<std::uint64_t>(__builtin_ctzll(bits));
  }

  std::uint64_t count() const { return count_; }
  std::uint64_t universe() const { return universe_; }

  // Iterates set bits in increasing index order.
  template <typename Visit>
  void forEach(Visit&& visit) const {
    forEachInWordRange(0, words_.size(), visit);
  }

  // Iterates the set bits of words [begin, end) in increasing index order.
  template <typename Visit>
  void forEachInWordRange(std::uint64_t begin, std::uint64_t end, Visit&& visit) const {
    for (std::uint64_t word = begin; word < end; ++word) {
      std::uint64_t bits = words_[word];
      while (bits != 0) {
        const int bit = __builtin_ctzll(bits);
        visit(word * 64 + static_cast<std::uint64_t>(bit));
        bits &= bits - 1;
      }
    }
  }

  std::uint64_t wordCount() const { return words_.size(); }

  // Word containing the ordinal-th set bit, from the sample directory: exact
  // to within 512 ordinals, which is all a chunk boundary needs.
  std::uint64_t sampleWordFor(std::uint64_t ordinal) const {
    if (selectSample_.empty()) return 0;
    const std::uint64_t slot = ordinal >> 9;
    return selectSample_[slot < selectSample_.size() ? slot : selectSample_.size() - 1];
  }

  std::size_t bytes() const {
    return words_.size() * sizeof(std::uint64_t) + ranks_.size() * sizeof(std::uint32_t)
        + selectSample_.size() * sizeof(std::uint64_t);
  }

  // Raw bitset words, for checkpointing only. A loader fills them and then
  // calls finalize() as if discovery had just finished.
  const std::vector<std::uint64_t>& words() const { return words_; }
  std::vector<std::uint64_t>& mutableWords() { return words_; }

 private:
  std::vector<std::uint64_t> words_;
  std::vector<std::uint32_t> ranks_;
  std::vector<std::uint64_t> selectSample_;
  std::uint64_t universe_;
  std::uint64_t count_ = 0;
};

// ---------------------------------------------------------------------------
// Solution over compact ordinals
// ---------------------------------------------------------------------------

struct Solution {
  ReachableIndex reachable;
  std::vector<std::uint8_t> value;          // ordinal -> LOSS/DRAW/WIN + 1, or VALUE_UNKNOWN
  std::vector<std::uint8_t> rank;           // ordinal -> attractor rank (0 for draws)
  std::vector<std::uint8_t> action;         // ordinal -> action | column << 2, or NO_ACTION
  std::uint64_t states = 0;
  std::uint64_t wins = 0;
  std::uint64_t draws = 0;
  std::uint64_t losses = 0;
  int maximumRank = 0;
  int rootValue = 0;
  std::uint64_t rootOrdinal = 0;

  explicit Solution(std::uint64_t universe) : reachable(universe) {}
};

std::uint8_t packValue(int outcome) { return static_cast<std::uint8_t>(outcome + 1); }
int unpackValue(std::uint8_t packed) { return static_cast<int>(packed) - 1; }

double secondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start).count() / 1000.0;
}

// ---------------------------------------------------------------------------
// Checkpoints: <path>.bitset once after discovery, <path>.round per round.
// ---------------------------------------------------------------------------

constexpr char CHECKPOINT_MAGIC[8] = {'C', '4', 'C', 'K', 'P', 'T', '1', '\0'};

struct CheckpointHeader {
  char magic[8];
  std::uint8_t rows;
  std::uint8_t columns;
  std::uint8_t connect;
  std::uint8_t zero;
  std::uint32_t pad;
  std::uint64_t universe;
};

CheckpointHeader checkpointHeader(const Geometry& geometry, int rows, int columns) {
  CheckpointHeader header{};
  std::memcpy(header.magic, CHECKPOINT_MAGIC, sizeof(header.magic));
  header.rows = static_cast<std::uint8_t>(rows);
  header.columns = static_cast<std::uint8_t>(columns);
  header.connect = static_cast<std::uint8_t>(geometry.connect);
  header.universe = geometry.total;
  return header;
}

// All checkpoint I/O moves in bounded chunks: multi-gigabyte single calls
// have short-read and torn-write on this platform, both silently.
constexpr std::size_t CHECKPOINT_CHUNK = std::size_t{256} << 20;

bool readExact(std::ifstream& in, void* target, std::size_t bytes) {
  char* cursor = static_cast<char*>(target);
  while (bytes > 0) {
    const std::size_t step = bytes < CHECKPOINT_CHUNK ? bytes : CHECKPOINT_CHUNK;
    in.read(cursor, static_cast<std::streamsize>(step));
    if (in.gcount() != static_cast<std::streamsize>(step)) return false;
    cursor += step;
    bytes -= step;
  }
  return true;
}

bool writeAll(std::ofstream& out, const void* source, std::size_t bytes) {
  const char* cursor = static_cast<const char*>(source);
  while (bytes > 0) {
    const std::size_t step = bytes < CHECKPOINT_CHUNK ? bytes : CHECKPOINT_CHUNK;
    out.write(cursor, static_cast<std::streamsize>(step));
    if (!out) return false;
    cursor += step;
    bytes -= step;
  }
  out.flush();
  return static_cast<bool>(out);
}

bool headerMatches(const CheckpointHeader& seen, const CheckpointHeader& want) {
  return std::memcmp(seen.magic, want.magic, sizeof(want.magic)) == 0
      && seen.rows == want.rows && seen.columns == want.columns
      && seen.connect == want.connect && seen.universe == want.universe;
}

bool loadBitsetCheckpoint(const std::string& path, const CheckpointHeader& want,
                          std::vector<std::uint64_t>& words) {
  std::ifstream in(path + ".bitset", std::ios::binary);
  if (!in) return false;
  const auto reject = [](const char* reason) {
    std::cerr << "[chaos] bitset checkpoint rejected: " << reason << std::endl;
    return false;
  };
  CheckpointHeader seen{};
  std::uint64_t wordCount = 0;
  if (!readExact(in, &seen, sizeof(seen))) return reject("truncated header");
  if (!headerMatches(seen, want)) return reject("header mismatch");
  if (!readExact(in, &wordCount, sizeof(wordCount))) return reject("truncated word count");
  if (wordCount != words.size()) return reject("word count mismatch");
  if (!readExact(in, words.data(), words.size() * sizeof(std::uint64_t))) {
    return reject("short read of the bitset body");
  }
  std::cerr << "[chaos] bitset checkpoint loaded" << std::endl;
  return true;
}

void writeBitsetCheckpoint(const std::string& path, const CheckpointHeader& header,
                           const std::vector<std::uint64_t>& words) {
  const std::string target = path + ".bitset";
  const std::string temporary = target + ".tmp";
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("could not write the bitset checkpoint");
    const std::uint64_t wordCount = words.size();
    if (!writeAll(out, &header, sizeof(header))
        || !writeAll(out, &wordCount, sizeof(wordCount))
        || !writeAll(out, words.data(), words.size() * sizeof(std::uint64_t))) {
      throw std::runtime_error("could not write the bitset checkpoint");
    }
  }
  std::remove(target.c_str());
  if (std::rename(temporary.c_str(), target.c_str()) != 0) {
    throw std::runtime_error("could not publish the bitset checkpoint");
  }
}

struct RoundCheckpoint {
  int round = 0;
  std::uint64_t settledTotal = 0;
  std::uint64_t drawTotal = 0;
};

bool loadRoundCheckpointFrom(const std::string& file, const CheckpointHeader& want,
                             std::uint64_t states, RoundCheckpoint& progress,
                             std::vector<std::uint8_t>& value,
                             std::vector<std::uint8_t>& rank,
                             std::vector<std::uint8_t>& action) {
  std::ifstream in(file, std::ios::binary);
  if (!in) return false;
  CheckpointHeader seen{};
  std::int32_t round = 0;
  std::uint64_t settledTotal = 0;
  std::uint64_t drawTotal = 0;
  std::uint64_t n = 0;
  const auto reject = [&file](const char* reason) {
    std::cerr << "[chaos] round checkpoint " << file << " rejected: " << reason << std::endl;
    return false;
  };
  if (!readExact(in, &seen, sizeof(seen)) || !headerMatches(seen, want)) return reject("header");
  if (!readExact(in, &round, sizeof(round)) || round <= 0) return reject("round");
  if (!readExact(in, &settledTotal, sizeof(settledTotal))) return reject("settled total");
  if (!readExact(in, &drawTotal, sizeof(drawTotal))) return reject("draw total");
  if (!readExact(in, &n, sizeof(n)) || n != states) return reject("state count");
  if (!readExact(in, value.data(), value.size())) return reject("value array");
  if (!readExact(in, rank.data(), rank.size())) return reject("rank array");
  if (!readExact(in, action.data(), action.size())) return reject("action array");
  std::cerr << "[chaos] round checkpoint loaded through round " << round << std::endl;
  progress.round = round;
  progress.settledTotal = settledTotal;
  progress.drawTotal = drawTotal;
  return true;
}

bool loadRoundCheckpoint(const std::string& path, const CheckpointHeader& want,
                         std::uint64_t states, RoundCheckpoint& progress,
                         std::vector<std::uint8_t>& value,
                         std::vector<std::uint8_t>& rank,
                         std::vector<std::uint8_t>& action) {
  return loadRoundCheckpointFrom(path + ".round", want, states, progress, value, rank, action)
      || loadRoundCheckpointFrom(path + ".round.tmp", want, states, progress, value, rank, action);
}

void writeRoundCheckpoint(const std::string& path, const CheckpointHeader& header,
                          const RoundCheckpoint& progress,
                          const std::vector<std::uint8_t>& value,
                          const std::vector<std::uint8_t>& rank,
                          const std::vector<std::uint8_t>& action) {
  const std::string target = path + ".round";
  const std::string temporary = target + ".tmp";
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("could not write the round checkpoint");
    const std::int32_t round = progress.round;
    const std::uint64_t n = value.size();
    if (!writeAll(out, &header, sizeof(header))
        || !writeAll(out, &round, sizeof(round))
        || !writeAll(out, &progress.settledTotal, sizeof(progress.settledTotal))
        || !writeAll(out, &progress.drawTotal, sizeof(progress.drawTotal))
        || !writeAll(out, &n, sizeof(n))
        || !writeAll(out, value.data(), value.size())
        || !writeAll(out, rank.data(), rank.size())
        || !writeAll(out, action.data(), action.size())) {
      throw std::runtime_error("could not write the round checkpoint");
    }
  }
  std::remove(target.c_str());
  if (std::rename(temporary.c_str(), target.c_str()) != 0) {
    throw std::runtime_error("could not publish the round checkpoint");
  }
}

Solution solve(const Geometry& geometry, const Board& root, bool verbose,
               const std::string& checkpointPath, int threadCount) {
  const auto start = std::chrono::steady_clock::now();
  Solution solution(geometry.total);
  const CheckpointHeader header = checkpointHeader(geometry, root.rows, root.columns);

  // Phase 1: discovery. Only the bitset is sized by the universe. A matching
  // bitset checkpoint replaces the whole phase.
  const bool discoveredFromCheckpoint = !checkpointPath.empty()
      && loadBitsetCheckpoint(checkpointPath, header, solution.reachable.mutableWords());
  if (discoveredFromCheckpoint) {
    solution.reachable.finalize();
    solution.states = solution.reachable.count();
    if (verbose) {
      std::cerr << "[chaos] discovery loaded from checkpoint states=" << solution.states
                << " seconds=" << secondsSince(start) << std::endl;
    }
  } else {
    std::vector<std::uint64_t> stack;
    const std::uint64_t rootIndex = canonicalIndex(geometry, root);
    solution.reachable.set(rootIndex);
    stack.push_back(rootIndex);
    EdgeList edges;
    Board board;
    std::uint64_t discovered = 0;
    while (!stack.empty()) {
      const std::uint64_t index = stack.back();
      stack.pop_back();
      ++discovered;
      decode(geometry, index, board);
      successors(geometry, board, edges);
      for (int e = 0; e < edges.count; ++e) {
        const Edge& edge = edges.values[e];
        if (edge.terminal != NOT_TERMINAL || solution.reachable.test(edge.next)) continue;
        solution.reachable.set(edge.next);
        stack.push_back(edge.next);
      }
      if (verbose && (discovered & 0xfffff) == 0) {
        std::cerr << "[chaos] discovered=" << discovered << " seconds=" << secondsSince(start) << std::endl;
      }
    }
    solution.reachable.finalize();
    solution.states = solution.reachable.count();
    if (verbose) {
      std::cerr << "[chaos] discovery complete states=" << solution.states
                << " bitsetMB=" << (solution.reachable.bytes() >> 20)
                << " seconds=" << secondsSince(start) << std::endl;
    }
    if (!checkpointPath.empty()) {
      writeBitsetCheckpoint(checkpointPath, header, solution.reachable.words());
    }
  }

  const std::uint64_t n = solution.states;
  solution.value.assign(n, VALUE_UNKNOWN);
  solution.rank.assign(n, 0);
  solution.action.assign(n, NO_ACTION);

  // Phases 2 and 3 in one: rank iteration with successors regenerated on
  // demand. Materialised successor lists cost about five bytes per edge plus
  // eight bytes per state for an ordinal-to-index table - tens of gigabytes
  // at 10^9 states - while regenerating an edge costs one canonicalIndex and
  // one rank probe. Each round sweeps only the still-unresolved states.
  //
  // Terminal edges are labelled from the mover's perspective (WIN means the
  // mover wins by playing it) and non-terminal children from the child's
  // mover's perspective, so a child value of LOSS is a win for the parent.
  //   win  at rank r : some move wins now (rank 0) or reaches a lost child of rank r-1
  //   loss at rank r : every move loses now or reaches a won child, max child rank r-1
  // A state whose children have all settled with no win on offer can only
  // draw; settling it immediately keeps later sweeps off it. Such draws are
  // final (settled children never change), their drawing action is chosen
  // against settled values, and they never drive the round counter, so
  // win/loss ranks and the maximum rank are exactly the classical ones.
  // Chunk the ordinal space for the sweep threads: word ranges holding
  // roughly equal numbers of states, dealt through an atomic cursor.
  const int threads = threadCount > 1 ? threadCount : 1;
  std::vector<std::uint64_t> chunkWord;
  {
    const std::uint64_t chunkTarget = static_cast<std::uint64_t>(threads) * 16;
    chunkWord.push_back(0);
    for (std::uint64_t chunk = 1; chunk < chunkTarget; ++chunk) {
      const std::uint64_t word =
          solution.reachable.sampleWordFor(chunk * n / chunkTarget);
      if (word > chunkWord.back()) chunkWord.push_back(word);
    }
    chunkWord.push_back(solution.reachable.wordCount());
  }
  std::vector<std::uint64_t> chunkBase(chunkWord.size() - 1);
  for (std::size_t chunk = 0; chunk + 1 < chunkWord.size(); ++chunk) {
    chunkBase[chunk] = chunkWord[chunk] < solution.reachable.wordCount()
        ? solution.reachable.rank(chunkWord[chunk] * 64)
        : n;
  }
  // Runs body(chunkIndex) across every chunk on the sweep threads and joins.
  const auto parallelChunks = [&](auto&& body) {
    if (threads == 1) {
      for (std::size_t chunk = 0; chunk + 1 < chunkWord.size(); ++chunk) body(chunk);
      return;
    }
    std::atomic<std::uint64_t> cursor{0};
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(threads));
    for (int t = 0; t < threads; ++t) {
      pool.emplace_back([&]() {
        for (;;) {
          const std::uint64_t chunk = cursor.fetch_add(1, std::memory_order_relaxed);
          if (chunk + 1 >= chunkWord.size()) return;
          body(static_cast<std::size_t>(chunk));
        }
      });
    }
    for (std::thread& worker : pool) worker.join();
  };

  RoundCheckpoint progress;
  if (!checkpointPath.empty()
      && loadRoundCheckpoint(checkpointPath, header, n, progress,
                             solution.value, solution.rank, solution.action)
      && verbose) {
    std::cerr << "[chaos] resumed after round=" << progress.round
              << " total=" << (progress.settledTotal + progress.drawTotal) << std::endl;
  }
  std::uint64_t settledTotal = progress.settledTotal;
  std::uint64_t drawTotal = progress.drawTotal;
  int round = progress.round + 1;
  {
    while (true) {
      std::atomic<std::uint64_t> settledShared{0};
      std::atomic<std::uint64_t> drawShared{0};
      parallelChunks([&](std::size_t chunk) {
        EdgeList edges;
        Board board;
        std::uint64_t localSettled = 0;
        std::uint64_t localDraws = 0;
        std::uint64_t ordinal = chunkBase[chunk];
        solution.reachable.forEachInWordRange(
            chunkWord[chunk], chunkWord[chunk + 1], [&](std::uint64_t index) {
          const std::uint64_t at = ordinal++;
          // Own slot: no thread but this one writes it, so a plain read is
          // race-free; other threads read it as a child through an acquire.
          if (solution.value[at] != VALUE_UNKNOWN) return;
          decode(geometry, index, board);
          successors(geometry, board, edges);
          if (edges.count == 0) return;

          bool win = false;
          std::uint8_t winAction = NO_ACTION;
          bool allLoss = true;
          int maxChildRank = -1;
          std::uint8_t lossAction = NO_ACTION;
          bool anyUnknown = false;
          bool anyWin = false;
          std::uint8_t drawAction = NO_ACTION;
          for (int e = 0; e < edges.count; ++e) {
            const Edge& edge = edges.values[e];
            int forMover;
            int childRank;
            if (edge.terminal != NOT_TERMINAL) {
              forMover = edge.terminal;   // already mover-relative
              childRank = 0;
            } else {
              const std::uint64_t child = solution.reachable.rank(edge.next);
              // Acquire pairs with the release below: a settled value seen
              // here guarantees the matching rank is visible too. A stale
              // UNKNOWN only defers the parent to the next round.
              const std::uint8_t packed = std::atomic_ref<const std::uint8_t>(
                  solution.value[child]).load(std::memory_order_acquire);
              if (packed == VALUE_UNKNOWN) {
                forMover = NOT_TERMINAL;
                childRank = 0;
              } else {
                const int fromChild = unpackValue(packed);
                forMover = fromChild == DRAW ? DRAW : -fromChild;
                childRank = std::atomic_ref<const std::uint8_t>(
                    solution.rank[child]).load(std::memory_order_relaxed);
              }
            }
            const std::uint8_t encoded =
                static_cast<std::uint8_t>(edge.action | (edge.column << 2));
            if (forMover == WIN && childRank == round - 1) {
              win = true;
              winAction = encoded;
              break;
            }
            if (forMover != LOSS) {
              allLoss = false;
            } else if (childRank > maxChildRank) {
              maxChildRank = childRank;
              lossAction = encoded;
            }
            if (forMover == NOT_TERMINAL) anyUnknown = true;
            else if (forMover == WIN) anyWin = true;
            else if (forMover == DRAW && drawAction == NO_ACTION) drawAction = encoded;
          }
          const auto publish = [&](int outcome, std::uint8_t stateRank, std::uint8_t action) {
            solution.action[at] = action;
            std::atomic_ref<std::uint8_t>(solution.rank[at])
                .store(stateRank, std::memory_order_relaxed);
            std::atomic_ref<std::uint8_t>(solution.value[at])
                .store(packValue(outcome), std::memory_order_release);
          };
          if (win) {
            publish(WIN, static_cast<std::uint8_t>(round), winAction);
            ++localSettled;
          } else if (allLoss && maxChildRank == round - 1) {
            publish(LOSS, static_cast<std::uint8_t>(round), lossAction);
            ++localSettled;
          } else if (!anyUnknown && !anyWin && !allLoss) {
            if (drawAction == NO_ACTION) {
              throw std::runtime_error("a drawn position has no drawing action");
            }
            publish(DRAW, 0, drawAction);
            ++localDraws;
          }
        });
        settledShared.fetch_add(localSettled, std::memory_order_relaxed);
        drawShared.fetch_add(localDraws, std::memory_order_relaxed);
      });
      const std::uint64_t settled = settledShared.load(std::memory_order_relaxed);
      const std::uint64_t drawSettled = drawShared.load(std::memory_order_relaxed);
      settledTotal += settled;
      drawTotal += drawSettled;
      if (verbose) {
        std::cerr << "[chaos] round=" << round << " settled=" << settled
                  << " draws=" << drawSettled
                  << " total=" << (settledTotal + drawTotal)
                  << " seconds=" << secondsSince(start) << std::endl;
      }
      if (settled == 0) break;
      if (!checkpointPath.empty()) {
        progress.round = round;
        progress.settledTotal = settledTotal;
        progress.drawTotal = drawTotal;
        writeRoundCheckpoint(checkpointPath, header, progress,
                             solution.value, solution.rank, solution.action);
      }
      ++round;
      if (round > 250) throw std::runtime_error("rank iteration did not converge");
    }
  }
  solution.maximumRank = round - 1;

  // Phase 4: everything still unresolved sits on a cycle and is a draw. Mark
  // them all first, then choose their drawing actions against final values;
  // the draws settled during the sweeps already carry final-value actions.
  {
    std::uint64_t cycleDraws = 0;
    for (std::uint64_t at = 0; at < n; ++at) {
      if (solution.value[at] == VALUE_UNKNOWN) {
        solution.value[at] = packValue(DRAW);
        solution.rank[at] = 0;
        solution.action[at] = NO_ACTION;
        ++cycleDraws;
      }
    }
    if (cycleDraws != 0) {
      // Values are final here, so the threads only race on their own action
      // slots and everything they read is settled.
      parallelChunks([&](std::size_t chunk) {
        EdgeList edges;
        Board board;
        std::uint64_t ordinal = chunkBase[chunk];
        solution.reachable.forEachInWordRange(
            chunkWord[chunk], chunkWord[chunk + 1], [&](std::uint64_t index) {
          const std::uint64_t at = ordinal++;
          if (solution.value[at] != packValue(DRAW) || solution.action[at] != NO_ACTION) {
            return;
          }
          decode(geometry, index, board);
          successors(geometry, board, edges);
          for (int e = 0; e < edges.count; ++e) {
            const Edge& edge = edges.values[e];
            const bool draws = edge.terminal == DRAW
                || (edge.terminal == NOT_TERMINAL
                    && solution.value[solution.reachable.rank(edge.next)] == packValue(DRAW));
            if (draws) {
              solution.action[at] =
                  static_cast<std::uint8_t>(edge.action | (edge.column << 2));
              break;
            }
          }
          if (solution.action[at] == NO_ACTION && edges.count != 0) {
            throw std::runtime_error("a drawn position has no drawing action");
          }
        });
      });
    }
  }
  for (std::uint64_t at = 0; at < n; ++at) {
    const std::uint8_t packed = solution.value[at];
    if (packed == packValue(WIN)) ++solution.wins;
    else if (packed == packValue(LOSS)) ++solution.losses;
    else ++solution.draws;
  }

  solution.rootOrdinal = solution.reachable.rank(canonicalIndex(geometry, root));
  solution.rootValue = unpackValue(solution.value[solution.rootOrdinal]);
  if (verbose) std::cerr << "[chaos] solved seconds=" << secondsSince(start) << std::endl;
  return solution;
}

// ---------------------------------------------------------------------------
// Certificates
// ---------------------------------------------------------------------------

struct PolicyRecord {
  std::uint64_t mover;
  std::uint64_t opponent;
  std::uint8_t rows;
  std::uint8_t columns;
  std::uint8_t action;
  std::uint8_t column;
  std::int8_t value;

  bool operator<(const PolicyRecord& other) const {
    if (rows != other.rows) return rows < other.rows;
    if (columns != other.columns) return columns < other.columns;
    if (mover != other.mover) return mover < other.mover;
    return opponent < other.opponent;
  }
};

void packBoard(const Board& board, std::uint64_t& mover, std::uint64_t& opponent) {
  const int stride = board.rows + 1;
  mover = 0;
  opponent = 0;
  for (int column = 0; column < board.columns; ++column) {
    for (int row = 0; row < board.rows; ++row) {
      const std::uint8_t cell = board.cells[row][column];
      if (cell == 0) continue;
      const std::uint64_t bit = std::uint64_t{1} << (column * stride + row);
      if (cell == 1) mover |= bit;
      else opponent |= bit;
    }
  }
}

struct ClosureStats {
  std::uint64_t states = 0;
  std::uint64_t aiStates = 0;
  std::uint64_t opponentStates = 0;
  std::uint64_t terminalAiWins = 0;
  std::uint64_t terminalAiLosses = 0;
  std::uint64_t terminalDraws = 0;
  std::uint64_t rankProgressChecked = 0;
  std::uint64_t drawSafetyChecked = 0;
  int rootValue = 0;
};

// Walks the closure a starting role reaches: the AI's one action at each of its
// own turns, every legal opponent reply. Verifies finite progress for wins and
// region safety for draws along the way, and optionally emits records.
ClosureStats closure(const Geometry& geometry, const Solution& solution, int role,
                     std::vector<PolicyRecord>* records) {
  ClosureStats stats;
  const std::uint64_t n = solution.states;
  const int moverValue = solution.rootValue;
  stats.rootValue = role == 1 ? moverValue : -moverValue;

  // Visit key: ordinal * 2 + aiTurn. A position can be reached with either side
  // to move after an odd chain of transformations. One bit per key.
  std::vector<std::uint64_t> visited((n * 2 + 63) / 64, 0);
  const auto testVisited = [&visited](std::uint64_t key) {
    return ((visited[key >> 6] >> (key & 63)) & 1) != 0;
  };
  const auto setVisited = [&visited](std::uint64_t key) {
    visited[key >> 6] |= std::uint64_t{1} << (key & 63);
  };
  // Per-role draw-action overrides: each role reaches a different closure,
  // and only drawn positions inside it ever get an override.
  std::unordered_map<std::uint64_t, std::uint8_t> chosen;
  std::vector<std::uint64_t> stack;
  const bool rootAi = role == 1;
  setVisited(solution.rootOrdinal * 2 + (rootAi ? 1 : 0));
  stack.push_back(solution.rootOrdinal * 2 + (rootAi ? 1 : 0));

  EdgeList edges;
  Board board;
  while (!stack.empty()) {
    const std::uint64_t key = stack.back();
    stack.pop_back();
    const std::uint64_t ordinal = key >> 1;
    const bool aiTurn = (key & 1) != 0;
    ++stats.states;

    decode(geometry, solution.reachable.select(ordinal), board);
    successors(geometry, board, edges);
    if (edges.count == 0) {
      ++stats.terminalDraws;
      continue;
    }

    auto follow = [&](const Edge& edge) {
      if (edge.terminal != NOT_TERMINAL) {
        if (edge.terminal == DRAW) ++stats.terminalDraws;
        else if ((edge.terminal == WIN) == aiTurn) ++stats.terminalAiWins;
        else ++stats.terminalAiLosses;
        return;
      }
      const std::uint64_t childKey = solution.reachable.rank(edge.next) * 2 + (aiTurn ? 0 : 1);
      if (testVisited(childKey)) return;
      setVisited(childKey);
      stack.push_back(childKey);
    };

    if (!aiTurn) {
      ++stats.opponentStates;
      for (int e = 0; e < edges.count; ++e) follow(edges.values[e]);
      continue;
    }

    ++stats.aiStates;
    const std::uint8_t packed = solution.value[ordinal];
    // Drawn positions: prefer a drawing action into an already-visited state.
    if (packed == packValue(DRAW) && chosen.find(ordinal) == chosen.end()) {
      int preferred = -1;
      int fallback = -1;
      for (int e = 0; e < edges.count; ++e) {
        const Edge& edge = edges.values[e];
        const bool safe = edge.terminal == DRAW
            || (edge.terminal == NOT_TERMINAL
                && solution.value[solution.reachable.rank(edge.next)] == packValue(DRAW));
        if (!safe) continue;
        if (fallback < 0) fallback = e;
        const bool known = edge.terminal == DRAW
            || testVisited(solution.reachable.rank(edge.next) * 2);
        if (known) {
          preferred = e;
          break;
        }
      }
      const int selected = preferred >= 0 ? preferred : fallback;
      if (selected >= 0) {
        chosen[ordinal] = static_cast<std::uint8_t>(
            edges.values[selected].action | (edges.values[selected].column << 2));
      }
    }
    const auto override = chosen.find(ordinal);
    const std::uint8_t action =
        override != chosen.end() ? override->second : solution.action[ordinal];
    if (action == NO_ACTION) throw std::runtime_error("closure reached an unsolved AI state");

    if (records != nullptr) {
      std::uint64_t mover = 0;
      std::uint64_t opponent = 0;
      packBoard(board, mover, opponent);
      records->push_back({mover, opponent,
                          static_cast<std::uint8_t>(board.rows), static_cast<std::uint8_t>(board.columns),
                          static_cast<std::uint8_t>(action & 3), static_cast<std::uint8_t>(action >> 2),
                          static_cast<std::int8_t>(unpackValue(packed))});
    }

    bool matched = false;
    for (int e = 0; e < edges.count; ++e) {
      const Edge& edge = edges.values[e];
      if (edge.action != (action & 3) || ((action & 3) == ACTION_DROP && edge.column != (action >> 2))) {
        continue;
      }
      if (packed == packValue(WIN)) {
        // Finite progress: win outright, or hand over a lost position of
        // strictly smaller rank.
        if (edge.terminal != WIN) {
          if (edge.terminal != NOT_TERMINAL) throw std::runtime_error("winning action is not a win");
          const std::uint64_t c = solution.reachable.rank(edge.next);
          if (solution.value[c] != packValue(LOSS) || solution.rank[c] >= solution.rank[ordinal]) {
            throw std::runtime_error("winning action does not reduce the proof rank");
          }
        }
        ++stats.rankProgressChecked;
      } else if (packed == packValue(DRAW)) {
        const bool safe = edge.terminal == DRAW
            || (edge.terminal == NOT_TERMINAL
                && solution.value[solution.reachable.rank(edge.next)] == packValue(DRAW));
        if (!safe) throw std::runtime_error("drawing action leaves the drawn region");
        ++stats.drawSafetyChecked;
      }
      follow(edge);
      matched = true;
      break;
    }
    if (!matched) throw std::runtime_error("stored action is not legal in its own position");
  }
  return stats;
}

void writePolicy(const std::string& path, const Geometry& geometry, int rows, int columns, int role,
                 const ClosureStats& stats, std::vector<PolicyRecord>& records) {
  std::sort(records.begin(), records.end());
  for (std::size_t index = 1; index < records.size(); ++index) {
    if (!(records[index - 1] < records[index])) throw std::runtime_error("duplicate policy record");
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("could not open the policy output");
  auto put = [&output](std::uint8_t value) { output.put(static_cast<char>(value)); };
  auto put32 = [&put](std::uint32_t value) { for (int s = 0; s < 4; ++s) put((value >> (s * 8)) & 0xff); };
  auto put64 = [&put](std::uint64_t value) { for (int s = 0; s < 8; ++s) put((value >> (s * 8)) & 0xff); };
  const char magic[8] = {'C', '4', 'C', 'F', 'U', 'L', '1', '\0'};
  output.write(magic, 8);
  put(1);
  put(static_cast<std::uint8_t>(rows));
  put(static_cast<std::uint8_t>(columns));
  put(static_cast<std::uint8_t>(geometry.connect));
  put(static_cast<std::uint8_t>(role));
  put(static_cast<std::uint8_t>(static_cast<std::int8_t>(stats.rootValue)));
  put(24);
  put(0);
  put32(static_cast<std::uint32_t>(records.size()));
  put32(static_cast<std::uint32_t>(stats.states));
  for (const PolicyRecord& record : records) {
    put64(record.mover);
    put64(record.opponent);
    put(record.rows);
    put(record.columns);
    put(record.action);
    put(record.column);
    put(static_cast<std::uint8_t>(record.value));
    put(0);
    put(0);
    put(0);
  }
  if (!output) throw std::runtime_error("could not write the complete policy");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int rows = 4;
    int columns = 4;
    int connect = 4;
    bool verbose = false;
    bool withClosure = false;
    std::string policyPrefix;
    std::string checkpointPath;
    int threadCount = 1;
    for (int index = 1; index < argc; ++index) {
      const std::string name = argv[index];
      auto next = [&]() -> std::string {
        if (index + 1 >= argc) throw std::runtime_error(name + " requires a value");
        return argv[++index];
      };
      if (name == "--rows") rows = std::stoi(next());
      else if (name == "--columns") columns = std::stoi(next());
      else if (name == "--connect") connect = std::stoi(next());
      else if (name == "--verbose") verbose = true;
      else if (name == "--closure") withClosure = true;
      else if (name == "--emit-policy") { policyPrefix = next(); withClosure = true; }
      else if (name == "--checkpoint") checkpointPath = next();
      else if (name == "--threads") threadCount = std::stoi(next());
      else throw std::runtime_error("unknown argument: " + name);
    }

    const Geometry geometry = makeGeometry(rows, columns, connect);
    Board root;
    root.clear(rows, columns);
    const auto start = std::chrono::steady_clock::now();
    const Solution solution = solve(geometry, root, verbose, checkpointPath, threadCount);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    std::cout << "{\"format\":\"connect4-chaos-exact-solution-v1\""
              << ",\"rows\":" << rows << ",\"columns\":" << columns << ",\"connect\":" << connect
              << ",\"indexSpace\":" << geometry.total
              << ",\"states\":" << solution.states
              << ",\"wins\":" << solution.wins << ",\"draws\":" << solution.draws
              << ",\"losses\":" << solution.losses
              << ",\"maximumRank\":" << solution.maximumRank
              << ",\"rootValue\":" << solution.rootValue
              << ",\"elapsedMs\":" << elapsed << "}\n";

    if (withClosure) {
      for (const int role : {1, 2}) {
        std::vector<PolicyRecord> records;
        const ClosureStats stats = closure(geometry, solution, role,
                                           policyPrefix.empty() ? nullptr : &records);
        if (!policyPrefix.empty()) {
          writePolicy(policyPrefix + "-role" + std::to_string(role) + ".bin",
                      geometry, rows, columns, role, stats, records);
        }
        std::cout << "{\"format\":\"connect4-chaos-closure-v1\""
                  << ",\"rows\":" << rows << ",\"columns\":" << columns << ",\"connect\":" << connect
                  << ",\"role\":" << role << ",\"rootValue\":" << stats.rootValue
                  << ",\"closureStates\":" << stats.states
                  << ",\"aiStates\":" << stats.aiStates
                  << ",\"opponentStates\":" << stats.opponentStates
                  << ",\"terminalAiWins\":" << stats.terminalAiWins
                  << ",\"terminalAiLosses\":" << stats.terminalAiLosses
                  << ",\"terminalDraws\":" << stats.terminalDraws
                  << ",\"rankProgressChecked\":" << stats.rankProgressChecked
                  << ",\"drawSafetyChecked\":" << stats.drawSafetyChecked << "}\n";
      }
    }
    if (!checkpointPath.empty()) {
      std::remove((checkpointPath + ".bitset").c_str());
      std::remove((checkpointPath + ".bitset.tmp").c_str());
      std::remove((checkpointPath + ".round").c_str());
      std::remove((checkpointPath + ".round.tmp").c_str());
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
