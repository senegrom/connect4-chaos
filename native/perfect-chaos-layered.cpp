// Layered exact solver for Chaos Mode boards too large to index whole.
//
// Drops add a piece; flips and rotations do not. Every cycle therefore lives
// inside one piece-count layer, and the layer graph is a DAG: layer k feeds
// only layer k+1 (drops) and itself (transformations). Solving proceeds
// backward from the fullest layer to the empty board, holding at most two
// adjacent layers in memory, so a board like 5x6 connect 4 - whose whole
// reachable graph is far past this machine's RAM and past a 32-bit global
// ordinal - fits in roughly ten gigabytes: each layer gets its own compact
// index and its own sub-2^32 ordinal space.
//
// Within a layer the value fixpoint is the same ranked iteration the
// monolithic solver (perfect-chaos-complete.cpp) uses, with drop children
// acting as settled terminals. Local ranks are not comparable to the
// monolithic solver's global ranks, so this program reports no maximum rank;
// states, wins, draws, losses and the root value are identical by
// construction and are cross-checked against the recorded monolithic results
// on every board small enough to solve both ways.
//
// Layer index: a board with piece count k in a given orientation is ranked by
// its column-height composition (lexicographic rank among all height vectors
// with that sum) times 2^k, plus its colour bits (mover bits of each column,
// bottom to top, columns left to right). A board and its transpose share a
// layer as two consecutive blocks, and horizontal mirroring is canonicalised
// by taking the smaller slot, exactly as in the monolithic index.
//
// Artifacts, one pair per layer in the output directory:
//   layer-<k>.bits    reachable-slot bitset, written as discovery closes k
//   layer-<k>.values  solved values by layer ordinal, written as k resolves
// Both double as checkpoints: a restarted run resumes at the first missing
// file, so a reboot costs at most one layer. On success the run leaves them
// in place and prints one JSON solution line to stdout.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int WIN = 1;
constexpr int DRAW = 0;
constexpr int LOSS = -1;
constexpr int NOT_TERMINAL = 2;
constexpr std::uint8_t VALUE_UNKNOWN = 3;
constexpr int MAX_SIDE = 8;

constexpr int ACTION_DROP = 0;
constexpr int ACTION_FLIP = 1;
constexpr int ACTION_ROTATE_CW = 2;
constexpr int ACTION_ROTATE_CCW = 3;

// ---------------------------------------------------------------------------
// Positions as bitboards: bit column * (rows + 1) + row, one guard bit per
// column so run detection cannot wrap between columns. Gravity keeps every
// column's pieces contiguous from the bottom, so a column's height is the
// size of its occupied segment and transformations move whole segments.
// The geometry caps boards at 7x7, which fits 64 bits with the guards.
// ---------------------------------------------------------------------------

struct Masks {
  std::uint64_t mover = 0;
  std::uint64_t opponent = 0;
};

// reversed[h][bits]: `bits` reversed within h bits, for flipping stacks.
struct ReverseTable {
  std::array<std::array<std::uint8_t, 1 << (MAX_SIDE - 1)>, MAX_SIDE> table{};
  ReverseTable() {
    for (int width = 1; width < MAX_SIDE; ++width) {
      for (int bits = 0; bits < (1 << width); ++bits) {
        int reversed = 0;
        for (int bit = 0; bit < width; ++bit) {
          if ((bits >> bit) & 1) reversed |= 1 << (width - 1 - bit);
        }
        table[width][bits] = static_cast<std::uint8_t>(reversed);
      }
    }
  }
};
const ReverseTable REVERSE;

bool maskHasLine(std::uint64_t mask, int rows, int connect) {
  const int stride = rows + 1;
  const int shifts[4] = {1, stride, stride + 1, stride - 1};
  for (const int shift : shifts) {
    std::uint64_t run = mask;
    for (int step = 1; step < connect && run != 0; ++step) {
      run &= mask >> (shift * step);
    }
    if (run != 0) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Layer geometry: per piece count, composition rank times colour bits
// ---------------------------------------------------------------------------

struct BlockShape {
  int rows = 0;
  int columns = 0;
  // comps[c][s]: height vectors for c columns, each 0..rows, summing to s.
  std::array<std::array<std::uint64_t, MAX_SIDE * MAX_SIDE + 1>, MAX_SIDE + 1> comps{};

  // prefix[c][s]: comps[c][0] + ... + comps[c][s], with prefix[c][-1] read
  // as zero through the +1 offset, so a column's rank contribution is one
  // subtraction instead of a loop over its height.
  std::array<std::array<std::uint64_t, MAX_SIDE * MAX_SIDE + 2>, MAX_SIDE + 1> prefix{};

  void build() {
    for (auto& row : comps) row.fill(0);
    comps[0][0] = 1;
    for (int c = 1; c <= columns; ++c) {
      for (int s = 0; s <= c * rows; ++s) {
        std::uint64_t total = 0;
        for (int h = 0; h <= rows && h <= s; ++h) total += comps[c - 1][s - h];
        comps[c][s] = total;
      }
    }
    for (auto& row : prefix) row.fill(0);
    for (int c = 0; c <= columns; ++c) {
      std::uint64_t running = 0;
      for (int s = 0; s <= MAX_SIDE * MAX_SIDE; ++s) {
        running += comps[c][s];
        prefix[c][s + 1] = running;
      }
    }
  }

  std::uint64_t layerSlots(int pieces) const {
    if (pieces < 0 || pieces > rows * columns) return 0;
    return comps[columns][pieces] << pieces;
  }
};

struct LayerGeometry {
  int connect = 0;
  int cellCount = 0;
  std::array<BlockShape, 2> blocks{};
  int blockCount = 0;

  int blockIndexFor(int rows, int columns) const {
    for (int index = 0; index < blockCount; ++index) {
      if (blocks[index].rows == rows && blocks[index].columns == columns) return index;
    }
    throw std::runtime_error("board shape is outside the layered index space");
  }

  std::uint64_t blockOffset(int block, int pieces) const {
    std::uint64_t offset = 0;
    for (int index = 0; index < block; ++index) offset += blocks[index].layerSlots(pieces);
    return offset;
  }

  std::uint64_t layerSlots(int pieces) const {
    std::uint64_t total = 0;
    for (int index = 0; index < blockCount; ++index) total += blocks[index].layerSlots(pieces);
    return total;
  }
};

LayerGeometry makeLayerGeometry(int rows, int columns, int connect) {
  if (rows < 1 || rows >= MAX_SIDE || columns < 1 || columns >= MAX_SIDE) {
    throw std::range_error("board dimensions are out of range");
  }
  if (connect < 1 || connect > std::max(rows, columns)) {
    throw std::range_error("connect length does not fit the board");
  }
  LayerGeometry geometry;
  geometry.connect = connect;
  geometry.cellCount = rows * columns;
  auto add = [&geometry](int r, int c) {
    BlockShape& block = geometry.blocks[geometry.blockCount++];
    block.rows = r;
    block.columns = c;
    block.build();
  };
  add(rows, columns);
  if (rows != columns) add(columns, rows);
  return geometry;
}

// The encode/decode currency is the height vector plus the colour word: each
// column's mover bits packed bottom-up, columns left to right.

std::uint64_t compositionRankOf(const BlockShape& block, const int* heights, int pieces) {
  // Sum of comps[left][remaining - lower] for lower in [0, height) equals the
  // prefix-sum difference over [remaining - height + 1, remaining], clamped to
  // the valid range [0, left * rows].
  std::uint64_t rank = 0;
  int remaining = pieces;
  for (int column = 0; column < block.columns; ++column) {
    const int columnsLeft = block.columns - column - 1;
    const int cap = columnsLeft * block.rows;
    int high = remaining < cap ? remaining : cap;
    int low = remaining - heights[column] + 1;
    if (low < 0) low = 0;
    if (high >= low) {
      rank += block.prefix[columnsLeft][high + 1] - block.prefix[columnsLeft][low];
    }
    remaining -= heights[column];
  }
  return rank;
}

// Decodes a layer slot into masks and heights; returns the block index.
int decodeLayerMasks(const LayerGeometry& geometry, int pieces, std::uint64_t slot,
                     Masks& masks, int* heights) {
  int blockIndex = 0;
  while (blockIndex + 1 < geometry.blockCount
         && slot >= geometry.blockOffset(blockIndex + 1, pieces)) {
    ++blockIndex;
  }
  const BlockShape& block = geometry.blocks[blockIndex];
  slot -= geometry.blockOffset(blockIndex, pieces);

  std::uint64_t compositionRank = slot >> pieces;
  std::uint64_t colours = pieces == 0 ? 0 : (slot & ((std::uint64_t{1} << pieces) - 1));
  const int stride = block.rows + 1;
  masks.mover = 0;
  masks.opponent = 0;
  int remaining = pieces;
  for (int column = 0; column < block.columns; ++column) {
    const int columnsLeft = block.columns - column - 1;
    int height = 0;
    for (; height <= block.rows; ++height) {
      const int rest = remaining - height;
      if (rest < 0 || rest > columnsLeft * block.rows) continue;
      const std::uint64_t below = block.comps[columnsLeft][rest];
      if (compositionRank < below) break;
      compositionRank -= below;
    }
    heights[column] = height;
    remaining -= height;
    const std::uint64_t occupied = (std::uint64_t{1} << height) - 1;
    const std::uint64_t segment = colours & occupied;
    colours >>= height;
    masks.mover |= segment << (column * stride);
    masks.opponent |= (occupied ^ segment) << (column * stride);
  }
  return blockIndex;
}

// Canonical slot straight from masks: pick the smaller orientation by
// comparing height vectors (leftmost column most significant, smaller height
// first) and then colour segments from the rightmost column's top, and encode
// only the winner. Matches min(encode(board), encode(mirror(board))).
std::uint64_t canonicalLayerSlotMasks(const LayerGeometry& geometry, int blockIndex,
                                      const Masks& masks, const int* heights, int pieces) {
  const BlockShape& block = geometry.blocks[blockIndex];
  const int columns = block.columns;
  const int stride = block.rows + 1;

  int order = 0;   // negative: direct orientation first; positive: mirror
  for (int column = 0; order == 0 && column < columns; ++column) {
    const int direct = heights[column];
    const int mirrored = heights[columns - 1 - column];
    if (direct != mirrored) order = direct < mirrored ? -1 : 1;
  }
  if (order == 0) {
    for (int column = columns - 1; order == 0 && column >= 0; --column) {
      const std::uint64_t width = (std::uint64_t{1} << heights[column]) - 1;
      const std::uint64_t direct = (masks.mover >> (column * stride)) & width;
      const std::uint64_t mirrored =
          (masks.mover >> ((columns - 1 - column) * stride)) & width;
      if (direct != mirrored) order = direct > mirrored ? 1 : -1;
    }
  }

  std::uint64_t colours = 0;
  std::uint64_t rank = 0;
  if (order <= 0) {
    int offset = 0;
    for (int column = 0; column < columns; ++column) {
      const std::uint64_t width = (std::uint64_t{1} << heights[column]) - 1;
      colours |= ((masks.mover >> (column * stride)) & width) << offset;
      offset += heights[column];
    }
    rank = compositionRankOf(block, heights, pieces);
  } else {
    int reversedHeights[MAX_SIDE];
    int offset = 0;
    for (int column = 0; column < columns; ++column) {
      const int source = columns - 1 - column;
      reversedHeights[column] = heights[source];
      const std::uint64_t width = (std::uint64_t{1} << heights[source]) - 1;
      colours |= ((masks.mover >> (source * stride)) & width) << offset;
      offset += heights[source];
    }
    rank = compositionRankOf(block, reversedHeights, pieces);
  }
  return geometry.blockOffset(blockIndex, pieces) + (rank << pieces) + colours;
}

// ---------------------------------------------------------------------------
// Edges: drops go to layer k+1, transformations stay in layer k
// ---------------------------------------------------------------------------

struct LayerEdge {
  std::int8_t terminal = NOT_TERMINAL;   // WIN / DRAW / LOSS for the mover
  bool sameLayer = false;                // transform edge when non-terminal
  std::uint64_t slot = 0;                // canonical layer slot of the child
};

struct LayerEdgeList {
  std::array<LayerEdge, MAX_SIDE + 3> values{};
  int count = 0;
};

// Same terminal labelling as the monolithic successors(): a transformation
// that completes lines for both players loses for the mover, and children
// swap roles so the next player becomes the mover - free on masks, the
// child's mover IS the parent's opponent mask. States never already contain
// a line and are never full, which lets the drop win check read the whole
// grown mask and lets transformations skip the full-board check.
void layerSuccessors(const LayerGeometry& geometry, int blockIndex, const Masks& masks,
                     const int* heights, int pieces, LayerEdgeList& edges) {
  edges.count = 0;
  const BlockShape& block = geometry.blocks[blockIndex];
  const int rows = block.rows;
  const int columns = block.columns;
  const int stride = rows + 1;

  const auto emitTerminal = [&edges](int terminal) {
    LayerEdge edge;
    edge.terminal = static_cast<std::int8_t>(terminal);
    edges.values[edges.count++] = edge;
  };

  for (int column = 0; column < columns; ++column) {
    const int height = heights[column];
    if (height >= rows) continue;
    const std::uint64_t grown =
        masks.mover | (std::uint64_t{1} << (column * stride + height));
    if (maskHasLine(grown, rows, geometry.connect)) {
      emitTerminal(WIN);
      continue;
    }
    if (pieces + 1 == geometry.cellCount) {
      emitTerminal(DRAW);
      continue;
    }
    int childHeights[MAX_SIDE];
    for (int c = 0; c < columns; ++c) childHeights[c] = heights[c];
    ++childHeights[column];
    const Masks child{masks.opponent, grown};
    LayerEdge edge;
    edge.sameLayer = false;
    edge.slot = canonicalLayerSlotMasks(geometry, blockIndex, child, childHeights, pieces + 1);
    edges.values[edges.count++] = edge;
  }

  const auto settleTransform = [&](const Masks& next, int nextBlock,
                                   const int* nextHeights, int nextRows) {
    const bool moverLine = maskHasLine(next.mover, nextRows, geometry.connect);
    const bool opponentLine = maskHasLine(next.opponent, nextRows, geometry.connect);
    if (moverLine || opponentLine) {
      emitTerminal(moverLine && !opponentLine ? WIN : (moverLine ? LOSS : LOSS));
      return;
    }
    const Masks child{next.opponent, next.mover};
    LayerEdge edge;
    edge.sameLayer = true;
    edge.slot = canonicalLayerSlotMasks(geometry, nextBlock, child, nextHeights, pieces);
    edges.values[edges.count++] = edge;
  };

  {
    // Flip: every column's stack reverses in place.
    Masks flipped;
    for (int column = 0; column < columns; ++column) {
      const int height = heights[column];
      const int base = column * stride;
      const std::uint64_t occupied = (std::uint64_t{1} << height) - 1;
      const std::uint64_t segment = (masks.mover >> base) & occupied;
      const std::uint64_t reversed = REVERSE.table[height][segment];
      flipped.mover |= reversed << base;
      flipped.opponent |= (occupied ^ reversed) << base;
    }
    settleTransform(flipped, blockIndex, heights, rows);
  }

  const int transposedBlock = geometry.blockCount == 1 ? 0 : 1 - blockIndex;
  const int targetStride = columns + 1;   // the transposed block's rows
  int rotatedHeights[MAX_SIDE];

  {
    // Clockwise: target column tc collects source row tc, scanning source
    // columns right to left; the collection order is the gravity order.
    Masks rotated;
    for (int targetColumn = 0; targetColumn < rows; ++targetColumn) {
      int height = 0;
      for (int sourceColumn = columns - 1; sourceColumn >= 0; --sourceColumn) {
        if (heights[sourceColumn] <= targetColumn) continue;
        const std::uint64_t bit =
            std::uint64_t{1} << (targetColumn * targetStride + height);
        if ((masks.mover >> (sourceColumn * stride + targetColumn)) & 1) rotated.mover |= bit;
        else rotated.opponent |= bit;
        ++height;
      }
      rotatedHeights[targetColumn] = height;
    }
    settleTransform(rotated, transposedBlock, rotatedHeights, columns);
  }

  {
    // Counter-clockwise: target column tc collects source row rows-1-tc,
    // scanning source columns left to right.
    Masks rotated;
    for (int targetColumn = 0; targetColumn < rows; ++targetColumn) {
      const int sourceRow = rows - 1 - targetColumn;
      int height = 0;
      for (int sourceColumn = 0; sourceColumn < columns; ++sourceColumn) {
        if (heights[sourceColumn] <= sourceRow) continue;
        const std::uint64_t bit =
            std::uint64_t{1} << (targetColumn * targetStride + height);
        if ((masks.mover >> (sourceColumn * stride + sourceRow)) & 1) rotated.mover |= bit;
        else rotated.opponent |= bit;
        ++height;
      }
      rotatedHeights[targetColumn] = height;
    }
    settleTransform(rotated, transposedBlock, rotatedHeights, columns);
  }
}

// ---------------------------------------------------------------------------
// Per-layer rank bitset
// ---------------------------------------------------------------------------

class LayerBits {
 public:
  explicit LayerBits(std::uint64_t slots)
      : words_((slots + 63) / 64 + (slots == 0 ? 1 : 0), 0), slots_(slots) {}
  LayerBits(LayerBits&&) = default;
  LayerBits& operator=(LayerBits&&) = default;

  bool test(std::uint64_t slot) const {
    return (words_[slot >> 6] >> (slot & 63)) & 1;
  }
  void set(std::uint64_t slot) {
    words_[slot >> 6] |= std::uint64_t{1} << (slot & 63);
  }

  // Thread-safe probe-and-mark; returns true when this call set the bit.
  bool atomicTestSet(std::uint64_t slot) {
    std::atomic_ref<std::uint64_t> word(words_[slot >> 6]);
    const std::uint64_t mask = std::uint64_t{1} << (slot & 63);
    return (word.fetch_or(mask, std::memory_order_relaxed) & mask) == 0;
  }
  void clearAll() { std::fill(words_.begin(), words_.end(), 0); }

  bool atomicTest(std::uint64_t slot) const {
    return (std::atomic_ref<const std::uint64_t>(words_[slot >> 6])
                .load(std::memory_order_relaxed) & (std::uint64_t{1} << (slot & 63))) != 0;
  }

  void finalize() {
    ranks_.assign(words_.size() + 1, 0);
    std::uint64_t running = 0;
    for (std::size_t word = 0; word < words_.size(); ++word) {
      ranks_[word] = static_cast<std::uint32_t>(running);
      running += static_cast<std::uint64_t>(__builtin_popcountll(words_[word]));
    }
    if (running > 0xffffffffull) {
      throw std::runtime_error("a layer exceeds the 32-bit ordinal directory");
    }
    ranks_[words_.size()] = static_cast<std::uint32_t>(running);
    count_ = running;
  }

  std::uint64_t rank(std::uint64_t slot) const {
    const std::uint64_t word = slot >> 6;
    const std::uint64_t below = words_[word] & ((std::uint64_t{1} << (slot & 63)) - 1);
    return static_cast<std::uint64_t>(ranks_[word])
        + static_cast<std::uint64_t>(__builtin_popcountll(below));
  }

  std::uint64_t count() const { return count_; }
  std::uint64_t wordCount() const { return words_.size(); }
  std::uint64_t rankAtWord(std::uint64_t word) const { return ranks_[word]; }

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

  template <typename Visit>
  void forEach(Visit&& visit) const {
    forEachInWordRange(0, words_.size(), visit);
  }

  std::vector<std::uint64_t>& mutableWords() { return words_; }
  const std::vector<std::uint64_t>& words() const { return words_; }

 private:
  std::vector<std::uint64_t> words_;
  std::vector<std::uint32_t> ranks_;
  std::uint64_t slots_ = 0;
  std::uint64_t count_ = 0;
};

// ---------------------------------------------------------------------------
// Chunked file I/O with layer headers
// ---------------------------------------------------------------------------

constexpr std::size_t IO_CHUNK = std::size_t{256} << 20;
constexpr char LAYER_MAGIC[8] = {'C', '4', 'L', 'A', 'Y', 'R', '1', '\0'};

struct LayerHeader {
  char magic[8];
  std::uint8_t rows;
  std::uint8_t columns;
  std::uint8_t connect;
  std::uint8_t kind;   // 0 bitset, 1 values
  std::uint32_t layer;
  std::uint64_t payload;   // words for a bitset, states for values
};

bool readExact(std::ifstream& in, void* target, std::size_t bytes) {
  char* cursor = static_cast<char*>(target);
  while (bytes > 0) {
    const std::size_t step = bytes < IO_CHUNK ? bytes : IO_CHUNK;
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
    const std::size_t step = bytes < IO_CHUNK ? bytes : IO_CHUNK;
    out.write(cursor, static_cast<std::streamsize>(step));
    if (!out) return false;
    cursor += step;
    bytes -= step;
  }
  out.flush();
  return static_cast<bool>(out);
}

void publishFile(const std::string& temporary, const std::string& target) {
  std::remove(target.c_str());
  if (std::rename(temporary.c_str(), target.c_str()) != 0) {
    throw std::runtime_error("could not publish " + target);
  }
}

LayerHeader headerFor(int rows, int columns, int connect, int kind, int layer,
                      std::uint64_t payload) {
  LayerHeader header{};
  std::memcpy(header.magic, LAYER_MAGIC, sizeof(header.magic));
  header.rows = static_cast<std::uint8_t>(rows);
  header.columns = static_cast<std::uint8_t>(columns);
  header.connect = static_cast<std::uint8_t>(connect);
  header.kind = static_cast<std::uint8_t>(kind);
  header.layer = static_cast<std::uint32_t>(layer);
  header.payload = payload;
  return header;
}

bool headerMatches(const LayerHeader& seen, const LayerHeader& want) {
  return std::memcmp(seen.magic, want.magic, sizeof(want.magic)) == 0
      && seen.rows == want.rows && seen.columns == want.columns
      && seen.connect == want.connect && seen.kind == want.kind
      && seen.layer == want.layer && seen.payload == want.payload;
}

std::string bitsPath(const std::string& directory, int layer) {
  return directory + "/layer-" + std::to_string(layer) + ".bits";
}
std::string valuesPath(const std::string& directory, int layer) {
  return directory + "/layer-" + std::to_string(layer) + ".values";
}

void writeLayerBits(const std::string& directory, int rows, int columns, int connect,
                    int layer, const LayerBits& bits) {
  const std::string target = bitsPath(directory, layer);
  const std::string temporary = target + ".tmp";
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("could not open " + temporary);
    const LayerHeader header = headerFor(rows, columns, connect, 0, layer, bits.wordCount());
    if (!writeAll(out, &header, sizeof(header))
        || !writeAll(out, bits.words().data(), bits.wordCount() * sizeof(std::uint64_t))) {
      throw std::runtime_error("could not write " + temporary);
    }
  }
  publishFile(temporary, target);
}

bool loadLayerBits(const std::string& directory, int rows, int columns, int connect,
                   int layer, LayerBits& bits) {
  std::ifstream in(bitsPath(directory, layer), std::ios::binary);
  if (!in) return false;
  LayerHeader seen{};
  const LayerHeader want = headerFor(rows, columns, connect, 0, layer, bits.wordCount());
  if (!readExact(in, &seen, sizeof(seen)) || !headerMatches(seen, want)) {
    std::cerr << "[layered] rejecting " << bitsPath(directory, layer) << std::endl;
    return false;
  }
  if (!readExact(in, bits.mutableWords().data(), bits.wordCount() * sizeof(std::uint64_t))) {
    std::cerr << "[layered] short read on " << bitsPath(directory, layer) << std::endl;
    return false;
  }
  bits.finalize();
  return true;
}

void writeLayerValues(const std::string& directory, int rows, int columns, int connect,
                      int layer, const std::vector<std::uint8_t>& values) {
  const std::string target = valuesPath(directory, layer);
  const std::string temporary = target + ".tmp";
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("could not open " + temporary);
    const LayerHeader header = headerFor(rows, columns, connect, 1, layer, values.size());
    if (!writeAll(out, &header, sizeof(header))
        || !writeAll(out, values.data(), values.size())) {
      throw std::runtime_error("could not write " + temporary);
    }
  }
  publishFile(temporary, target);
}

bool loadLayerValues(const std::string& directory, int rows, int columns, int connect,
                     int layer, std::vector<std::uint8_t>& values) {
  std::ifstream in(valuesPath(directory, layer), std::ios::binary);
  if (!in) return false;
  LayerHeader seen{};
  const LayerHeader want = headerFor(rows, columns, connect, 1, layer, values.size());
  if (!readExact(in, &seen, sizeof(seen)) || !headerMatches(seen, want)) {
    std::cerr << "[layered] rejecting " << valuesPath(directory, layer) << std::endl;
    return false;
  }
  if (!readExact(in, values.data(), values.size())) {
    std::cerr << "[layered] short read on " << valuesPath(directory, layer) << std::endl;
    return false;
  }
  return true;
}

std::uint8_t packValue(int outcome) { return static_cast<std::uint8_t>(outcome + 1); }
int unpackValue(std::uint8_t packed) { return static_cast<int>(packed) - 1; }

double secondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start).count() / 1000.0;
}

// Runs body(wordBegin, wordEnd) across chunked word ranges of a layer.
template <typename Body>
void parallelWordRanges(std::uint64_t wordCount, int threads, Body&& body) {
  const std::uint64_t chunkCount =
      std::min<std::uint64_t>(std::max<std::uint64_t>(1, wordCount),
                              static_cast<std::uint64_t>(threads) * 32);
  const std::uint64_t step = (wordCount + chunkCount - 1) / chunkCount;
  if (threads <= 1) {
    for (std::uint64_t begin = 0; begin < wordCount; begin += step) {
      body(begin, std::min(wordCount, begin + step));
    }
    return;
  }
  std::atomic<std::uint64_t> cursor{0};
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(threads));
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back([&]() {
      for (;;) {
        const std::uint64_t chunk = cursor.fetch_add(1, std::memory_order_relaxed);
        const std::uint64_t begin = chunk * step;
        if (begin >= wordCount) return;
        body(begin, std::min(wordCount, begin + step));
      }
    });
  }
  for (std::thread& worker : pool) worker.join();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int rows = 4;
    int columns = 4;
    int connect = 4;
    int threads = 1;
    bool verbose = false;
    std::string output;
    for (int index = 1; index < argc; ++index) {
      const std::string name = argv[index];
      auto next = [&]() -> std::string {
        if (index + 1 >= argc) throw std::runtime_error(name + " requires a value");
        return argv[++index];
      };
      if (name == "--rows") rows = std::stoi(next());
      else if (name == "--columns") columns = std::stoi(next());
      else if (name == "--connect") connect = std::stoi(next());
      else if (name == "--threads") threads = std::stoi(next());
      else if (name == "--output") output = next();
      else if (name == "--verbose") verbose = true;
      else throw std::runtime_error("unknown argument: " + name);
    }
    if (output.empty()) throw std::runtime_error("--output directory is required");
    std::filesystem::create_directories(output);

    const auto start = std::chrono::steady_clock::now();
    const LayerGeometry geometry = makeLayerGeometry(rows, columns, connect);
    const int cellCount = geometry.cellCount;

    // The empty board is composition rank zero with no colour bits: slot 0.
    const std::uint64_t rootSlot = 0;

    // ---- Discovery, layer by layer upward. -------------------------------
    // Each layer is seeded by the previous layer's drop children, closed
    // under transformations by sweeping until stable (transform chains are
    // shallow, so no queue is needed), and written to disk.
    for (int k = 0; k < cellCount; ++k) {
      {
        LayerBits probe(geometry.layerSlots(k));
        if (loadLayerBits(output, rows, columns, connect, k, probe)) continue;
      }
      LayerBits current(geometry.layerSlots(k));
      // The delta holds the states added by the latest pass; each closure
      // sweep expands only the delta, so no state is re-expanded.
      LayerBits delta(geometry.layerSlots(k));
      if (k == 0) {
        current.set(rootSlot);
        delta.set(rootSlot);
      } else {
        LayerBits previous(geometry.layerSlots(k - 1));
        if (!loadLayerBits(output, rows, columns, connect, k - 1, previous)) {
          throw std::runtime_error("missing bits for layer " + std::to_string(k - 1));
        }
        parallelWordRanges(previous.wordCount(), threads, [&](std::uint64_t wb, std::uint64_t we) {
          Masks masks;
          int heights[MAX_SIDE];
          LayerEdgeList edges;
          previous.forEachInWordRange(wb, we, [&](std::uint64_t slot) {
            const int blockIndex = decodeLayerMasks(geometry, k - 1, slot, masks, heights);
            layerSuccessors(geometry, blockIndex, masks, heights, k - 1, edges);
            for (int e = 0; e < edges.count; ++e) {
              const LayerEdge& edge = edges.values[e];
              if (edge.terminal == NOT_TERMINAL && !edge.sameLayer
                  && current.atomicTestSet(edge.slot)) {
                delta.atomicTestSet(edge.slot);
              }
            }
          });
        });
      }
      // One scratch bitset per layer, cleared between sweeps: reallocating
      // gigabyte buffers every sweep invited allocation failure whenever the
      // machine was briefly short of commit space.
      LayerBits next(geometry.layerSlots(k));
      for (;;) {
        next.clearAll();
        std::atomic<std::uint64_t> added{0};
        parallelWordRanges(delta.wordCount(), threads, [&](std::uint64_t wb, std::uint64_t we) {
          Masks masks;
          int heights[MAX_SIDE];
          LayerEdgeList edges;
          std::uint64_t localAdded = 0;
          delta.forEachInWordRange(wb, we, [&](std::uint64_t slot) {
            const int blockIndex = decodeLayerMasks(geometry, k, slot, masks, heights);
            layerSuccessors(geometry, blockIndex, masks, heights, k, edges);
            for (int e = 0; e < edges.count; ++e) {
              const LayerEdge& edge = edges.values[e];
              if (edge.terminal == NOT_TERMINAL && edge.sameLayer
                  && current.atomicTestSet(edge.slot)) {
                next.atomicTestSet(edge.slot);
                ++localAdded;
              }
            }
          });
          added.fetch_add(localAdded, std::memory_order_relaxed);
        });
        if (added.load(std::memory_order_relaxed) == 0) break;
        std::swap(delta, next);
      }
      writeLayerBits(output, rows, columns, connect, k, current);
      current.finalize();
      if (verbose) {
        std::cerr << "[layered] discovered layer=" << k << " states=" << current.count()
                  << " seconds=" << secondsSince(start) << std::endl;
      }
    }

    // ---- Retrograde, layer by layer downward. ----------------------------
    std::uint64_t totalStates = 0;
    std::uint64_t totalWins = 0;
    std::uint64_t totalDraws = 0;
    std::uint64_t totalLosses = 0;
    LayerBits above(0);
    bool haveAbove = false;
    std::vector<std::uint8_t> aboveValues;
    std::vector<std::uint8_t> values;
    std::vector<std::uint8_t> localRanks;

    for (int k = cellCount - 1; k >= 0; --k) {
      LayerBits bits(geometry.layerSlots(k));
      if (!loadLayerBits(output, rows, columns, connect, k, bits)) {
        throw std::runtime_error("missing bits for layer " + std::to_string(k));
      }
      const std::uint64_t n = bits.count();
      values.assign(n, VALUE_UNKNOWN);

      if (n != 0 && loadLayerValues(output, rows, columns, connect, k, values)) {
        if (verbose) {
          std::cerr << "[layered] reloaded layer=" << k << " states=" << n << std::endl;
        }
      } else if (n != 0) {
        localRanks.assign(n, 0);
        int round = 1;
        for (;;) {
          std::atomic<std::uint64_t> settledShared{0};
          parallelWordRanges(bits.wordCount(), threads, [&](std::uint64_t wb, std::uint64_t we) {
            Masks masks;
            int heights[MAX_SIDE];
            LayerEdgeList edges;
            std::uint64_t localSettled = 0;
            std::uint64_t ordinal = bits.rankAtWord(wb);
            bits.forEachInWordRange(wb, we, [&](std::uint64_t slot) {
              const std::uint64_t at = ordinal++;
              if (values[at] != VALUE_UNKNOWN) return;
              const int blockIndex = decodeLayerMasks(geometry, k, slot, masks, heights);
              layerSuccessors(geometry, blockIndex, masks, heights, k, edges);
              if (edges.count == 0) return;

              bool win = false;
              bool allLoss = true;
              int maxChildRank = -1;
              bool anyUnknown = false;
              bool anyWin = false;
              bool anyDraw = false;
              for (int e = 0; e < edges.count; ++e) {
                const LayerEdge& edge = edges.values[e];
                int forMover;
                int childRank;
                if (edge.terminal != NOT_TERMINAL) {
                  forMover = edge.terminal;   // already mover-relative
                  childRank = 0;
                } else if (!edge.sameLayer) {
                  // Drop child: the layer above is fully solved.
                  const int fromChild =
                      unpackValue(aboveValues[above.rank(edge.slot)]);
                  forMover = fromChild == DRAW ? DRAW : -fromChild;
                  childRank = 0;
                } else {
                  const std::uint64_t child = bits.rank(edge.slot);
                  // Acquire pairs with the release in publish: a settled
                  // value seen here has its local rank visible too.
                  const std::uint8_t packed = std::atomic_ref<const std::uint8_t>(
                      values[child]).load(std::memory_order_acquire);
                  if (packed == VALUE_UNKNOWN) {
                    anyUnknown = true;
                    allLoss = false;
                    continue;
                  }
                  const int fromChild = unpackValue(packed);
                  forMover = fromChild == DRAW ? DRAW : -fromChild;
                  childRank = std::atomic_ref<const std::uint8_t>(
                      localRanks[child]).load(std::memory_order_relaxed);
                }
                if (forMover == WIN && childRank == round - 1) {
                  win = true;
                  break;
                }
                if (forMover != LOSS) allLoss = false;
                else if (childRank > maxChildRank) maxChildRank = childRank;
                if (forMover == WIN) anyWin = true;
                else if (forMover == DRAW) anyDraw = true;
              }
              const auto publish = [&](int outcome, std::uint8_t stateRank) {
                std::atomic_ref<std::uint8_t>(localRanks[at])
                    .store(stateRank, std::memory_order_relaxed);
                std::atomic_ref<std::uint8_t>(values[at])
                    .store(packValue(outcome), std::memory_order_release);
              };
              if (win) {
                publish(WIN, static_cast<std::uint8_t>(round));
                ++localSettled;
              } else if (allLoss && maxChildRank == round - 1) {
                publish(LOSS, static_cast<std::uint8_t>(round));
                ++localSettled;
              } else if (!anyUnknown && !anyWin && !allLoss) {
                if (!anyDraw) {
                  throw std::runtime_error("a drawn state has no drawing edge");
                }
                publish(DRAW, 0);
              }
            });
            settledShared.fetch_add(localSettled, std::memory_order_relaxed);
          });
          if (settledShared.load(std::memory_order_relaxed) == 0) break;
          ++round;
          if (round > 250) throw std::runtime_error("layer iteration did not converge");
        }
        // Anything still unresolved sits on a transformation cycle: a draw.
        for (std::uint64_t at = 0; at < n; ++at) {
          if (values[at] == VALUE_UNKNOWN) values[at] = packValue(DRAW);
        }
        writeLayerValues(output, rows, columns, connect, k, values);
        if (verbose) {
          std::cerr << "[layered] solved layer=" << k << " states=" << n
                    << " rounds=" << round << " seconds=" << secondsSince(start) << std::endl;
        }
      }

      totalStates += n;
      for (std::uint64_t at = 0; at < n; ++at) {
        const std::uint8_t packed = values[at];
        if (packed == packValue(WIN)) ++totalWins;
        else if (packed == packValue(LOSS)) ++totalLosses;
        else ++totalDraws;
      }

      above = std::move(bits);
      haveAbove = true;
      aboveValues = std::move(values);
      values = std::vector<std::uint8_t>();
    }

    if (!haveAbove || aboveValues.empty()) {
      throw std::runtime_error("layer 0 came out empty");
    }
    const int rootValue = unpackValue(aboveValues[above.rank(rootSlot)]);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    std::uint64_t slotTotal = 0;
    for (int k = 0; k < cellCount; ++k) slotTotal += geometry.layerSlots(k);

    std::cout << "{\"format\":\"connect4-chaos-exact-solution-layered-v1\""
              << ",\"rows\":" << rows << ",\"columns\":" << columns
              << ",\"connect\":" << connect
              << ",\"indexSpace\":" << slotTotal
              << ",\"states\":" << totalStates
              << ",\"wins\":" << totalWins << ",\"draws\":" << totalDraws
              << ",\"losses\":" << totalLosses
              << ",\"rootValue\":" << rootValue
              << ",\"elapsedMs\":" << elapsed << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
