// Pair-scheduled exact solver for Chaos Mode: piece-count layers split by
// mover-count pairs.
//
// Two facts about Chaos edges make the schedule work. A transformation
// passes the turn without moving a piece between owners, so from a state
// whose mover holds m of k pieces it reaches a state whose mover holds
// k - m: every transformation - and therefore every repetition cycle - is
// confined to the mover-count PAIR {m, k - m} of its layer. A drop hands
// the turn over after adding a mover piece, so from mover-count m it
// reaches mover-count k - m in layer k + 1: each pair's drops land in at
// most two pairs of the next layer, whose values are final by then.
//
// The solver therefore processes one (layer, pair) block at a time:
// discovery seeds a block from the two source blocks below it and closes it
// under transformations; resolution first streams the block's at-most-two
// drop-target blocks to precompute a per-state drop summary (any winning
// drop, all drops losing, any drawing drop - all any rank rule ever needs,
// since drop children are final), then runs the ranked iteration entirely
// inside the block, tracking "settled last round" as bitmaps. Peak memory is
// one block plus one streamed target block instead of two whole layers,
// which is what brings 6x6 c4 into reach of a 64 GB machine.
//
// Within a block, colours are indexed by their combinadic rank among words
// of their observed popcount - no assumption about which mover counts are
// reachable is ever made (transform-stalling makes every split reachable);
// the popcount is read from the word itself and both sides of a pair get
// their own sub-range.
//
// Compositions are enumerated mirror-canonically: states are canonicalised
// over horizontal mirroring anyway, so only height tuples lexicographically
// no larger than their own mirror can ever hold a state. Leaving the other
// half out of the slot space entirely halves every bitset directory - the
// dominant memory term on large boards - while changing no state, count or
// value.
//
// Artifacts, one pair of files per block, in the output directory:
//   pair-<k>-<j>.bits     reachable-slot bitset for pair {j, k-j} of layer k
//   pair-<k>-<j>.values   solved values by block ordinal
// Both double as checkpoints; a restarted run resumes at the first missing
// block. One JSON solution line goes to stdout on success. Counts are
// cross-checked against the layered and monolithic solvers on every board
// solved by more than one of them.

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
constexpr int MAX_CELLS = 49;   // boards are capped at 7x7

constexpr int ACTION_DROP = 0;
constexpr int ACTION_FLIP = 1;
constexpr int ACTION_ROTATE_CW = 2;
constexpr int ACTION_ROTATE_CCW = 3;

// ---------------------------------------------------------------------------
// Positions as bitboards, exactly as in perfect-chaos-layered.cpp.
// ---------------------------------------------------------------------------

struct Masks {
  std::uint64_t mover = 0;
  std::uint64_t opponent = 0;
};

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
// Geometry: compositions, binomials, and the pair-block slot spaces
// ---------------------------------------------------------------------------

struct BinomialTable {
  std::array<std::array<std::uint64_t, MAX_CELLS + 1>, MAX_CELLS + 1> at{};
  BinomialTable() {
    for (int n = 0; n <= MAX_CELLS; ++n) {
      at[n][0] = 1;
      for (int k = 1; k <= n; ++k) {
        at[n][k] = at[n - 1][k - 1] + (k <= n - 1 ? at[n - 1][k] : 0);
      }
    }
  }
};
const BinomialTable BINOMIAL;

// Combinadic rank of a word among words with its own popcount, in numeric
// (colex) order; the inverse takes the popcount explicitly.
std::uint64_t colourRankM(std::uint64_t word) {
  std::uint64_t rank = 0;
  int seen = 0;
  while (word != 0) {
    const int position = __builtin_ctzll(word);
    ++seen;
    rank += BINOMIAL.at[position][seen];
    word &= word - 1;
  }
  return rank;
}

std::uint64_t colourUnrankM(std::uint64_t rank, int bits, int ones) {
  std::uint64_t word = 0;
  for (int remaining = ones; remaining >= 1; --remaining) {
    int position = remaining - 1;
    while (position + 1 < bits && BINOMIAL.at[position + 1][remaining] <= rank) ++position;
    rank -= BINOMIAL.at[position][remaining];
    word |= std::uint64_t{1} << position;
  }
  return word;
}

struct BlockShape {
  int rows = 0;
  int columns = 0;
  // Mirror-canonical compositions only, tabulated: canon[pieces] lists the
  // packed height tuples that are lexicographically no larger than their own
  // left-right mirror, in lexicographic order; rankOf inverts the listing.
  // canonicalPairSlot always ranks the canonical orientation, so the mirrored
  // half of the composition space never needs slots.
  std::vector<std::vector<std::uint32_t>> canon;
  std::vector<std::uint32_t> rankOf;
  static constexpr std::uint32_t NOT_CANONICAL = 0xFFFFFFFFu;

  std::uint32_t pack(const int* heights) const {
    std::uint32_t code = 0;
    for (int c = 0; c < columns; ++c) {
      code |= static_cast<std::uint32_t>(heights[c]) << (3 * c);
    }
    return code;
  }
  void unpack(std::uint32_t code, int* heights) const {
    for (int c = 0; c < columns; ++c) heights[c] = (code >> (3 * c)) & 7;
  }

  void build() {
    canon.assign(static_cast<std::size_t>(rows) * columns + 1, {});
    rankOf.assign(std::size_t{1} << (3 * columns), NOT_CANONICAL);
    int heights[MAX_SIDE] = {};
    for (;;) {
      bool canonical = true;
      for (int c = 0; c < columns; ++c) {
        const int mirrored = heights[columns - 1 - c];
        if (heights[c] != mirrored) {
          canonical = heights[c] < mirrored;
          break;
        }
      }
      if (canonical) {
        int pieces = 0;
        for (int c = 0; c < columns; ++c) pieces += heights[c];
        const std::uint32_t code = pack(heights);
        rankOf[code] = static_cast<std::uint32_t>(canon[pieces].size());
        canon[pieces].push_back(code);
      }
      int column = columns - 1;
      while (column >= 0 && heights[column] == rows) heights[column--] = 0;
      if (column < 0) break;
      ++heights[column];
    }
  }
};

// The mover count identifying a pair: always the larger side.
int pairOf(int pieces, int moverCount) { return std::max(moverCount, pieces - moverCount); }

struct PairGeometry {
  int connect = 0;
  int cellCount = 0;
  std::array<BlockShape, 2> blocks{};
  int blockCount = 0;

  int blockIndexFor(int rows, int columns) const {
    for (int index = 0; index < blockCount; ++index) {
      if (blocks[index].rows == rows && blocks[index].columns == columns) return index;
    }
    throw std::runtime_error("board shape is outside the pair index space");
  }

  // Colour words per composition inside pair {j, k-j}: the j side first.
  std::uint64_t pairColourSlots(int pieces, int pairId) const {
    const std::uint64_t high = BINOMIAL.at[pieces][pairId];
    if (pairId * 2 == pieces) return high;
    return high + BINOMIAL.at[pieces][pieces - pairId];
  }

  std::uint64_t blockPairSlots(int block, int pieces, int pairId) const {
    return blocks[block].canon[pieces].size() * pairColourSlots(pieces, pairId);
  }

  std::uint64_t blockPairOffset(int block, int pieces, int pairId) const {
    std::uint64_t offset = 0;
    for (int index = 0; index < block; ++index) offset += blockPairSlots(index, pieces, pairId);
    return offset;
  }

  std::uint64_t pairSlots(int pieces, int pairId) const {
    std::uint64_t total = 0;
    for (int index = 0; index < blockCount; ++index) total += blockPairSlots(index, pieces, pairId);
    return total;
  }
};

PairGeometry makePairGeometry(int rows, int columns, int connect) {
  if (rows < 1 || rows >= MAX_SIDE || columns < 1 || columns >= MAX_SIDE) {
    throw std::range_error("board dimensions are out of range");
  }
  if (connect < 1 || connect > std::max(rows, columns)) {
    throw std::range_error("connect length does not fit the board");
  }
  PairGeometry geometry;
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

// Colour word of the mover mask, columns left to right, bottom to top.
std::uint64_t colourWordOf(const Masks& masks, const int* heights, int columns, int stride) {
  std::uint64_t colours = 0;
  int offset = 0;
  for (int column = 0; column < columns; ++column) {
    const std::uint64_t width = (std::uint64_t{1} << heights[column]) - 1;
    colours |= ((masks.mover >> (column * stride)) & width) << offset;
    offset += heights[column];
  }
  return colours;
}

// Sub-slot of a colour word inside its pair: j-side words first, then the
// k-j side, each ordered by combinadic rank (numeric order within a side).
std::uint64_t colourSubslot(std::uint64_t word, int pieces, int pairId) {
  const int ones = __builtin_popcountll(word);
  std::uint64_t base = 0;
  if (ones != pairId) {
    if (ones != pieces - pairId) throw std::runtime_error("colour word outside its pair");
    base = BINOMIAL.at[pieces][pairId];
  }
  return base + colourRankM(word);
}

// Slot of a board (given as masks + heights) inside pair (pieces, pairId),
// canonicalised over horizontal mirroring, which preserves the mover count
// and therefore the pair.
std::uint64_t canonicalPairSlot(const PairGeometry& geometry, int blockIndex,
                                const Masks& masks, const int* heights,
                                int pieces, int pairId) {
  const BlockShape& block = geometry.blocks[blockIndex];
  const int columns = block.columns;
  const int stride = block.rows + 1;

  int order = 0;
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
    colours = colourWordOf(masks, heights, columns, stride);
    rank = block.rankOf[block.pack(heights)];
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
    rank = block.rankOf[block.pack(reversedHeights)];
  }
  if (rank == BlockShape::NOT_CANONICAL) {
    throw std::runtime_error("canonicalisation reached a non-canonical composition");
  }
  return geometry.blockPairOffset(blockIndex, pieces, pairId)
      + rank * geometry.pairColourSlots(pieces, pairId)
      + colourSubslot(colours, pieces, pairId);
}

// Decodes a pair slot to masks, heights and mover count; returns block index.
int decodePairSlot(const PairGeometry& geometry, int pieces, int pairId,
                   std::uint64_t slot, Masks& masks, int* heights, int& moverCount) {
  int blockIndex = 0;
  while (blockIndex + 1 < geometry.blockCount
         && slot >= geometry.blockPairOffset(blockIndex + 1, pieces, pairId)) {
    ++blockIndex;
  }
  const BlockShape& block = geometry.blocks[blockIndex];
  slot -= geometry.blockPairOffset(blockIndex, pieces, pairId);

  const std::uint64_t colourSlots = geometry.pairColourSlots(pieces, pairId);
  std::uint64_t compositionRank = slot / colourSlots;
  std::uint64_t sub = slot % colourSlots;
  moverCount = pairId;
  if (sub >= BINOMIAL.at[pieces][pairId]) {
    sub -= BINOMIAL.at[pieces][pairId];
    moverCount = pieces - pairId;
  }
  const std::uint64_t colours = colourUnrankM(sub, pieces, moverCount);

  block.unpack(block.canon[pieces][compositionRank], heights);
  const int stride = block.rows + 1;
  masks.mover = 0;
  masks.opponent = 0;
  std::uint64_t rest = colours;
  for (int column = 0; column < block.columns; ++column) {
    const int height = heights[column];
    const std::uint64_t occupied = (std::uint64_t{1} << height) - 1;
    const std::uint64_t segment = rest & occupied;
    rest >>= height;
    masks.mover |= segment << (column * stride);
    masks.opponent |= (occupied ^ segment) << (column * stride);
  }
  return blockIndex;
}

// ---------------------------------------------------------------------------
// Edges. Transform children stay in the pair; drop children carry the target
// pair of the next layer.
// ---------------------------------------------------------------------------

struct PairEdge {
  std::int8_t terminal = NOT_TERMINAL;
  bool sameLayer = false;
  std::uint8_t targetPair = 0;    // drops only: pair id in layer pieces + 1
  std::uint64_t slot = 0;
};

struct PairEdgeList {
  std::array<PairEdge, MAX_SIDE + 3> values{};
  int count = 0;
};

void pairSuccessors(const PairGeometry& geometry, int blockIndex, const Masks& masks,
                    const int* heights, int pieces, int moverCount, PairEdgeList& edges) {
  edges.count = 0;
  const BlockShape& block = geometry.blocks[blockIndex];
  const int rows = block.rows;
  const int columns = block.columns;
  const int stride = rows + 1;

  const auto emitTerminal = [&edges](int terminal, bool sameLayer) {
    PairEdge edge;
    edge.terminal = static_cast<std::int8_t>(terminal);
    edge.sameLayer = sameLayer;
    edges.values[edges.count++] = edge;
  };

  // Drops: child mover count is pieces - moverCount in layer pieces + 1.
  const int dropMover = pieces - moverCount;
  const int dropPair = pairOf(pieces + 1, dropMover);
  for (int column = 0; column < columns; ++column) {
    const int height = heights[column];
    if (height >= rows) continue;
    const std::uint64_t grown =
        masks.mover | (std::uint64_t{1} << (column * stride + height));
    if (maskHasLine(grown, rows, geometry.connect)) {
      emitTerminal(WIN, false);
      continue;
    }
    if (pieces + 1 == geometry.cellCount) {
      emitTerminal(DRAW, false);
      continue;
    }
    int childHeights[MAX_SIDE];
    for (int c = 0; c < columns; ++c) childHeights[c] = heights[c];
    ++childHeights[column];
    const Masks child{masks.opponent, grown};
    PairEdge edge;
    edge.sameLayer = false;
    edge.targetPair = static_cast<std::uint8_t>(dropPair);
    edge.slot = canonicalPairSlot(geometry, blockIndex, child, childHeights,
                                  pieces + 1, dropPair);
    edges.values[edges.count++] = edge;
  }

  // Transforms: child mover count is pieces - moverCount, same layer, and
  // pairOf is unchanged, so the child stays in this block.
  const auto settleTransform = [&](const Masks& next, int nextBlock,
                                   const int* nextHeights, int nextRows) {
    const bool moverLine = maskHasLine(next.mover, nextRows, geometry.connect);
    const bool opponentLine = maskHasLine(next.opponent, nextRows, geometry.connect);
    if (moverLine || opponentLine) {
      emitTerminal(moverLine && opponentLine ? LOSS : (moverLine ? WIN : LOSS), true);
      return;
    }
    const Masks child{next.opponent, next.mover};
    PairEdge edge;
    edge.sameLayer = true;
    edge.slot = canonicalPairSlot(geometry, nextBlock, child, nextHeights,
                                  pieces, pairOf(pieces, moverCount));
    edges.values[edges.count++] = edge;
  };

  {
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
  const int targetStride = columns + 1;
  int rotatedHeights[MAX_SIDE];

  {
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
// Block bitset with u64 rank directory (as in the layered solver)
// ---------------------------------------------------------------------------

class BlockBits {
 public:
  explicit BlockBits(std::uint64_t slots)
      : words_((slots + 63) / 64 + (slots == 0 ? 1 : 0), 0), slots_(slots) {}
  BlockBits(BlockBits&&) = default;
  BlockBits& operator=(BlockBits&&) = default;

  bool test(std::uint64_t slot) const {
    return (words_[slot >> 6] >> (slot & 63)) & 1;
  }
  void set(std::uint64_t slot) {
    words_[slot >> 6] |= std::uint64_t{1} << (slot & 63);
  }
  bool atomicTestSet(std::uint64_t slot) {
    std::atomic_ref<std::uint64_t> word(words_[slot >> 6]);
    const std::uint64_t mask = std::uint64_t{1} << (slot & 63);
    return (word.fetch_or(mask, std::memory_order_relaxed) & mask) == 0;
  }
  void clearAll() { std::fill(words_.begin(), words_.end(), 0); }

  void finalize() {
    ranks_.assign(words_.size() + 1, 0);
    std::uint64_t running = 0;
    for (std::size_t word = 0; word < words_.size(); ++word) {
      ranks_[word] = running;
      running += static_cast<std::uint64_t>(__builtin_popcountll(words_[word]));
    }
    ranks_[words_.size()] = running;
    count_ = running;
  }

  std::uint64_t rank(std::uint64_t slot) const {
    const std::uint64_t word = slot >> 6;
    const std::uint64_t below = words_[word] & ((std::uint64_t{1} << (slot & 63)) - 1);
    return ranks_[word] + static_cast<std::uint64_t>(__builtin_popcountll(below));
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
  void forEach(Visit&& visit) const { forEachInWordRange(0, words_.size(), visit); }

  std::vector<std::uint64_t>& mutableWords() { return words_; }
  const std::vector<std::uint64_t>& words() const { return words_; }

 private:
  std::vector<std::uint64_t> words_;
  std::vector<std::uint64_t> ranks_;
  std::uint64_t slots_ = 0;
  std::uint64_t count_ = 0;
};

// Two bits per state; unknown -> settled clears bits, so publication is a
// release fetch_and (as in the layered solver).
class PackedValues {
 public:
  void assign(std::uint64_t states, std::uint8_t fill) {
    states_ = states;
    std::uint64_t pattern = 0;
    for (int slot = 0; slot < 32; ++slot) {
      pattern |= static_cast<std::uint64_t>(fill & 3) << (slot * 2);
    }
    words_.assign((states + 31) / 32 + (states == 0 ? 1 : 0), pattern);
  }
  std::uint8_t get(std::uint64_t at) const {
    return (words_[at >> 5] >> ((at & 31) * 2)) & 3;
  }
  std::uint8_t getAcquire(std::uint64_t at) const {
    return (std::atomic_ref<const std::uint64_t>(words_[at >> 5])
                .load(std::memory_order_acquire) >> ((at & 31) * 2)) & 3;
  }
  void publish(std::uint64_t at, std::uint8_t value) {
    const int shift = static_cast<int>(at & 31) * 2;
    const std::uint64_t clear =
        ~(static_cast<std::uint64_t>((value ^ 3) & 3) << shift);
    std::atomic_ref<std::uint64_t>(words_[at >> 5])
        .fetch_and(clear, std::memory_order_release);
  }
  std::uint64_t size() const { return states_; }

 private:
  std::vector<std::uint64_t> words_;
  std::uint64_t states_ = 0;
};

// One bit per state with atomic set; used for round tracking and summaries.
class StateBits {
 public:
  void assign(std::uint64_t states) {
    words_.assign((states + 63) / 64 + (states == 0 ? 1 : 0), 0);
  }
  bool test(std::uint64_t at) const {
    return (words_[at >> 6] >> (at & 63)) & 1;
  }
  bool testAcquire(std::uint64_t at) const {
    return ((std::atomic_ref<const std::uint64_t>(words_[at >> 6])
                 .load(std::memory_order_acquire) >> (at & 63)) & 1) != 0;
  }
  void set(std::uint64_t at) {
    words_[at >> 6] |= std::uint64_t{1} << (at & 63);
  }
  void atomicSet(std::uint64_t at) {
    std::atomic_ref<std::uint64_t>(words_[at >> 6])
        .fetch_or(std::uint64_t{1} << (at & 63), std::memory_order_release);
  }
  void clearAll() { std::fill(words_.begin(), words_.end(), 0); }
  void swapWith(StateBits& other) { words_.swap(other.words_); }

 private:
  std::vector<std::uint64_t> words_;
};

// ---------------------------------------------------------------------------
// Chunked file I/O with block headers
// ---------------------------------------------------------------------------

constexpr std::size_t IO_CHUNK = std::size_t{256} << 20;
constexpr char PAIR_MAGIC[8] = {'C', '4', 'P', 'A', 'I', 'R', '2', '\0'};

struct PairHeader {
  char magic[8];
  std::uint8_t rows, columns, connect, kind;   // kind: 0 bits, 1 values
  std::uint16_t layer, pairId;
  std::uint64_t payload;
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

PairHeader headerFor(int rows, int columns, int connect, int kind, int layer,
                     int pairId, std::uint64_t payload) {
  PairHeader header{};
  std::memcpy(header.magic, PAIR_MAGIC, sizeof(header.magic));
  header.rows = static_cast<std::uint8_t>(rows);
  header.columns = static_cast<std::uint8_t>(columns);
  header.connect = static_cast<std::uint8_t>(connect);
  header.kind = static_cast<std::uint8_t>(kind);
  header.layer = static_cast<std::uint16_t>(layer);
  header.pairId = static_cast<std::uint16_t>(pairId);
  header.payload = payload;
  return header;
}

bool headerMatches(const PairHeader& seen, const PairHeader& want) {
  return std::memcmp(seen.magic, want.magic, sizeof(want.magic)) == 0
      && seen.rows == want.rows && seen.columns == want.columns
      && seen.connect == want.connect && seen.kind == want.kind
      && seen.layer == want.layer && seen.pairId == want.pairId
      && seen.payload == want.payload;
}

std::string bitsPath(const std::string& d, int layer, int pairId) {
  return d + "/pair-" + std::to_string(layer) + "-" + std::to_string(pairId) + ".bits";
}
std::string valuesPath(const std::string& d, int layer, int pairId) {
  return d + "/pair-" + std::to_string(layer) + "-" + std::to_string(pairId) + ".values";
}

void writeBlockBits(const std::string& directory, int rows, int columns, int connect,
                    int layer, int pairId, const BlockBits& bits) {
  const std::string target = bitsPath(directory, layer, pairId);
  const std::string temporary = target + ".tmp";
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("could not open " + temporary);
    const PairHeader header =
        headerFor(rows, columns, connect, 0, layer, pairId, bits.wordCount());
    if (!writeAll(out, &header, sizeof(header))
        || !writeAll(out, bits.words().data(), bits.wordCount() * sizeof(std::uint64_t))) {
      throw std::runtime_error("could not write " + temporary);
    }
  }
  publishFile(temporary, target);
}

bool loadBlockBits(const std::string& directory, int rows, int columns, int connect,
                   int layer, int pairId, BlockBits& bits) {
  std::ifstream in(bitsPath(directory, layer, pairId), std::ios::binary);
  if (!in) return false;
  PairHeader seen{};
  const PairHeader want =
      headerFor(rows, columns, connect, 0, layer, pairId, bits.wordCount());
  if (!readExact(in, &seen, sizeof(seen)) || !headerMatches(seen, want)) {
    std::cerr << "[paired] rejecting " << bitsPath(directory, layer, pairId) << std::endl;
    return false;
  }
  if (!readExact(in, bits.mutableWords().data(), bits.wordCount() * sizeof(std::uint64_t))) {
    std::cerr << "[paired] short read on " << bitsPath(directory, layer, pairId) << std::endl;
    return false;
  }
  bits.finalize();
  return true;
}

void writeBlockValues(const std::string& directory, int rows, int columns, int connect,
                      int layer, int pairId, const PackedValues& values) {
  const std::string target = valuesPath(directory, layer, pairId);
  const std::string temporary = target + ".tmp";
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("could not open " + temporary);
    const PairHeader header =
        headerFor(rows, columns, connect, 1, layer, pairId, values.size());
    if (!writeAll(out, &header, sizeof(header))) {
      throw std::runtime_error("could not write " + temporary);
    }
    std::vector<std::uint8_t> staging;
    const std::uint64_t chunk = std::uint64_t{64} << 20;
    for (std::uint64_t begin = 0; begin < values.size(); begin += chunk) {
      const std::uint64_t count = std::min(chunk, values.size() - begin);
      staging.resize(count);
      for (std::uint64_t index = 0; index < count; ++index) {
        staging[index] = values.get(begin + index);
      }
      if (!writeAll(out, staging.data(), count)) {
        throw std::runtime_error("could not write " + temporary);
      }
    }
  }
  publishFile(temporary, target);
}

bool loadBlockValues(const std::string& directory, int rows, int columns, int connect,
                     int layer, int pairId, PackedValues& values) {
  std::ifstream in(valuesPath(directory, layer, pairId), std::ios::binary);
  if (!in) return false;
  PairHeader seen{};
  const PairHeader want =
      headerFor(rows, columns, connect, 1, layer, pairId, values.size());
  if (!readExact(in, &seen, sizeof(seen)) || !headerMatches(seen, want)) {
    std::cerr << "[paired] rejecting " << valuesPath(directory, layer, pairId) << std::endl;
    return false;
  }
  std::vector<std::uint8_t> staging;
  const std::uint64_t chunk = std::uint64_t{64} << 20;
  for (std::uint64_t begin = 0; begin < values.size(); begin += chunk) {
    const std::uint64_t count = std::min(chunk, values.size() - begin);
    staging.resize(count);
    if (!readExact(in, staging.data(), count)) {
      std::cerr << "[paired] short read on " << valuesPath(directory, layer, pairId) << std::endl;
      return false;
    }
    for (std::uint64_t index = 0; index < count; ++index) {
      values.publish(begin + index, staging[index]);
    }
  }
  return true;
}

std::uint8_t packValue(int outcome) { return static_cast<std::uint8_t>(outcome + 1); }
int unpackValue(std::uint8_t packed) { return static_cast<int>(packed) - 1; }

double secondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start).count() / 1000.0;
}

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

int highestPair(int pieces) { return pieces; }
int lowestPair(int pieces) { return (pieces + 1) / 2; }

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
    const PairGeometry geometry = makePairGeometry(rows, columns, connect);
    const int cellCount = geometry.cellCount;

    // ---- Discovery: layers ascending, one block at a time. ---------------
    int topLayer = -1;
    for (int k = 0; k < cellCount; ++k) {
      std::uint64_t layerStates = 0;
      for (int j = lowestPair(k); j <= highestPair(k); ++j) {
        {
          BlockBits probe(geometry.pairSlots(k, j));
          if (loadBlockBits(output, rows, columns, connect, k, j, probe)) {
            layerStates += probe.count();
            continue;
          }
        }
        BlockBits current(geometry.pairSlots(k, j));
        if (k == 0) {
          current.set(0);   // the empty board: mover count 0, pair 0
        } else {
          // Drops from mover count m in layer k-1 land at mover count
          // k-1-m here; for that to lie in {j, k-j}, m must be j-1 or
          // k-1-j, whose pairs are j-1 and j.
          for (const int sourcePair : {j - 1, j}) {
            if (sourcePair < lowestPair(k - 1) || sourcePair > highestPair(k - 1)) continue;
            BlockBits source(geometry.pairSlots(k - 1, sourcePair));
            if (!loadBlockBits(output, rows, columns, connect, k - 1, sourcePair, source)) {
              throw std::runtime_error("missing source block " + std::to_string(k - 1)
                                       + "-" + std::to_string(sourcePair));
            }
            parallelWordRanges(source.wordCount(), threads,
                               [&](std::uint64_t wb, std::uint64_t we) {
              Masks masks;
              int heights[MAX_SIDE];
              int moverCount;
              PairEdgeList edges;
              source.forEachInWordRange(wb, we, [&](std::uint64_t slot) {
                const int blockIndex = decodePairSlot(geometry, k - 1, sourcePair, slot,
                                                      masks, heights, moverCount);
                pairSuccessors(geometry, blockIndex, masks, heights, k - 1, moverCount, edges);
                for (int e = 0; e < edges.count; ++e) {
                  const PairEdge& edge = edges.values[e];
                  if (edge.terminal == NOT_TERMINAL && !edge.sameLayer
                      && edge.targetPair == j) {
                    current.atomicTestSet(edge.slot);
                  }
                }
              });
            });
          }
        }
        // Close under transformations, which are confined to the block.
        // Sweeping until stable re-expands settled states but needs no
        // extra bitsets, which matters at 6x6 block sizes.
        for (;;) {
          std::atomic<std::uint64_t> added{0};
          parallelWordRanges(current.wordCount(), threads,
                             [&](std::uint64_t wb, std::uint64_t we) {
            Masks masks;
            int heights[MAX_SIDE];
            int moverCount;
            PairEdgeList edges;
            std::uint64_t localAdded = 0;
            current.forEachInWordRange(wb, we, [&](std::uint64_t slot) {
              const int blockIndex = decodePairSlot(geometry, k, j, slot,
                                                    masks, heights, moverCount);
              pairSuccessors(geometry, blockIndex, masks, heights, k, moverCount, edges);
              for (int e = 0; e < edges.count; ++e) {
                const PairEdge& edge = edges.values[e];
                if (edge.terminal == NOT_TERMINAL && edge.sameLayer
                    && current.atomicTestSet(edge.slot)) {
                  ++localAdded;
                }
              }
            });
            added.fetch_add(localAdded, std::memory_order_relaxed);
          });
          if (added.load(std::memory_order_relaxed) == 0) break;
        }
        writeBlockBits(output, rows, columns, connect, k, j, current);
        current.finalize();
        layerStates += current.count();
        if (verbose) {
          std::cerr << "[paired] discovered block " << k << "-" << j
                    << " states=" << current.count()
                    << " seconds=" << secondsSince(start) << std::endl;
        }
      }
      if (layerStates == 0 && k > 0) break;
      topLayer = k;
    }

    // ---- Resolution: layers descending, one block at a time. -------------
    std::uint64_t totalStates = 0;
    std::uint64_t totalWins = 0;
    std::uint64_t totalDraws = 0;
    std::uint64_t totalLosses = 0;

    for (int k = topLayer; k >= 0; --k) {
      for (int j = highestPair(k); j >= lowestPair(k); --j) {
        BlockBits bits(geometry.pairSlots(k, j));
        if (!loadBlockBits(output, rows, columns, connect, k, j, bits)) {
          throw std::runtime_error("missing bits for block " + std::to_string(k)
                                   + "-" + std::to_string(j));
        }
        const std::uint64_t n = bits.count();
        PackedValues values;
        values.assign(n, VALUE_UNKNOWN);
        if (n == 0) continue;

        if (loadBlockValues(output, rows, columns, connect, k, j, values)) {
          if (verbose) {
            std::cerr << "[paired] reloaded block " << k << "-" << j
                      << " states=" << n << std::endl;
          }
        } else {
          // Drop summaries: one streamed pass per target block, plus the
          // terminal drops handled in the first pass. Drop children are
          // final, so three bits per state cover every rank rule.
          StateBits dropWin, dropNonLoss, dropDraw;
          dropWin.assign(n);
          dropNonLoss.assign(n);
          dropDraw.assign(n);

          int targets[2];
          int targetCount = 0;
          if (k + 1 <= topLayer) {
            for (const int t : {j, j + 1}) {
              if (t >= lowestPair(k + 1) && t <= highestPair(k + 1)) {
                targets[targetCount++] = t;
              }
            }
          }
          for (int pass = 0; pass < std::max(1, targetCount); ++pass) {
            const bool haveTarget = pass < targetCount;
            BlockBits targetBits(haveTarget ? geometry.pairSlots(k + 1, targets[pass]) : 0);
            PackedValues targetValues;
            if (haveTarget) {
              if (!loadBlockBits(output, rows, columns, connect, k + 1, targets[pass], targetBits)) {
                throw std::runtime_error("missing target block for summaries");
              }
              targetValues.assign(targetBits.count(), VALUE_UNKNOWN);
              if (targetBits.count() != 0
                  && !loadBlockValues(output, rows, columns, connect, k + 1, targets[pass], targetValues)) {
                throw std::runtime_error("missing target values for summaries");
              }
            }
            const bool firstPass = pass == 0;
            parallelWordRanges(bits.wordCount(), threads,
                               [&](std::uint64_t wb, std::uint64_t we) {
              Masks masks;
              int heights[MAX_SIDE];
              int moverCount;
              PairEdgeList edges;
              std::uint64_t ordinal = bits.rankAtWord(wb);
              bits.forEachInWordRange(wb, we, [&](std::uint64_t slot) {
                const std::uint64_t at = ordinal++;
                const int blockIndex = decodePairSlot(geometry, k, j, slot,
                                                      masks, heights, moverCount);
                pairSuccessors(geometry, blockIndex, masks, heights, k, moverCount, edges);
                for (int e = 0; e < edges.count; ++e) {
                  const PairEdge& edge = edges.values[e];
                  if (edge.sameLayer) continue;
                  if (edge.terminal != NOT_TERMINAL) {
                    if (!firstPass) continue;
                    // A drop terminal is a mover win or a filling draw.
                    if (edge.terminal == WIN) dropWin.atomicSet(at);
                    else dropDraw.atomicSet(at);
                    dropNonLoss.atomicSet(at);
                    continue;
                  }
                  if (!haveTarget || edge.targetPair != targets[pass]) continue;
                  const int fromChild =
                      unpackValue(targetValues.get(targetBits.rank(edge.slot)));
                  const int forMover = fromChild == DRAW ? DRAW : -fromChild;
                  if (forMover == WIN) dropWin.atomicSet(at);
                  if (forMover != LOSS) dropNonLoss.atomicSet(at);
                  if (forMover == DRAW) dropDraw.atomicSet(at);
                }
              });
            });
          }

          // Ranked iteration inside the block. A child settled in round r
          // has rank r; terminals and drops rank zero.
          StateBits settledPrev, settledCur;
          settledPrev.assign(n);
          settledCur.assign(n);
          int round = 1;
          for (;;) {
            std::atomic<std::uint64_t> settledShared{0};
            parallelWordRanges(bits.wordCount(), threads,
                               [&](std::uint64_t wb, std::uint64_t we) {
              Masks masks;
              int heights[MAX_SIDE];
              int moverCount;
              PairEdgeList edges;
              std::uint64_t localSettled = 0;
              std::uint64_t ordinal = bits.rankAtWord(wb);
              bits.forEachInWordRange(wb, we, [&](std::uint64_t slot) {
                const std::uint64_t at = ordinal++;
                if (values.get(at) != VALUE_UNKNOWN) return;
                const int blockIndex = decodePairSlot(geometry, k, j, slot,
                                                      masks, heights, moverCount);
                pairSuccessors(geometry, blockIndex, masks, heights, k, moverCount, edges);

                bool win = false;
                bool allLoss = true;
                bool anyUnknown = false;
                bool anyWinAvailable = false;
                bool anyDrawAvailable = false;
                bool lossWitnessPrev = false;

                if (dropWin.test(at)) {
                  anyWinAvailable = true;
                  if (round == 1) win = true;
                }
                if (dropNonLoss.test(at)) allLoss = false;
                if (dropDraw.test(at)) anyDrawAvailable = true;

                for (int e = 0; e < edges.count && !win; ++e) {
                  const PairEdge& edge = edges.values[e];
                  if (!edge.sameLayer) continue;   // drops live in the summaries
                  if (edge.terminal != NOT_TERMINAL) {
                    // Transform terminals have rank zero: WIN or LOSS only.
                    if (edge.terminal == WIN) {
                      anyWinAvailable = true;
                      if (round == 1) win = true;
                    }
                    if (edge.terminal != LOSS) allLoss = false;
                    continue;
                  }
                  const std::uint64_t child = bits.rank(edge.slot);
                  const std::uint8_t packed = values.getAcquire(child);
                  if (packed == VALUE_UNKNOWN) {
                    anyUnknown = true;
                    allLoss = false;
                    continue;
                  }
                  const int fromChild = unpackValue(packed);
                  const int forMover = fromChild == DRAW ? DRAW : -fromChild;
                  if (forMover == WIN) {
                    anyWinAvailable = true;
                    if (settledPrev.testAcquire(child)) win = true;
                  }
                  if (forMover != LOSS) allLoss = false;
                  else if (settledPrev.testAcquire(child)) lossWitnessPrev = true;
                  if (forMover == DRAW) anyDrawAvailable = true;
                }

                if (win) {
                  values.publish(at, packValue(WIN));
                  settledCur.atomicSet(at);
                  ++localSettled;
                } else if (allLoss && (round == 1 || lossWitnessPrev)) {
                  values.publish(at, packValue(LOSS));
                  settledCur.atomicSet(at);
                  ++localSettled;
                } else if (!anyUnknown && !anyWinAvailable && !allLoss) {
                  if (!anyDrawAvailable) {
                    throw std::runtime_error("a drawn state has no drawing edge");
                  }
                  values.publish(at, packValue(DRAW));
                }
              });
              settledShared.fetch_add(localSettled, std::memory_order_relaxed);
            });
            if (settledShared.load(std::memory_order_relaxed) == 0) break;
            settledPrev.swapWith(settledCur);
            settledCur.clearAll();
            ++round;
            if (round > 250) throw std::runtime_error("block iteration did not converge");
          }
          for (std::uint64_t at = 0; at < n; ++at) {
            if (values.get(at) == VALUE_UNKNOWN) values.publish(at, packValue(DRAW));
          }
          writeBlockValues(output, rows, columns, connect, k, j, values);
          if (verbose) {
            std::cerr << "[paired] solved block " << k << "-" << j
                      << " states=" << n << " rounds=" << round
                      << " seconds=" << secondsSince(start) << std::endl;
          }
        }

        totalStates += n;
        for (std::uint64_t at = 0; at < n; ++at) {
          const std::uint8_t packed = values.get(at);
          if (packed == packValue(WIN)) ++totalWins;
          else if (packed == packValue(LOSS)) ++totalLosses;
          else ++totalDraws;
        }
      }
    }

    // The root is the single state of block (0, 0).
    int rootValue = 0;
    {
      BlockBits bits(geometry.pairSlots(0, 0));
      if (!loadBlockBits(output, rows, columns, connect, 0, 0, bits) || bits.count() == 0) {
        throw std::runtime_error("root block came out empty");
      }
      PackedValues values;
      values.assign(bits.count(), VALUE_UNKNOWN);
      if (!loadBlockValues(output, rows, columns, connect, 0, 0, values)) {
        throw std::runtime_error("root values missing");
      }
      rootValue = unpackValue(values.get(bits.rank(0)));
    }

    std::uint64_t slotTotal = 0;
    for (int k = 0; k < cellCount; ++k) {
      for (int j = lowestPair(k); j <= highestPair(k); ++j) {
        slotTotal += geometry.pairSlots(k, j);
      }
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "{\"format\":\"connect4-chaos-exact-solution-paired-v1\""
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
