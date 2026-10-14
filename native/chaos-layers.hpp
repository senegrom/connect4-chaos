// What the two layer-based Chaos solvers (perfect-chaos-layered and
// perfect-chaos-paired) share: positions as bitboard masks, the flip and the
// rotations, a board's canonical orientation, line detection on a mask,
// values packed two bits to a state, and the checkpoint files a layer or
// block is stored in.
//
// A mask has bit column * (rows + 1) + row, one guard bit per column so run
// detection cannot wrap between columns. Gravity keeps every column's pieces
// contiguous from the bottom, so a column's height is the size of its occupied
// segment and transformations move whole segments. Both solvers cap boards
// at 7x7, which fits 64 bits with the guards.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "atomic-load.hpp"
#include "checkpoint-io.hpp"

namespace connect4 {

// Board sides stay below this: 7x7 at most.
constexpr int MAX_SIDE = 8;

struct Masks {
  std::uint64_t mover = 0;
  std::uint64_t opponent = 0;
};

// reversed[h][bits]: `bits` reversed within h bits, for flipping stacks of up
// to seven pieces.
struct ReverseTable {
  static constexpr int MAX_HEIGHT = 7;
  std::array<std::array<std::uint8_t, 1 << MAX_HEIGHT>, MAX_HEIGHT + 1> table{};
  ReverseTable() {
    for (int width = 1; width <= MAX_HEIGHT; ++width) {
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
inline const ReverseTable REVERSE;

inline bool maskHasLine(std::uint64_t mask, int rows, int connect) {
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

// Hands each transformation of a rows x columns board to visit(next,
// nextHeights, rotated), in the order both solvers list their edges: the flip,
// then the clockwise and the counter-clockwise rotation. A rotation transposes
// the board, so its result has `columns` rows; on a board that is not square
// that shape is the index space's other block.
template <typename Visit>
void forEachTransform(const Masks& masks, const int* heights, int rows, int columns,
                      Visit&& visit) {
  const int stride = rows + 1;
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
    visit(flipped, heights, false);
  }

  const int targetStride = columns + 1;   // the rotated board's rows
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
    visit(rotated, rotatedHeights, true);
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
    visit(rotated, rotatedHeights, true);
  }
}

// Whichever of a board and its mirror image comes first: its colour word,
// each column's mover bits packed bottom-up, columns left to right, and its
// height vector.
struct CanonicalOrientation {
  std::uint64_t colours;
  const int* heights;
};

// Height vectors decide first (leftmost column most significant, smaller
// height first), then colour segments from the rightmost column's top. Both
// solvers' slots follow that order, so encoding only this orientation matches
// min(encode(board), encode(mirror(board))). The heights are the board's own,
// or the mirror image's written to `mirroredHeights`.
//
// This runs for every edge of every state. It copies heights only for the
// mirror image, and takes the mover mask alone, the only one it reads, so the
// compiler can pass it in a register instead of storing both masks first.
inline CanonicalOrientation canonicalOrientation(std::uint64_t mover, const int* heights,
                                                 int rows, int columns,
                                                 int* mirroredHeights) {
  const int stride = rows + 1;
  int order = 0;   // negative: direct orientation first; positive: mirror
  for (int column = 0; order == 0 && column < columns; ++column) {
    const int direct = heights[column];
    const int mirrored = heights[columns - 1 - column];
    if (direct != mirrored) order = direct < mirrored ? -1 : 1;
  }
  if (order == 0) {
    for (int column = columns - 1; order == 0 && column >= 0; --column) {
      const std::uint64_t width = (std::uint64_t{1} << heights[column]) - 1;
      const std::uint64_t direct = (mover >> (column * stride)) & width;
      const std::uint64_t mirrored = (mover >> ((columns - 1 - column) * stride)) & width;
      if (direct != mirrored) order = direct > mirrored ? 1 : -1;
    }
  }

  std::uint64_t colours = 0;
  int offset = 0;
  if (order <= 0) {
    for (int column = 0; column < columns; ++column) {
      const std::uint64_t width = (std::uint64_t{1} << heights[column]) - 1;
      colours |= ((mover >> (column * stride)) & width) << offset;
      offset += heights[column];
    }
    return {colours, heights};
  }
  for (int column = 0; column < columns; ++column) {
    const int source = columns - 1 - column;
    mirroredHeights[column] = heights[source];
    const std::uint64_t width = (std::uint64_t{1} << heights[source]) - 1;
    colours |= ((mover >> (source * stride)) & width) << offset;
    offset += heights[source];
  }
  return {colours, mirroredHeights};
}

// The packed value of a state not settled yet.
constexpr std::uint8_t VALUE_UNKNOWN = 3;

// Two bits per state: LOSS 0, DRAW 1, WIN 2, UNKNOWN 3. The only transition
// is UNKNOWN -> settled, which clears bits, so concurrent publication is a
// release fetch_and on the shared word and never disturbs the other thirty-
// one states packed beside it.
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
    return (atomicLoad(words_[at >> 5], std::memory_order_acquire) >> ((at & 31) * 2)) & 3;
  }

  // Requires the current value to be UNKNOWN (all ones in the field).
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

// A layer or block is stored as two files, read and written in chunks
// (checkpoint-io.hpp): the bitset of its reachable slots, and its solved
// values one byte per state. Both open with a 32-byte header that each solver
// defines. Its last word is the CRC-32 of everything after the header, and a
// file whose header differs anywhere else from the one expected counts as
// missing.
template <typename Header>
bool headerMatches(const Header& seen, const Header& want) {
  static_assert(std::has_unique_object_representations_v<Header>,
                "a header without padding compares as its bytes");
  static_assert(offsetof(Header, crc) + sizeof(Header::crc) == sizeof(Header),
                "the checksum is a header's last word");
  return std::memcmp(&seen, &want, offsetof(Header, crc)) == 0;
}

// Writes a reachable-slot bitset under header, which gets its checksum here.
template <typename Header, typename Bits>
void writeBitsFile(const std::string& target, Header header, const Bits& bits) {
  const std::string temporary = target + ".tmp";
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("could not open " + temporary);
  header.crc = crc32(bits.words().data(), bits.wordCount() * sizeof(std::uint64_t));
  if (!writeAll(out, &header, sizeof(header))
      || !writeAll(out, bits.words().data(), bits.wordCount() * sizeof(std::uint64_t))) {
    throw std::runtime_error("could not write " + temporary);
  }
  publishDurably(out, temporary, target);
}

// Loads a bitset written under `want` and builds its rank directory. False
// when the file is missing, or fails a check, which is logged under `tag`.
template <typename Header, typename Bits>
bool loadBitsFile(const std::string& path, const Header& want, const char* tag, Bits& bits) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  Header seen{};
  if (!readExact(in, &seen, sizeof(seen)) || !headerMatches(seen, want)) {
    std::cerr << '[' << tag << "] rejecting " << path << std::endl;
    return false;
  }
  const std::size_t bytes = bits.wordCount() * sizeof(std::uint64_t);
  if (!readExact(in, bits.mutableWords().data(), bytes)) {
    std::cerr << '[' << tag << "] short read on " << path << std::endl;
    return false;
  }
  if (crc32(bits.words().data(), bytes) != seen.crc) {
    std::cerr << '[' << tag << "] rejecting " << path << ": checksum mismatch" << std::endl;
    return false;
  }
  bits.finalize();
  return true;
}

template <typename Header>
void writeValuesFile(const std::string& target, Header header, const PackedValues& values) {
  const std::string temporary = target + ".tmp";
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("could not open " + temporary);
  if (!writeAll(out, &header, sizeof(header))) {
    throw std::runtime_error("could not write " + temporary);
  }
  Crc32 crc;
  std::vector<std::uint8_t> staging;
  const std::uint64_t chunk = std::uint64_t{64} << 20;
  for (std::uint64_t begin = 0; begin < values.size(); begin += chunk) {
    const std::uint64_t count = std::min(chunk, values.size() - begin);
    staging.resize(count);
    for (std::uint64_t index = 0; index < count; ++index) {
      staging[index] = values.get(begin + index);
    }
    crc.update(staging.data(), count);
    if (!writeAll(out, staging.data(), count)) {
      throw std::runtime_error("could not write " + temporary);
    }
  }
  // The checksum is known only now: rewrite the header in place.
  header.crc = crc.value();
  out.seekp(0);
  if (!writeAll(out, &header, sizeof(header))) {
    throw std::runtime_error("could not write " + temporary);
  }
  publishDurably(out, temporary, target);
}

// On any failure the values are reset to unknown, so a caller that goes on to
// solve the layer or block never inherits part of a rejected file.
template <typename Header>
bool loadValuesFile(const std::string& path, const Header& want, const char* tag,
                    PackedValues& values) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  Header seen{};
  if (!readExact(in, &seen, sizeof(seen)) || !headerMatches(seen, want)) {
    std::cerr << '[' << tag << "] rejecting " << path << std::endl;
    return false;
  }
  const auto reject = [&](const char* reason) {
    std::cerr << '[' << tag << "] rejecting " << path << ": " << reason << std::endl;
    values.assign(values.size(), VALUE_UNKNOWN);
    return false;
  };
  Crc32 crc;
  std::vector<std::uint8_t> staging;
  const std::uint64_t chunk = std::uint64_t{64} << 20;
  for (std::uint64_t begin = 0; begin < values.size(); begin += chunk) {
    const std::uint64_t count = std::min(chunk, values.size() - begin);
    staging.resize(count);
    if (!readExact(in, staging.data(), count)) return reject("short read");
    crc.update(staging.data(), count);
    for (std::uint64_t index = 0; index < count; ++index) {
      // A solved layer or block holds LOSS, DRAW or WIN only; 3 would read
      // as unknown.
      if (staging[index] >= VALUE_UNKNOWN) return reject("value byte out of range");
      values.publish(begin + index, staging[index]);
    }
  }
  if (crc.value() != seen.crc) return reject("checksum mismatch");
  return true;
}

}  // namespace connect4
