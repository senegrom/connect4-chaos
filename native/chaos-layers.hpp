// What the two layer-based Chaos solvers (perfect-chaos-layered and
// perfect-chaos-paired) share: positions as bitboard masks, the stack reversal
// a flip applies, line detection on a mask, and values packed two bits to a
// state.
//
// A mask has bit column * (rows + 1) + row, one guard bit per column so run
// detection cannot wrap between columns. Gravity keeps every column's pieces
// contiguous from the bottom, so a column's height is the size of its occupied
// segment and transformations move whole segments. Both solvers cap boards
// at 7x7, which fits 64 bits with the guards.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

#include "atomic-load.hpp"

namespace connect4 {

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

}  // namespace connect4
