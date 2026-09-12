#ifndef CONNECT4_ATOMIC_LOAD_HPP
#define CONNECT4_ATOMIC_LOAD_HPP

#include <atomic>

namespace connect4 {

// Solver buffers are mutable storage, also exposed through const reader views.
// C++20 libc++ rejects atomic_ref<const T>::load. Instantiate atomic_ref<T>
// without weakening the load-only interface or changing its memory ordering.
// The cast creates the reference only; this helper never writes through it.
template <typename T>
T atomicLoad(const T& value, std::memory_order order) noexcept {
  return std::atomic_ref<T>(const_cast<T&>(value)).load(order);
}

}  // namespace connect4

#endif
