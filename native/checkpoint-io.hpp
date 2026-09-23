// Durable, self-checking checkpoint files and exception-safe worker pools for
// the long-running exact solvers (perfect-chaos-complete, -layered, -paired).
//
// A solve can run for days across reboots, so every checkpoint it resumes
// from must be exactly what an earlier run wrote. Three things make that so:
//   * each file carries a CRC-32 of everything after its header, because a
//     power loss can leave a correctly sized file whose tail reads back as
//     zeros - and a packed LOSS is zero;
//   * a finished file is flushed to stable storage before it is renamed into
//     place, so a surviving name never points at blocks that were not written;
//   * a solver-format version in every header retires checkpoints written
//     before a fix that can change a stored bit or value.
#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace connect4 {

static_assert(std::endian::native == std::endian::little,
              "checkpoint files and their checksums are little-endian");

// IEEE CRC-32, the polynomial zlib uses, eight bytes per step. Updates chain,
// so a file can be checksummed in the chunks it is read or written in.
class Crc32 {
 public:
  void update(const void* data, std::size_t bytes) {
    const auto& table = tables();
    const auto* cursor = static_cast<const unsigned char*>(data);
    std::uint32_t crc = ~state_;
    while (bytes >= 8) {
      std::uint32_t low = 0;
      std::uint32_t high = 0;
      std::memcpy(&low, cursor, 4);
      std::memcpy(&high, cursor + 4, 4);
      low ^= crc;
      crc = table[7][low & 0xff] ^ table[6][(low >> 8) & 0xff]
          ^ table[5][(low >> 16) & 0xff] ^ table[4][low >> 24]
          ^ table[3][high & 0xff] ^ table[2][(high >> 8) & 0xff]
          ^ table[1][(high >> 16) & 0xff] ^ table[0][high >> 24];
      cursor += 8;
      bytes -= 8;
    }
    while (bytes-- > 0) crc = table[0][(crc ^ *cursor++) & 0xff] ^ (crc >> 8);
    state_ = ~crc;
  }
  std::uint32_t value() const { return state_; }

 private:
  using Tables = std::array<std::array<std::uint32_t, 256>, 8>;
  static const Tables& tables() {
    static const Tables built = [] {
      Tables table{};
      for (std::uint32_t index = 0; index < 256; ++index) {
        std::uint32_t crc = index;
        for (int bit = 0; bit < 8; ++bit) crc = (crc & 1) ? 0xEDB88320u ^ (crc >> 1) : crc >> 1;
        table[0][index] = crc;
      }
      for (std::uint32_t index = 0; index < 256; ++index) {
        for (int slice = 1; slice < 8; ++slice) {
          const std::uint32_t previous = table[slice - 1][index];
          table[slice][index] = (previous >> 8) ^ table[0][previous & 0xff];
        }
      }
      return table;
    }();
    return built;
  }
  std::uint32_t state_ = 0;
};

inline std::uint32_t crc32(const void* data, std::size_t bytes) {
  Crc32 crc;
  crc.update(data, bytes);
  return crc.value();
}

// Flushes a written file's data to stable storage.
inline void syncFile(const std::string& path) {
#ifdef _WIN32
  const int descriptor = _open(path.c_str(), _O_RDWR | _O_BINARY);
  if (descriptor < 0) throw std::runtime_error("could not reopen " + path + " to flush it");
  const bool synced = _commit(descriptor) == 0;
  _close(descriptor);
#else
  const int descriptor = ::open(path.c_str(), O_RDWR);
  if (descriptor < 0) throw std::runtime_error("could not reopen " + path + " to flush it");
  const bool synced = ::fsync(descriptor) == 0;
  ::close(descriptor);
#endif
  if (!synced) throw std::runtime_error("could not flush " + path + " to disk");
}

// Closes a finished file, checks the close - the last block is only written by
// the flush - flushes it to disk and only then renames it over target.
inline void publishDurably(std::ofstream& out, const std::string& temporary,
                           const std::string& target) {
  out.close();
  if (out.fail()) throw std::runtime_error("could not write " + temporary);
  syncFile(temporary);
  std::remove(target.c_str());
  if (std::rename(temporary.c_str(), target.c_str()) != 0) {
    throw std::runtime_error("could not publish " + target);
  }
}

// Runs work(thread, failed) on count threads and joins them all. An exception
// escaping a std::thread calls std::terminate, which ended a solve with no
// message; here the first one is captured, raises `failed` so the others can
// stop early, and is rethrown in the caller once every thread has joined.
template <typename Work>
void runThreads(int count, Work&& work) {
  std::vector<std::exception_ptr> errors(static_cast<std::size_t>(count));
  std::atomic<bool> failed{false};
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(count));
  try {
    for (int thread = 0; thread < count; ++thread) {
      pool.emplace_back([&, thread]() {
        try {
          work(thread, failed);
        } catch (...) {
          errors[static_cast<std::size_t>(thread)] = std::current_exception();
          failed.store(true, std::memory_order_relaxed);
        }
      });
    }
  } catch (...) {
    failed.store(true, std::memory_order_relaxed);
    for (std::thread& worker : pool) worker.join();
    throw;
  }
  for (std::thread& worker : pool) worker.join();
  for (const std::exception_ptr& error : errors) {
    if (error) std::rethrow_exception(error);
  }
}

}  // namespace connect4
