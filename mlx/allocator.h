// Copyright © 2023 Apple Inc.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "mlx/api.h"
#include "mlx/traceback.h"

namespace mlx::core::allocator {

// Simple wrapper around buffer pointers
// WARNING: Only Buffer objects constructed from and those that wrap
//          raw pointers from mlx::allocator are supported.
class MLX_API Buffer {
 private:
  void* ptr_;

 public:
  explicit Buffer(void* ptr) : ptr_(ptr) {};

  // Get the raw data pointer from the buffer
  void* raw_ptr();

  // Get the buffer pointer from the buffer
  const void* ptr() const {
    return ptr_;
  };
  void* ptr() {
    return ptr_;
  };
};

struct MLX_API MemoryEvent {
  enum Action {
    AllocNew, // Allocates from OS
    AllocReuse, // Reuse memory from cache
    AllocMakeBuffer, // Allocates from OS specifically for external memory
    FreeActiveToCache, // Free active memory to cache
    FreeActiveToOS, // Free active memory directly to OS
    Release, // Release active memory to OS specifically for external memory
    FreeCacheToOS, // Free memory from cache to OS to clear cache
    Unknown, // N/A
  };

  static constexpr bool is_alloc(Action action) {
    return action == AllocNew || action == AllocReuse ||
        action == AllocMakeBuffer;
  }

  const void* buffer_ptr{nullptr};
  size_t size{0};
  size_t requested_size{0};
  int64_t timestamp{0};
  Action action{Unknown};
  std::string primitive_name;
  detail::TracebackId traceback{detail::no_traceback};
};

class MLX_API Allocator {
  /** Abstract base class for a memory allocator. */
 public:
  virtual Buffer malloc(size_t size) = 0;
  virtual void free(Buffer buffer) = 0;
  virtual size_t size(Buffer buffer) const = 0;
  virtual Buffer make_buffer(void* ptr, size_t size) {
    return Buffer{nullptr};
  };
  virtual void release(Buffer buffer) {}
  virtual void record_memory_events(
      bool enabled /* = true */,
      size_t max_entries /* = 0 */) {}
  virtual std::vector<MemoryEvent> get_memory_events() {
    return {};
  }

  Allocator() = default;
  Allocator(const Allocator& other) = delete;
  Allocator(Allocator&& other) = delete;
  Allocator& operator=(const Allocator& other) = delete;
  Allocator& operator=(Allocator&& other) = delete;
  virtual ~Allocator() = default;
};

MLX_API Allocator& allocator();

inline Buffer malloc(size_t size) {
  return allocator().malloc(size);
}

inline void free(Buffer buffer) {
  allocator().free(buffer);
}

// Make a Buffer from a raw pointer of the given size without a copy.  If a
// no-copy conversion is not possible then the returned buffer.ptr() will be
// nullptr. Any buffer created with this function must be released with
// release(buffer)
inline Buffer make_buffer(void* ptr, size_t size) {
  return allocator().make_buffer(ptr, size);
};

// Release a buffer from the allocator made with make_buffer
inline void release(Buffer buffer) {
  allocator().release(buffer);
}

MLX_API bool can_reuse_alien_buffer(void* ptr);

} // namespace mlx::core::allocator
