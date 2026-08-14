#pragma once

#include <atomic>
#include <string_view>

namespace mlx::core::detail {

struct OpInfo {
  std::string_view primitive_name;
};

// Stores the operation context for each thread.
inline thread_local OpInfo current_op;

inline std::atomic<bool> op_tracking_enabled{false};

inline bool op_tracking() {
  return op_tracking_enabled.load(std::memory_order_relaxed);
}

inline void set_op_tracking(bool enabled) {
  op_tracking_enabled.store(enabled, std::memory_order_relaxed);
}

struct OpContext {
  explicit OpContext(std::string_view primitive_name) : active(op_tracking()) {
    if (active)
      current_op = {.primitive_name = primitive_name};
  }
  ~OpContext() {
    if (active)
      current_op = {};
  }

  OpContext(const OpContext&) = delete;
  OpContext& operator=(const OpContext&) = delete;

 private:
  bool active;
};

} // namespace mlx::core::detail
