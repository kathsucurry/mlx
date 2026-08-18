#include <atomic>

#include "mlx/traceback.h"

namespace mlx::core::detail {

static std::atomic<TracebackCaptureFunc> traceback_capture_func{nullptr};
static std::atomic<bool> traceback_tracking_enabled{false};

void set_traceback_tracking(bool enabled) {
  traceback_tracking_enabled.store(enabled, std::memory_order_relaxed);
}

void set_traceback_capture_func(TracebackCaptureFunc func) {
  traceback_capture_func.store(func, std::memory_order_relaxed);
}

TracebackId capture_traceback() {
  if (!traceback_tracking_enabled.load(std::memory_order_relaxed))
    return no_traceback;

  auto func = traceback_capture_func.load(std::memory_order_relaxed);

  return func ? func() : no_traceback;
}

} // namespace mlx::core::detail
