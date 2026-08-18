#pragma once

#include <cstdint>

#include "mlx/api.h"

namespace mlx::core::detail {

/* A handle to a captured call site. */
using TracebackId = uint32_t;
inline constexpr TracebackId no_traceback = 0;

/* A hook for capturing the call site that created an array. */
using TracebackCaptureFunc = TracebackId (*)();

MLX_API void set_traceback_tracking(bool enabled);

/* Installs the capture hook. */
MLX_API void set_traceback_capture_func(TracebackCaptureFunc func);

/** Invokes the installed capture hook to obtain the index to the stack
 * in the traceback table, or returns no_traceback if no hook is installed. */
MLX_API TracebackId capture_traceback();

} // namespace mlx::core::detail
