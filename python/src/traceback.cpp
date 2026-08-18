#include <cstdint>
#include <unordered_map>
#include <vector>

#include <Python.h>
#include <nanobind/nanobind.h>

#include "python/src/traceback.h"

namespace nb = nanobind;

namespace {

using mlx::core::detail::no_traceback;
using mlx::core::detail::TracebackId;

// The maximum number of innermost frames kept per capture.
constexpr size_t max_depth{32};

// Recall that TracebackId is 32-bit. We use the top 8 bits to indicate session
// id and the bottom 24 bits to indicate entry id. So traceback ID is
// ((session_id << 24) | index) and the max number of entries at once is (1 <<
// 24) - 1.
constexpr int entry_bit_num{24};
// Currently bound the number of entries in the interned stack per session.
constexpr uint32_t max_entries{(1 << entry_bit_num) - 1};
constexpr uint32_t index_mask = max_entries;

struct Frame {
  PyObject* code; // Strong ref, held until the session resets
  int line;
  bool operator==(const Frame&) const = default;
};

// Interned stacks with innermost frame first. Slot 0 is reserved since id 0
// is used for no_traceback.
std::vector<std::vector<Frame>> stacks{{}};
std::unordered_map<uint64_t, std::vector<uint32_t>> stacks_by_hash;

// Used to distinguish the session. When an old session is terminated,
// the interned stack is cleared but the arrays created during the old session
// would still hold the ids (i.e., dangling reference). session_id is used to
// resolve this issue.
uint32_t session_id = 1;

uint64_t hash_frames(const std::vector<Frame>& frames) {
  // Perform Fowler-Noll-Vo hash function (the hash method was arbitrarily
  // chosen).
  constexpr uint64_t FNV_prime = 1099511628211ull;
  uint64_t h = 14695981039346656037ull;
  for (const auto& f : frames) {
    h = (h ^ reinterpret_cast<uintptr_t>(f.code)) * FNV_prime;
    h = (h ^ static_cast<uint64_t>(f.line)) * FNV_prime;
  }
  return h;
}

TracebackId capture_traceback_impl() {
  // Check if it's safe to capture, otherwise skip.
  // It's safe if the interpreter is alive and THIS thread holds the GIL
  // (i.e., we're in a Python call, not an eval worker thread).
  if (!Py_IsInitialized() || !PyGILState_Check())
    return no_traceback;

  std::vector<Frame> frames;
  frames.reserve(max_depth);
  PyFrameObject* frame = PyEval_GetFrame();
  Py_XINCREF(frame);

  while (frame && frames.size() < max_depth) {
    frames.push_back(
        {.code = (PyObject*)PyFrame_GetCode(frame),
         .line = PyFrame_GetLineNumber(frame)});
    PyFrameObject* back = PyFrame_GetBack(frame);
    Py_DECREF(frame);
    frame = back;
  }
  Py_XDECREF(frame);

  auto release_scratch = [&]() {
    for (auto& f : frames)
      Py_DECREF(f.code);
  };

  auto h = hash_frames(frames);
  if (auto it = stacks_by_hash.find(h); it != stacks_by_hash.end()) {
    for (auto index : it->second) {
      // Frame stack is already included in the interned stack.
      if (stacks[index] == frames) {
        release_scratch();
        return (session_id << entry_bit_num) | index;
      }
    }
  }

  // Frames are new but the interned stack is full.
  if (stacks.size() > max_entries) {
    release_scratch();
    return no_traceback;
  }

  // Frames are new and can be added into the stack.
  auto index = static_cast<uint32_t>(stacks.size());
  stacks.push_back(std::move(frames));
  stacks_by_hash[h].push_back(index);
  return (session_id << entry_bit_num) | index;
}

} // namespace

void install_traceback_capture() {
  mlx::core::detail::set_traceback_capture_func(&capture_traceback_impl);
}

void start_new_traceback_session() {
  session_id = (session_id % ((1 << (32 - entry_bit_num)) - 1)) + 1;
  for (auto& stack : stacks)
    for (auto& f : stack)
      Py_DECREF(f.code);
  // Index 0 is reserved for no_traceback.
  stacks.assign(1, {});
  stacks_by_hash.clear();
}

nb::object resolve_traceback(TracebackId id) {
  if (id == no_traceback || (id >> entry_bit_num) != session_id)
    return nb::none(); // never captured or expired session
  auto index = id & index_mask;

  nb::list out;
  const auto& frames = stacks[index];
  // The innermost is stored first, so we want to emit the outermost to
  // match Python tracebacks.
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    auto code = nb::borrow(it->code);
#if PY_VERSION_HEX >= 0x030B0000
    auto name = code.attr("co_qualname"); // "Model.forward"
#else
    auto name = code.attr("co_name"); // no qualname before 3.11
#endif
    out.append(nb::make_tuple(code.attr("co_filename"), name, it->line));
  }
  return out;
}
