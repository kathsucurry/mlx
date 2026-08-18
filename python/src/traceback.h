#pragma once

#include <nanobind/nanobind.h>

#include "mlx/traceback.h"

/* Installs the Python stack-capture hook into core. Called once at module
 * init. */
void install_traceback_capture();

/* Starts a new recording session: expire ids from previous sessions and
 * clears the interning table. GIL must be held. */
void start_new_traceback_session();

/* Resolves an id to a list of (filename, function, line) tuples, outermost
 * first. Returns None for no_traceback or ids from an old session. GIL must
 * be held. */
nanobind::object resolve_traceback(mlx::core::detail::TracebackId id);
