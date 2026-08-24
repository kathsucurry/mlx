# Copyright © 2023-2024 Apple Inc.

import pickle
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx_tests
from mlx._memory_viz import dump_snapshot, to_snapshot

VIZ_ALLOC_ACTIONS = {"alloc", "segment_alloc"}
VIZ_FREE_ACTIONS = {"free_completed", "segment_free"}


def create_event(
    action,
    addr=0x1000,
    size=4096,
    timestamp=0,
    elapsed=0,
    traceback=None,
    primitive="",
    stream=0,
):
    return {
        "addr": addr,
        "size": size,
        "requested_size": size,
        "timestamp_us": timestamp,
        "elapsed_us": elapsed,
        "action": action,
        "primitive_name": primitive,
        "stream": stream,
        "traceback": traceback,
        "user_metadata": {"primitive": primitive or "unknown"},
    }


class TestMemoryViz(mlx_tests.MLXTestCase):
    def test_snapshot_keys(self):
        # Currently only includes device_traces and segments field.
        snapshot = to_snapshot([])
        self.assertEqual(set(snapshot.keys()), {"device_traces", "segments"})
        self.assertEqual(snapshot["device_traces"], [[]])
        self.assertEqual(snapshot["segments"], [])

    def test_timeline_action_mapping(self):
        # Each MLX action feeds one or both viz timelines: the active timeline
        # (alloc/free_completed) and the reserved timeline (segment_*).
        expected = {
            "AllocNew": ["segment_alloc", "alloc"],
            "AllocMakeBuffer": ["segment_alloc", "alloc"],
            "AllocReuse": ["alloc"],
            "FreeActiveToCache": ["free_completed"],
            "FreeActiveToOS": ["free_completed", "segment_free"],
            "Release": ["free_completed", "segment_free"],
            "FreeCacheToOS": ["segment_free"],
        }
        for mlx_action, viz_actions in expected.items():
            trace = to_snapshot([create_event(mlx_action)])["device_traces"][0]
            self.assertEqual(
                [event["action"] for event in trace], viz_actions, mlx_action
            )

    def test_unknown_actions_are_skipped(self):
        snapshot = to_snapshot(
            [create_event("Unknown"), create_event("NewFutureAction")]
        )
        self.assertEqual(snapshot["device_traces"][0], [])

    def test_entry_fields(self):
        trace = to_snapshot(
            [create_event("AllocReuse", addr=0x2000, size=8192, timestamp=42)]
        )
        (entry,) = trace["device_traces"][0]
        self.assertEqual(entry["addr"], 0x2000)
        self.assertEqual(entry["size"], 8192)
        self.assertEqual(entry["time_us"], 42)
        self.assertEqual(entry["stream"], 0)
        self.assertEqual(entry["frames"], [])
        self.assertEqual(entry["user_metadata"], {"primitive": "unknown"})

    def test_frames_shape_and_order(self):
        # resolve_traceback() returns (filename, function, line) tuples with
        # the outermost frame first (Python's traceback convention). The viz
        # prints frames in the reversed order, so the converter must reverse.
        traceback = [
            ("train.py", "main", 10),
            ("model.py", "Model.forward", 42),
        ]
        trace = to_snapshot([create_event("AllocReuse", traceback=traceback)])
        (entry,) = trace["device_traces"][0]
        self.assertEqual(
            entry["frames"],
            [
                {"filename": "model.py", "line": 42, "name": "Model.forward"},
                {"filename": "train.py", "line": 10, "name": "main"},
            ],
        )

    def test_missing_traceback_is_empty_frames(self):
        for traceback in (None, []):
            trace = to_snapshot([create_event("AllocReuse", traceback=traceback)])
            (entry,) = trace["device_traces"][0]
            self.assertEqual(entry["frames"], [])

    def test_category_from_primitive_name(self):
        trace = to_snapshot([create_event("AllocReuse", primitive="Matmul")])
        (entry,) = trace["device_traces"][0]
        self.assertEqual(entry["category"], "Matmul")
        self.assertEqual(entry["user_metadata"], {"primitive": "Matmul"})

        # Empty primitive names fall back to the viz's default category.
        trace = to_snapshot([create_event("AllocReuse", primitive="")])
        (entry,) = trace["device_traces"][0]
        self.assertEqual(entry["category"], "unknown")
        self.assertEqual(entry["user_metadata"], {"primitive": "unknown"})

    def test_both_timelines_pair_allocs_with_frees(self):
        # The PyTorch's viz matches frees to allocs by address, separately per timeline.
        # An unmatched free becomes a phantom element attributed to the free
        # site, so every free the converter emits must follow a matching alloc.
        # If the corresponding allocation is performed before the trace, the viz
        # tolerates this by rendering it as initially-allocated. So only check
        # the buffers whose alloc is in the trace.
        events = [
            create_event("AllocNew", addr=0x1000, timestamp=1),
            create_event("AllocNew", addr=0x2000, timestamp=2),
            create_event("FreeActiveToCache", addr=0x1000, timestamp=3),
            create_event("AllocReuse", addr=0x1000, timestamp=4),
            create_event("FreeActiveToOS", addr=0x2000, timestamp=5),
            create_event("Release", addr=0x1000, timestamp=6),
            create_event("FreeCacheToOS", addr=0x3000, timestamp=7),
        ]
        for alloc, free in (
            ("alloc", "free_completed"),
            ("segment_alloc", "segment_free"),
        ):
            live = set()
            for entry in to_snapshot(events)["device_traces"][0]:
                if entry["action"] == alloc:
                    self.assertNotIn(entry["addr"], live)
                    live.add(entry["addr"])
                elif entry["action"] == free and entry["addr"] in live:
                    live.discard(entry["addr"])
            self.assertEqual(live, set(), alloc)

    def test_dump_snapshot(self):
        events = [
            create_event("AllocNew", timestamp=1),
            create_event("FreeActiveToOS", timestamp=2),
        ]
        with tempfile.TemporaryDirectory() as d:
            path = dump_snapshot(events, Path(d) / "snap.pickle")
            with open(path, "rb") as f:
                loaded = pickle.load(f)
        self.assertEqual(loaded, to_snapshot(events))

    @unittest.skipIf(
        not mx.metal.is_available(), "Memory events recording are Metal only"
    )
    def test_converts_real_events(self):
        self.addCleanup(mx.record_memory_events, False)
        mx.synchronize()
        mx.clear_cache()
        mx.record_memory_events(True)

        a = mx.zeros((4096,))
        mx.eval(a)
        del a
        mx.synchronize()
        mx.clear_cache()

        events = mx.get_memory_events()
        self.assertGreater(len(events), 0)

        trace = to_snapshot(events)["device_traces"][0]
        self.assertGreater(len(trace), 0)
        seen = set()
        for entry in trace:
            seen.add(entry["action"])
            self.assertIn(entry["action"], VIZ_ALLOC_ACTIONS | VIZ_FREE_ACTIONS)
            self.assertIsInstance(entry["addr"], int)
            self.assertGreater(entry["size"], 0)
            self.assertGreaterEqual(entry["time_us"], 0)
            self.assertIsInstance(entry["frames"], list)
        # alloc + free + clear_cache must populate both timelines.
        self.assertIn("alloc", seen)
        self.assertIn("segment_alloc", seen)
        self.assertIn("free_completed", seen)
        self.assertIn("segment_free", seen)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
