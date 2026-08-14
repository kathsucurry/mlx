# Copyright © 2023-2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests


class TestMemory(mlx_tests.MLXTestCase):
    def test_memory_info(self):
        old_limit = mx.set_cache_limit(0)

        a = mx.zeros((4096,))
        mx.eval(a)
        del a
        self.assertEqual(mx.get_cache_memory(), 0)
        self.assertEqual(mx.set_cache_limit(old_limit), 0)
        self.assertEqual(mx.set_cache_limit(old_limit), old_limit)

        old_limit = mx.set_memory_limit(10)
        self.assertEqual(mx.set_memory_limit(old_limit), 10)
        self.assertEqual(mx.set_memory_limit(old_limit), old_limit)

        # Query active and peak memory
        a = mx.zeros((4096,))
        mx.eval(a)
        mx.synchronize()
        active_mem = mx.get_active_memory()
        self.assertTrue(active_mem >= 4096 * 4)

        b = mx.zeros((4096,))
        mx.eval(b)
        del b
        mx.synchronize()

        new_active_mem = mx.get_active_memory()
        self.assertEqual(new_active_mem, active_mem)
        peak_mem = mx.get_peak_memory()
        self.assertTrue(peak_mem >= 4096 * 8)

        if mx.metal.is_available():
            cache_mem = mx.get_cache_memory()
            self.assertTrue(cache_mem >= 4096 * 4)

        mx.clear_cache()
        self.assertEqual(mx.get_cache_memory(), 0)

        mx.reset_peak_memory()
        self.assertEqual(mx.get_peak_memory(), 0)

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_wired_memory(self):
        old_limit = mx.set_wired_limit(1000)
        old_limit = mx.set_wired_limit(0)
        self.assertEqual(old_limit, 1000)

        max_size = mx.device_info(mx.gpu)["max_recommended_working_set_size"]
        with self.assertRaises(ValueError):
            mx.set_wired_limit(max_size + 10)

    def test_active_memory_count(self):
        mx.synchronize()
        mx.clear_cache()
        init_mem = mx.get_active_memory()
        a = mx.zeros((128, 128))
        mx.eval(a)
        mx.synchronize()
        del a
        a = mx.zeros((90, 128))
        mx.eval(a)
        mx.synchronize()
        del a
        self.assertEqual(init_mem, mx.get_active_memory())

    def test_memory_events_recording_available(self):
        # Currently only Metal actually records, but the function itself is callable on every backend.
        self.addCleanup(mx.record_memory_events, False)
        mx.record_memory_events(True)
        mx.eval(mx.zeros((4096,)))
        mx.synchronize()
        events = mx.get_memory_events()
        self.assertIsInstance(events, list)
        if not mx.metal.is_available():
            self.assertEqual(events, [])

    @unittest.skipIf(
        not mx.metal.is_available(), "Memory events recording are Metal only"
    )
    def test_memory_events_recording_schema(self):
        self.addCleanup(mx.record_memory_events, False)
        mx.synchronize()
        mx.clear_cache()

        mx.record_memory_events(True)
        a = mx.zeros((4096,))
        mx.eval(a)
        mx.synchronize()

        events = mx.get_memory_events()
        self.assertGreater(len(events), 0)
        for e in events:
            self.assertEqual(
                set(e.keys()),
                {
                    "buffer_ptr",
                    "size",
                    "requested_size",
                    "timestamp_us",
                    "action",
                    "primitive_name",
                },
            )
            self.assertGreater(e["buffer_ptr"], 0)
            self.assertGreater(e["size"], 0)
            self.assertGreaterEqual(e["timestamp_us"], 0)
            self.assertIsInstance(e["primitive_name"], str)
            # Unknown is never emitted; seeing it means a label is unmapped.
            self.assertNotEqual(e["action"], "Unknown")

        stamps = [e["timestamp_us"] for e in events]
        self.assertEqual(stamps, sorted(stamps))

    @unittest.skipIf(
        not mx.metal.is_available(), "Memory events recording are Metal only"
    )
    def test_memory_events_recording_toggle(self):
        self.addCleanup(mx.record_memory_events, False)

        mx.record_memory_events(True)
        mx.eval(mx.zeros((4096,)))
        mx.synchronize()
        self.assertGreater(len(mx.get_memory_events()), 0)

        # Disabling stops recording but keeps what was already collected.
        mx.record_memory_events(False)
        n = len(mx.get_memory_events())
        self.assertGreater(n, 0)
        mx.eval(mx.zeros((8192,)))
        mx.synchronize()
        self.assertEqual(len(mx.get_memory_events()), n)

        # Re-enabling clears the buffer.
        mx.record_memory_events(True)
        self.assertEqual(mx.get_memory_events(), [])

    @unittest.skipIf(
        not mx.metal.is_available(), "Memory events recording are Metal only"
    )
    def test_memory_events_recording_max_entries(self):
        self.addCleanup(mx.record_memory_events, False)

        mx.record_memory_events(True, 10)
        for i in range(64):
            mx.eval(mx.zeros((1024 + i,)))
        mx.synchronize()
        events = mx.get_memory_events()
        self.assertEqual(len(events), 10)

        stamps = [event["timestamp_us"] for event in events]
        self.assertEqual(stamps, sorted(stamps))

        # 0 indicates unlimited entries.
        mx.record_memory_events(True, 0)
        for i in range(64):
            mx.eval(mx.zeros((1024 + i,)))
        mx.synchronize()
        self.assertGreater(len(mx.get_memory_events()), 10)

    @unittest.skipIf(
        not mx.metal.is_available(), "Memory events recording are Metal only"
    )
    def test_memory_events_recording_actions(self):
        self.addCleanup(mx.record_memory_events, False)
        mx.synchronize()
        mx.clear_cache()
        mx.record_memory_events(True)

        # Allocate, release to the cache, then reallocate the same size.
        for _ in range(2):
            a = mx.zeros((4096,))
            mx.eval(a)
            mx.synchronize()
            del a
            mx.synchronize()

        actions = [e["action"] for e in mx.get_memory_events()]
        self.assertIn("AllocNew", actions)
        self.assertIn("FreeActiveToCache", actions)
        self.assertIn("AllocReuse", actions)

        # Clearing the cache releases cached buffers back to the OS.
        mx.clear_cache()
        actions = [e["action"] for e in mx.get_memory_events()]
        self.assertIn("FreeCacheToOS", actions)

    @unittest.skipIf(
        not mx.metal.is_available(), "Memory events recording are Metal only"
    )
    def test_memory_events_recording_buffer_lifetime(self):
        self.addCleanup(mx.record_memory_events, False)
        mx.synchronize()
        mx.clear_cache()
        mx.record_memory_events(True)

        for i in range(8):
            a = mx.zeros((4096 + i,))
            mx.eval(a)
            del a
        mx.synchronize()
        mx.clear_cache()

        # Replay the memory events: every buffer's transitions must be consistent.
        live_buffers = set()
        for event in mx.get_memory_events():
            ptr, action = event["buffer_ptr"], event["action"]
            if action in ("AllocNew", "AllocMakeBuffer"):
                self.assertNotIn(ptr, live_buffers)
                live_buffers.add(ptr)
                continue

            if ptr in live_buffers:  # only considers relevant buffers
                # If action == "AllocReuse", then the buffer is still alive.
                if action in ("FreeActiveToOS", "FreeCacheToOS", "Release"):
                    live_buffers.discard(ptr)
        self.assertEqual(live_buffers, set())

    @unittest.skipIf(
        not mx.metal.is_available(), "Memory events recording are Metal only"
    )
    def test_memory_events_recording_primitive_name(self):
        alloc_actions = {"AllocNew", "AllocReuse", "AllocMakeBuffer"}

        self.addCleanup(mx.record_memory_events, False)
        mx.synchronize()
        mx.clear_cache()
        mx.record_memory_events(True)

        # Use odd dimensions so the matmul output has a distinctive byte size.
        a = mx.zeros((257, 129))
        b = mx.zeros((129, 65))
        mx.eval(a, b)
        c = mx.matmul(a, b)
        mx.eval(c)
        mx.synchronize()

        events = mx.get_memory_events()
        self.assertGreater(len(events), 0)

        alloc_names = {
            event["primitive_name"]
            for event in events
            if event["action"] in alloc_actions
        }
        self.assertIn("Full", alloc_names)
        self.assertIn("Matmul", alloc_names)

        # The output buffer of the matmul is attributed to Matmul, not to
        # whichever op happened to run next.
        matmul_allocs = [
            event
            for event in events
            if event["action"] in alloc_actions
            and event["requested_size"] == 257 * 65 * 4
        ]
        self.assertGreater(len(matmul_allocs), 0)
        for event in matmul_allocs:
            self.assertEqual(event["primitive_name"], "Matmul")

        # Primitive names should not be available in memory frees.
        for event in events:
            if event["action"] not in alloc_actions:
                self.assertEqual(event["primitive_name"], "")


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
