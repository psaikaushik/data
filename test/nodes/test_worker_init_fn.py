# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import unittest

from torch.testing._internal.common_utils import IS_WINDOWS, TestCase
from torchdata.nodes.map import ParallelMapper

from .utils import MockSource


class TestWorkerInitFn(TestCase):
    def test_thread_workers_init_fn_called(self) -> None:
        """worker_init_fn is called once per thread worker with the correct id."""
        num_workers = 3
        called_ids = []
        lock = threading.Lock()

        def init_fn(worker_id: int) -> None:
            with lock:
                called_ids.append(worker_id)

        src = MockSource(num_samples=30)
        node = ParallelMapper(src, lambda x: x, num_workers=num_workers, worker_init_fn=init_fn)
        node.reset()
        list(node)

        self.assertEqual(len(called_ids), num_workers)
        self.assertEqual(sorted(called_ids), list(range(num_workers)))

    def test_inline_no_init_fn_called(self) -> None:
        """worker_init_fn is NOT called when num_workers=0."""
        called = []

        def init_fn(worker_id: int) -> None:
            called.append(worker_id)

        src = MockSource(num_samples=10)
        node = ParallelMapper(src, lambda x: x, num_workers=0, worker_init_fn=init_fn)
        node.reset()
        list(node)

        self.assertEqual(called, [])

    def test_none_init_fn_works(self) -> None:
        """worker_init_fn=None (default) still processes all items correctly."""
        n = 20
        src = MockSource(num_samples=n)
        node = ParallelMapper(src, lambda x: x, num_workers=2)
        node.reset()
        results = list(node)
        self.assertEqual(len(results), n)

    def test_init_fn_sets_state_visible_to_map_fn(self) -> None:
        """State set by worker_init_fn in a thread-local is visible to map_fn."""
        import threading

        _local = threading.local()

        def init_fn(worker_id: int) -> None:
            _local.initialized = True
            _local.worker_id = worker_id

        def map_fn(x):
            x["initialized"] = getattr(_local, "initialized", False)
            x["map_worker_id"] = getattr(_local, "worker_id", -1)
            return x

        num_workers = 2
        src = MockSource(num_samples=20)
        node = ParallelMapper(src, map_fn, num_workers=num_workers, worker_init_fn=init_fn)
        node.reset()
        results = list(node)

        for r in results:
            self.assertTrue(r["initialized"])
            self.assertIn(r["map_worker_id"], range(num_workers))

    @unittest.skipIf(IS_WINDOWS, "forkserver not supported on Windows")
    def test_process_workers_init_fn_called(self) -> None:
        """worker_init_fn is called in each process worker (functional check)."""
        n = 20
        src = MockSource(num_samples=n)
        node = ParallelMapper(
            src,
            lambda x: x,
            num_workers=2,
            method="process",
            multiprocessing_context="forkserver",
            worker_init_fn=lambda worker_id: None,
        )
        node.reset()
        results = list(node)
        self.assertEqual(len(results), n)

    def test_init_fn_exception_propagates(self) -> None:
        """An exception in worker_init_fn propagates to the caller."""

        def bad_init(worker_id: int) -> None:
            raise RuntimeError("init failed")

        src = MockSource(num_samples=10)
        node = ParallelMapper(src, lambda x: x, num_workers=2, worker_init_fn=bad_init)
        node.reset()
        with self.assertRaises(Exception):
            list(node)
