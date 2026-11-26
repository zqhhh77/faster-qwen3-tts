from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .batching import DepthBatcher


class DepthBatcherTests(unittest.TestCase):
    def test_pairs_keep_each_requests_inputs_outputs_and_eos_flag(self):
        def paired_run(outputs, feed):
            self.assertFalse(feed["r0/allow_eos"][0])
            self.assertTrue(feed["r1/allow_eos"][0])
            return [feed["r0/noise"], feed["r0/hidden"], feed["r1/noise"], feed["r1/hidden"]]

        with patch("backend.m5.batching.Thread.start"):
            batcher = DepthBatcher(None, SimpleNamespace(run=paired_run))
        self.addCleanup(batcher.close)
        futures = [Future(), Future()]
        for index, future in enumerate(futures):
            batcher.enter()
            batcher.queue.put((dict(hidden=np.array([index + 10]), noise=np.array([index + 20]),
                                   allow_eos=np.array([bool(index)])), future))
        batcher.worker.start()
        for index, future in enumerate(futures):
            code, hidden = future.result(timeout=2)
            np.testing.assert_array_equal(code, [index + 20])
            np.testing.assert_array_equal(hidden, [index + 10])
            batcher.leave()

    def test_pair_failure_reaches_both_requests_and_single_request_recovers(self):
        def fail(*args):
            raise RuntimeError("模拟推理失败")

        with patch("backend.m5.batching.Thread.start"):
            batcher = DepthBatcher(SimpleNamespace(run=lambda outputs, feed: [feed["value"]]),
                                   SimpleNamespace(run=fail))
        self.addCleanup(batcher.close)
        futures = [Future(), Future()]
        for future in futures:
            batcher.enter()
            batcher.queue.put(({"value": 1}, future))
        batcher.worker.start()
        for future in futures:
            with self.assertRaisesRegex(RuntimeError, "模拟推理失败"):
                future.result(timeout=2)
            batcher.leave()
        self.assertEqual(batcher.run(None, {"value": 7}), [7])

    def test_close_drains_running_request_and_rejects_new_work(self):
        started, release = Event(), Event()

        def single_run(outputs, feed):
            started.set()
            if not release.wait(2):
                raise RuntimeError("测试未释放推理")
            return [feed["value"]]

        batcher = DepthBatcher(SimpleNamespace(run=single_run), None)
        self.addCleanup(batcher.close)
        with ThreadPoolExecutor(max_workers=2) as pool:
            result = pool.submit(batcher.run, None, {"value": 3})
            self.assertTrue(started.wait(2))
            closing = pool.submit(batcher.close)
            release.set()
            self.assertEqual(result.result(timeout=2), [3])
            closing.result(timeout=2)
        self.assertFalse(batcher.worker.is_alive())
        with self.assertRaisesRegex(RuntimeError, "已停止"):
            batcher.run(None, {})
