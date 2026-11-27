from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .cuda_graph import CudaGraph


class CudaGraphTests(unittest.TestCase):
    def test_shared_buffers_run_on_one_thread_and_results_are_independent(self):
        threads = []

        def initialize(graph, model):
            threads.append(get_ident())
            state = np.zeros(1, np.float32)

            def update(value):
                threads.append(get_ident())
                state[:] = value

            graph.inputs = {"value": SimpleNamespace(update_inplace=update)}
            graph.outputs = [SimpleNamespace(numpy=lambda: state.copy() * 2)]
            graph.io = object()
            graph.session = SimpleNamespace(run_with_iobinding=lambda io: threads.append(get_ident()))

        with patch.object(CudaGraph, "_initialize", initialize):
            graph = CudaGraph(None)
        try:
            with ThreadPoolExecutor(max_workers=2) as callers:
                tasks = [callers.submit(graph.run, None, {"value": np.array([i], np.float32)}) for i in (3, 7)]
                values = [task.result(timeout=2)[0] for task in tasks]
            np.testing.assert_array_equal(values[0], [6])
            np.testing.assert_array_equal(values[1], [14])
            self.assertEqual(len(set(threads)), 1)
            self.assertNotEqual(threads[0], get_ident())
            with self.assertRaisesRegex(ValueError, "输入字段"):
                graph.run(None, {"wrong": np.array([1], np.float32)})
            np.testing.assert_array_equal(graph.run(None, {"value": np.array([9], np.float32)})[0], [18])
        finally:
            graph.close()
        with self.assertRaises(RuntimeError):
            graph.run(None, {"value": np.array([1], np.float32)})

    def test_capture_failure_is_propagated_and_executor_closed(self):
        with patch.object(CudaGraph, "_initialize", side_effect=RuntimeError("捕获失败")), \
                patch("backend.m5.cuda_graph.ThreadPoolExecutor.shutdown", autospec=True,
                      side_effect=ThreadPoolExecutor.shutdown) as shutdown:
            with self.assertRaisesRegex(RuntimeError, "捕获失败"):
                CudaGraph(None)
            shutdown.assert_called_once()
