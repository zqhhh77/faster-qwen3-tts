from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, Mock, patch
import subprocess
import sys

import psutil

import grpc

from backend.m5.generated import inference_pb2 as pb
from backend.m5.generated import inference_pb2_grpc as rpc
from backend.m5.tune import batch, combinations, corpus, recommend, resolve_requests, stream_request, summarize
from backend.m5.processes import ProcessTree


class Inference(rpc.InferenceServiceServicer):
    def __init__(self):
        self.lock = threading.Lock()
        self.active = self.maximum = 0

    def Synthesize(self, request, context):
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        try:
            yield pb.SynthesizeEvent(audio=b"\x00\x00" * 240)
            threading.Event().wait(.03)
            if request.input == "fail":
                context.abort(grpc.StatusCode.INTERNAL, "推理失败")
            yield pb.SynthesizeEvent(done=pb.Completion(samples=240))
        finally:
            with self.lock:
                self.active -= 1


class TuneTests(unittest.TestCase):
    def test_process_tree_counts_child_memory_and_stops_children(self):
        child_code = "memory=bytearray(24*1024*1024); print('ready',flush=True); import time; time.sleep(30)"
        parent_code = ("import subprocess,sys,time; "
                       f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}],stdout=subprocess.PIPE,text=True); "
                       "print(child.stdout.readline(),flush=True); time.sleep(30)")
        process = subprocess.Popen([sys.executable, "-c", parent_code], stdout=subprocess.PIPE, text=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        tree = ProcessTree(process)
        try:
            self.assertEqual(process.stdout.readline().strip(), "ready")
            members = tree.members()
            self.assertGreaterEqual(len(members), 2)
            self.assertGreater(tree.resident_bytes(), 24 * 1024**2)
        finally:
            tree.stop()
            process.stdout.close()
        self.assertTrue(all(not process.is_running() or process.status() == psutil.STATUS_ZOMBIE for process in members))

    def test_candidates_respect_requested_concurrency_and_cpu_budget(self):
        for concurrency in (1, 3, 6, 10, 64):
            rows = combinations(concurrency, 8)
            self.assertTrue(rows)
            self.assertTrue(all(1 <= w <= min(concurrency, 32) and w * t <= 8 for w, t in rows))
        self.assertEqual(combinations(6, 6, [1, 2, 6], [1, 2]), [(1, 1), (1, 2), (2, 1), (2, 2), (6, 1)])

    def test_same_text_with_different_voices_is_distinct(self):
        info = pb.ModelInfo(voices=["a", "b"], default_voice="a", languages=["Chinese"])
        requests = [{"input": "你好", "voice": "a"}, {"input": "你好", "voice": "b"}]
        self.assertEqual(len(resolve_requests(requests, info, None, "chinese")), 2)
        with self.assertRaises(ValueError):
            resolve_requests([{"input": "你好"}, {"input": "你好", "voice": "a"}], info, None, "Chinese")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.json"
            path.write_text(json.dumps(requests), encoding="utf-8")
            self.assertEqual(corpus(None, 2, str(path)), requests)
            path.write_text('[{"input": ""}]', encoding="utf-8")
            with self.assertRaises(ValueError):
                corpus(None, 1, str(path))
        self.assertEqual(len(corpus(None, 10)), 10)

    def test_recommendation_excludes_failures_and_uses_throughput_first(self):
        good = dict(ok=True, requests_per_s=2, first_p95_s=3, peak_rss_mb=100, underruns=0)
        faster_but_failed = dict(good, ok=False, requests_per_s=10)
        lower_latency = dict(good, requests_per_s=1, first_p95_s=.1)
        self.assertIs(recommend([faster_but_failed, lower_latency, good]), good)
        self.assertIsNone(recommend([faster_but_failed]))
        stalled = dict(good, requests_per_s=4, underruns=1)
        self.assertIs(recommend([good, stalled]), stalled)
        self.assertIs(recommend([stalled, faster_but_failed, good], require_continuous=True), good)
        self.assertIsNone(recommend([stalled], require_continuous=True))

    def test_playback_consumes_buffer_and_counts_only_gaps_between_audio_events(self):
        chunk = pb.SynthesizeEvent(audio=b"\x00\x00" * 6000)  # 250 ms at 24 kHz.
        call = MagicMock()
        call.__iter__.return_value = [chunk] * 5 + [pb.SynthesizeEvent(done=pb.Completion(samples=30000))]
        call.code.return_value = grpc.StatusCode.OK
        stub = Mock()
        stub.Synthesize.return_value = call
        # Early second chunk, third exactly on time, then 250 ms and 125 ms without audio.
        with patch("backend.m5.tune.time.perf_counter", side_effect=[0, 3, 3.125, 3.5, 4, 4.375, 10]):
            row = stream_request(stub, pb.ModelInfo(identity="model"), "文本", "a", "Chinese", 20)
        self.assertEqual((row["underruns"], row["stall_s"], row["max_stall_s"]), (2, .375, .25))
        self.assertEqual(row["audio_s"], 1.25)

    def test_streaming_batch_bounds_execution_and_counts_queue_time(self):
        service = Inference()
        server = grpc.server(ThreadPoolExecutor(max_workers=8))
        rpc.add_InferenceServiceServicer_to_server(service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = rpc.InferenceServiceStub(channel)
                info = pb.ModelInfo(identity="a" * 64, concurrency=2)
                requests = [dict(input=f"任务{i}", voice="a", language="Chinese") for i in range(6)]
                result = batch(stub, info, requests, 3)
                self.assertTrue(all(row["ok"] for row in result["requests"]))
                self.assertEqual(service.maximum, 2)
                self.assertGreater(max(row["queue_s"] for row in result["requests"]), .02)
                summary = summarize(2, 1, [result], 1024**2, .5)
                self.assertEqual(summary["successes"], 6)
                self.assertEqual(summary["underruns"], 0)
                self.assertGreater(summary["first_p95_s"], .02)
                self.assertEqual(summary["peak_rss_mb"], 1)
                failed = batch(stub, info, [dict(input="fail", voice="a", language="Chinese")], 3)
                self.assertFalse(summarize(2, 1, [failed], 0, 0)["ok"])
                cancel = Mock(side_effect=channel.close)
                with patch("concurrent.futures.Future.result", side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        batch(stub, info, requests, 3, cancel)
                cancel.assert_called_once()
        finally:
            server.stop(0).wait()


if __name__ == "__main__":
    unittest.main()
