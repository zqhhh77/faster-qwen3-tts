from concurrent.futures import ThreadPoolExecutor
import threading
import unittest

import grpc

from backend.m5.generated import inference_pb2 as pb
from backend.m5.generated import inference_pb2_grpc as rpc
from backend.m5.server import Service


class FakeRuntime:
    identity = "a" * 64
    model_name = "m5-test"
    voices = {"default": {}}
    default_voice = "default"
    normalization = False

    def __init__(self):
        self.started = threading.Event()
        self.resume = threading.Event()
        self.closed = threading.Event()
        self.truncated = False

    def generate(self, text, voice, language):
        try:
            yield b"\x01\x00" * 4800
            self.started.set()
            self.resume.wait(3)
            if self.truncated:
                raise RuntimeError("truncated")
            yield b"\x02\x00" * 4800
        finally:
            self.closed.set()


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.start_service([self.runtime])
        self.request = pb.SynthesizeRequest(input="你好" * 300, voice="default", language="Chinese",
                                             speed=1, expected_identity=self.runtime.identity)

    def start_service(self, runtimes):
        self.server = grpc.server(ThreadPoolExecutor(max_workers=4))
        self.service = Service(runtimes)
        rpc.add_InferenceServiceServicer_to_server(self.service, self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        self.stub = rpc.InferenceServiceStub(self.channel)

    def tearDown(self):
        self.runtime.resume.set()
        self.channel.close()
        self.server.stop(0).wait()

    def test_streams_before_completion_and_rejects_concurrent_generation(self):
        call = self.stub.Synthesize(self.request, timeout=5)
        self.assertTrue(next(call).audio)
        with self.assertRaises(grpc.RpcError) as caught:
            next(self.stub.Synthesize(self.request, timeout=2))
        self.assertEqual(caught.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)
        self.assertTrue(self.stub.GetInfo(pb.GetInfoRequest(), timeout=2).voices)
        self.runtime.resume.set()
        events = list(call)
        self.assertEqual(events[-1].done.samples, 9600)
        self.assertEqual(call.code(), grpc.StatusCode.OK)

    def test_truncation_has_error_status_and_no_done(self):
        self.runtime.truncated = True
        self.runtime.resume.set()
        call = self.stub.Synthesize(self.request, timeout=5)
        self.assertTrue(next(call).audio)
        with self.assertRaises(grpc.RpcError) as caught:
            list(call)
        self.assertEqual(caught.exception.code(), grpc.StatusCode.INTERNAL)

    def test_cancel_closes_generator(self):
        call = self.stub.Synthesize(self.request, timeout=5)
        next(call)
        call.cancel()
        self.runtime.resume.set()
        self.assertTrue(self.runtime.closed.wait(3))

    def test_wrong_identity_rejected(self):
        self.request.expected_identity = "b" * 64
        with self.assertRaises(grpc.RpcError) as caught:
            next(self.stub.Synthesize(self.request, timeout=2))
        self.assertEqual(caught.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)

    def test_two_instances_generate_concurrently_and_release_slots(self):
        other = FakeRuntime()
        self.addCleanup(other.resume.set)
        self.channel.close()
        self.server.stop(0).wait()
        self.start_service([self.runtime, other])
        a = self.stub.Synthesize(self.request, timeout=5)
        b = self.stub.Synthesize(self.request, timeout=5)
        self.assertTrue(next(a).audio)
        self.assertTrue(next(b).audio)
        self.assertTrue(self.runtime.started.wait(1))
        self.assertTrue(other.started.wait(1))
        self.assertEqual(self.stub.GetInfo(pb.GetInfoRequest(), timeout=2).concurrency, 2)
        with self.assertRaises(grpc.RpcError) as caught:
            next(self.stub.Synthesize(self.request, timeout=2))
        self.assertEqual(caught.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)
        self.runtime.resume.set()
        other.resume.set()
        self.assertEqual(list(a)[-1].done.samples, 9600)
        self.assertEqual(list(b)[-1].done.samples, 9600)
        self.assertTrue(self.runtime.closed.wait(1))
        self.assertTrue(other.closed.wait(1))
        self.assertEqual(self.service.available.qsize(), 2)


if __name__ == "__main__":
    unittest.main()
