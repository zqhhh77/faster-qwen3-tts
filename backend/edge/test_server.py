from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import grpc

from backend.edge import server


class EdgeServerTest(IsolatedAsyncioTestCase):
    def test_optional_proxy(self):
        self.assertIsNone(server.read_proxy(" "))
        self.assertEqual(server.read_proxy("http://moto:9006"), "http://moto:9006")
        with self.assertRaises(ValueError):
            server.read_proxy("socks5://moto:9006")

    async def test_public_aliases_and_completion(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "voices.json"
            path.write_text('{"default":"zh-CN-XiaoxiaoNeural","broadcast":"zh-CN-YunxiNeural"}')
            service = server.Service(server.read_voices(path), 1, "ffmpeg", "http://moto:9006")
            path.write_text('{"default":"zh-CN-YunxiNeural"}')
            changed = server.Service(server.read_voices(path), 1, "ffmpeg")
            self.assertNotEqual(service.info.identity, changed.info.identity)
            path.write_text('{"default":"a","default":"b"}')
            with self.assertRaises(ValueError):
                server.read_voices(path)

        grpc_server = grpc.aio.server()
        server.rpc.add_InferenceServiceServicer_to_server(service, grpc_server)
        port = grpc_server.add_insecure_port("127.0.0.1:0")
        await grpc_server.start()
        self.addAsyncCleanup(grpc_server.stop, 0)

        async def decode(text, voice, ffmpeg, proxy):
            self.assertEqual(voice, "zh-CN-YunxiNeural")
            self.assertEqual(proxy, "http://moto:9006")
            yield b"\x00\x01"
            if text == "中断":
                raise RuntimeError("上游中断")

        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            client = server.rpc.InferenceServiceStub(channel)
            info = await client.GetInfo(server.pb.GetInfoRequest(), timeout=5)
            self.assertEqual(list(info.voices), ["broadcast", "default"])
            self.assertEqual(info.default_voice, "default")
            request = server.pb.SynthesizeRequest(
                input="你好", voice="zh-CN-YunxiNeural", language="Chinese", speed=1,
                expected_identity=info.identity,
            )
            with self.assertRaises(grpc.aio.AioRpcError) as rejected:
                await client.Synthesize(request, timeout=5).read()
            self.assertEqual(rejected.exception.code(), grpc.StatusCode.NOT_FOUND)
            request.voice = "broadcast"
            with patch.object(server, "decode", decode):
                events = [event async for event in client.Synthesize(request, timeout=5)]
                self.assertEqual(events[0].audio, b"\x00\x01")
                self.assertEqual(events[1].done.samples, 1)
                request.input = "中断"
                with self.assertLogs(level="ERROR"):
                    stream = client.Synthesize(request, timeout=5)
                    self.assertEqual((await stream.read()).WhichOneof("event"), "audio")
                    with self.assertRaises(grpc.aio.AioRpcError) as interrupted:
                        await stream.read()
                self.assertEqual(interrupted.exception.code(), grpc.StatusCode.INTERNAL)
