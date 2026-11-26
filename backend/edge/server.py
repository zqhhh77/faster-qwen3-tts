"""Edge TTS 后端通过内部 gRPC 返回 PCM。"""
import asyncio
from contextlib import aclosing, suppress
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal

import edge_tts
import grpc

from backend.m5.generated import inference_pb2 as pb
from backend.m5.generated import inference_pb2_grpc as rpc


def read_voices(path):
    def unique(pairs):
        voices = dict(pairs)
        if len(voices) != len(pairs):
            raise ValueError("Edge 音色名称不能重复")
        return voices

    voices = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique)
    if not isinstance(voices, dict) or "default" not in voices:
        raise ValueError("Edge 音色映射必须包含 default")
    for alias, voice in voices.items():
        if not alias.strip() or alias != alias.strip() or len(alias) > 128:
            raise ValueError("Edge 公开音色名称无效")
        if not isinstance(voice, str) or not re.fullmatch(r"[A-Za-z0-9-]+", voice):
            raise ValueError(f"Edge 音色 {alias} 必须配置预设音色短名称")
    return voices


class MotoEventLoop(asyncio.ProactorEventLoop if os.name == "nt" else asyncio.SelectorEventLoop):
    def __init__(self, address):
        host, separator, port = address.rpartition(":")
        if not separator or not host or not port.isdecimal() or not 1 <= int(port) <= 65535:
            raise ValueError("EDGE_TTS_MOTO_ADDR 必须使用 主机:端口 格式")
        super().__init__()
        self.moto_host = host.removeprefix("[").removesuffix("]")
        self.moto_port = int(port)

    async def getaddrinfo(self, host, port, **kwargs):
        # 改写 TCP 目标，WebSocket URL 仍使用原域名。
        if host == "speech.platform.bing.com" and port == 443:
            host, port = self.moto_host, self.moto_port
        return await super().getaddrinfo(host, port, **kwargs)


async def decode(text, voice, ffmpeg):
    process = await asyncio.create_subprocess_exec(
        ffmpeg, "-hide_banner", "-loglevel", "error", "-xerror",
        "-f", "mp3", "-probesize", "4096", "-analyzeduration", "100000", "-i", "pipe:0",
        "-f", "s16le", "-ar", "24000", "-ac", "1", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    async def feed_mp3():
        try:
            async with aclosing(edge_tts.Communicate(text, voice).stream()) as stream:
                async for chunk in stream:
                    if chunk["type"] == "audio":
                        process.stdin.write(chunk["data"])
                        await process.stdin.drain()
        finally:
            process.stdin.close()

    feed = asyncio.create_task(feed_mp3())
    try:
        while True:
            try:
                pcm = await process.stdout.readexactly(4096)
            except asyncio.IncompleteReadError as exc:
                if len(exc.partial) % 2:
                    raise RuntimeError("Edge 解码返回不完整的 PCM 采样") from exc
                if exc.partial:
                    yield exc.partial
                break
            yield pcm
        await feed
        if await process.wait():
            raise RuntimeError("Edge MP3 解码失败")
    finally:
        feed.cancel()
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        await asyncio.gather(feed, return_exceptions=True)
        await process.stdout.read()
        await process.wait()


class Service(rpc.InferenceServiceServicer):
    def __init__(self, voices, workers, ffmpeg):
        self.voices = voices
        self.ffmpeg = ffmpeg
        self.slots = asyncio.Semaphore(workers)
        identity = json.dumps(voices, sort_keys=True).encode()
        self.info = pb.ModelInfo(
            identity=hashlib.sha256(b"edge-tts-v1:" + identity).hexdigest(),
            model="edge-tts", voices=sorted(voices), default_voice="default",
            languages=["Chinese", "English"], sample_rate=24000, channels=1,
            sample_format="s16le", concurrency=workers,
        )

    async def GetInfo(self, request, context):
        return self.info

    async def Synthesize(self, request, context):
        if request.expected_identity != self.info.identity:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "音色映射已变化")
        if (not request.input.strip() or len(request.input) > 2048 or request.speed != 1
                or request.instruct or request.language not in self.info.languages):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "文本、语言、指令或语速无效")
        voice = self.voices.get(request.voice or self.info.default_voice)
        if voice is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "指定音色不存在")
        if self.slots.locked():
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "Edge 后端正在处理其他请求")
        async with self.slots:
            samples = 0
            try:
                async with aclosing(decode(request.input, voice, self.ffmpeg)) as chunks:
                    async for pcm in chunks:
                        samples += len(pcm) // 2
                        await context.write(pb.SynthesizeEvent(audio=pcm))
                if not samples:
                    raise RuntimeError("Edge 后端没有生成音频")
            except Exception:
                logging.exception("Edge 语音生成失败")
                await context.abort(grpc.StatusCode.INTERNAL, "语音生成失败，已接收音频应丢弃")
            await context.write(pb.SynthesizeEvent(done=pb.Completion(samples=samples)))


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    voices = read_voices(os.environ["EDGE_TTS_VOICES_FILE"])
    workers = int(os.environ.get("EDGE_TTS_WORKERS", "4"))
    if not 1 <= workers <= 256:
        raise ValueError("EDGE_TTS_WORKERS 必须为 1 到 256")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Edge 后端需要 PATH 中的 FFmpeg")
    server = grpc.aio.server(maximum_concurrent_rpcs=workers + 16,
                             options=(("grpc.max_receive_message_length", 65536),))
    rpc.add_InferenceServiceServicer_to_server(Service(voices, workers, ffmpeg), server)
    address = os.environ.get("EDGE_TTS_LISTEN", "[::]:50051")
    server.add_insecure_port(address)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))
    await server.start()
    logging.info("Edge 后端监听 %s，公开音色 %d 个，并发 %d", address, len(voices), workers)
    try:
        await stop.wait()
    finally:
        await server.stop(10)


if __name__ == "__main__":
    moto = os.environ.get("EDGE_TTS_MOTO_ADDR", "").strip() or "115.28.184.80:9010"
    if moto == "direct":
        asyncio.run(main())
    else:
        with asyncio.Runner(loop_factory=lambda: MotoEventLoop(moto)) as runner:
            runner.run(main())
