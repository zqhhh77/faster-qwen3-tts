"""Internal gRPC endpoint; only the Go gateway exposes HTTP to clients."""
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from queue import Empty, Queue
import signal
import threading
import time

import grpc

from .generated import inference_pb2 as pb
from .generated import inference_pb2_grpc as rpc
from .runtime import LANGUAGES, Runtime
from .batching import DepthBatcher


class Service(rpc.InferenceServiceServicer):
    def __init__(self, runtimes):
        if not runtimes or len({r.identity for r in runtimes}) != 1:
            raise ValueError("推理实例必须使用一致的模型和音色配置")
        self.runtime = runtimes[0]
        self.capacity = len(runtimes)
        self.depth_batcher = None
        if getattr(self.runtime, "depth_batch", False):
            if self.capacity != 2:
                raise ValueError("深度合批需要 M5_WORKERS=2")
            self.depth_batcher = DepthBatcher.load(self.runtime.model)
            for runtime in runtimes:
                runtime.model.sessions["depth_frame"] = self.depth_batcher
            logging.info("双请求深度合批已启用，各请求独立保留音色和解码状态")
        self.available = Queue(maxsize=self.capacity)
        for runtime in runtimes:
            self.available.put_nowait(runtime)

    def close(self):
        if self.depth_batcher is not None:
            self.depth_batcher.close()

    def GetInfo(self, request, context):
        r = self.runtime
        return pb.ModelInfo(identity=r.identity, model=r.model_name, voices=sorted(r.voices),
                            default_voice=r.default_voice, languages=LANGUAGES, sample_rate=24000,
                            channels=1, sample_format="s16le", concurrency=self.capacity,
                            text_normalization=r.normalization)

    def Synthesize(self, request, context):
        r = self.runtime
        if request.expected_identity != r.identity:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "模型身份已变化")
        if (not request.input.strip() or len(request.input) > 2048 or request.speed != 1
                or request.instruct or request.language not in LANGUAGES):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "文本、语言、指令或语速无效")
        voice = request.voice or r.default_voice
        if voice not in r.voices:
            context.abort(grpc.StatusCode.NOT_FOUND, "指定音色不存在")
        try:
            r = self.available.get_nowait()
        except Empty:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "模型正在处理其他请求")
        started, first, samples = time.perf_counter(), None, 0
        chunks = None
        if self.depth_batcher is not None:
            self.depth_batcher.enter()
        try:
            chunks = r.generate(request.input, voice, request.language)
            while context.is_active():
                try:
                    chunk = next(chunks)
                except StopIteration:
                    break
                if not context.is_active():
                    return
                if first is None:
                    first = time.perf_counter() - started
                for offset in range(0, len(chunk), 262144):
                    if not context.is_active():
                        return
                    part = chunk[offset:offset + 262144]
                    samples += len(part) // 2
                    yield pb.SynthesizeEvent(audio=part)
            if context.is_active():
                if not samples:
                    raise RuntimeError("模型没有生成音频")
                yield pb.SynthesizeEvent(done=pb.Completion(samples=samples))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception:
            logging.exception("M5 推理失败")
            context.abort(grpc.StatusCode.INTERNAL, "语音生成失败，已接收音频应丢弃")
        finally:
            try:
                if chunks is not None:
                    chunks.close()
            finally:
                if self.depth_batcher is not None:
                    self.depth_batcher.leave()
                self.available.put_nowait(r)
                logging.info("M5 请求结束 voice=%s chars=%d first_audio_s=%s elapsed_s=%.3f audio_s=%.3f",
                             voice, len(request.input), first, time.perf_counter()-started, samples/24000)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    workers = int(os.environ.get("M5_WORKERS", "1"))
    if not 1 <= workers <= 32:
        raise ValueError("M5_WORKERS 必须为 1 到 32")
    runtimes = []
    for index in range(workers):
        logging.info("加载 M5 CPU 实例 %d/%d", index + 1, workers)
        phrases = runtimes[0].alert_phrases if runtimes else None
        fields = runtimes[0].field_cache if runtimes else None
        runtimes.append(Runtime(shared_alert_phrases=phrases, shared_field_cache=fields))
    server = grpc.server(ThreadPoolExecutor(max_workers=max(4, workers + 2)),
                         maximum_concurrent_rpcs=workers + 16,
                         options=(("grpc.max_receive_message_length", 65536),
                                  ("grpc.max_send_message_length", 1048576)))
    service = Service(runtimes)
    rpc.add_InferenceServiceServicer_to_server(service, server)
    address = os.environ.get("M5_LISTEN", "[::]:50051")
    if not server.add_insecure_port(address):
        service.close()
        raise RuntimeError(f"无法监听 gRPC 地址: {address}")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    server.start()
    logging.info("M5 gRPC 监听 %s，独立模型实例数 %d", address, workers)
    try:
        stop.wait()
    finally:
        server.stop(10).wait()
        service.close()


if __name__ == "__main__":
    main()
