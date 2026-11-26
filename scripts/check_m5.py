"""Run the real CPU model through gRPC -> Go -> all existing HTTP audio transports."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.m5.processes import ProcessTree
from backend.m5.tune import playback
from examples.http_streaming_client import receive_frames


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/M5-TTS-50M-ONNX")
    parser.add_argument("--gateway", default=".cache/gateway.exe" if os.name == "nt" else ".cache/gateway")
    parser.add_argument("--output", default=".cache/m5-check")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--text", default="你好，欢迎使用语音合成服务。", help="真实模型检查使用的正文")
    parser.add_argument("--startup-buffer-ms", type=int, default=0, help="网关首块到达后的额外开播缓冲，0～1000 毫秒")
    args = parser.parse_args()
    if not 0 <= args.startup_buffer_ms <= 1000:
        parser.error("startup-buffer-ms 必须为 0～1000")
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    voice_file = output / "voices.json"
    reference = {"wav": str(root / "backend/m5/default.wav"),
                 "text": "猪携带该疾病，然后通过蚊子传染给人类。"}
    voice_file.write_text(json.dumps({"default": reference, "alternate": reference}, ensure_ascii=False), encoding="utf-8")
    grpc_port, http_port = free_port(), free_port()
    env = dict(os.environ, M5_MODEL=str((root / args.model).resolve()),
               M5_LISTEN=f"127.0.0.1:{grpc_port}", M5_THREADS="2", M5_MAX_FRAMES="256",
               M5_WORKERS=str(args.workers),
               M5_TEXT_NORMALIZATION="off", M5_DEFAULT_VOICE="default",
               M5_VOICES_FILE=str(voice_file),
               TTS_BACKEND_TYPE="m5", TTS_BACKEND_GRPC_ADDR=f"127.0.0.1:{grpc_port}",
               TTS_LISTEN=f"127.0.0.1:{http_port}", TTS_MAX_CONCURRENT_TASKS=str(args.workers),
               TTS_STARTUP_BUFFER_MS=str(args.startup_buffer_ms),
               TTS_TASK_TIMEOUT_SECONDS="120", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    processes = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="cache-", dir=output) as cache_dir, \
            (output / "backend.log").open("w", encoding="utf-8") as backend_log, \
            (output / "gateway.log").open("w", encoding="utf-8") as gateway_log:
        env["TTS_CACHE_DIR"] = cache_dir
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        base = f"http://127.0.0.1:{http_port}"
        try:
            processes.append(ProcessTree(subprocess.Popen([sys.executable, "-m", "backend.m5.server"], cwd=root,
                                               env=env, stdout=backend_log, stderr=subprocess.STDOUT,
                                               creationflags=flags)))
            processes.append(ProcessTree(subprocess.Popen([str((root / args.gateway).resolve())], cwd=root,
                                               env=env, stdout=gateway_log, stderr=subprocess.STDOUT,
                                               creationflags=flags)))
            deadline = time.monotonic() + 120
            while True:
                if any(p.process.poll() is not None for p in processes):
                    raise RuntimeError("服务启动失败，请检查输出目录中的日志")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=5) as response:
                        health = json.load(response)
                    if health["code"] == 200:
                        break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("服务未就绪")
                    time.sleep(0.2)
            ready = time.perf_counter() - started
            with urllib.request.urlopen(base + "/v1/audio/voices") as response:
                voices = json.load(response)
            assert voices["rows"] == [{"voice": "alternate", "type": "Base", "default": False},
                                       {"voice": "default", "type": "Base", "default": True}], voices
            request = {"input": args.text}

            def framed(voice=None):
                payload = dict(request)
                if voice is not None:
                    payload["voice"] = voice
                sent = time.perf_counter()
                req = urllib.request.Request(base + "/v1/audio/speech", data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as response:
                    assert response.headers["X-Audio-Sample-Format"] == "s16le"
                    audio, chunks = bytearray(), []
                    for block in receive_frames(response):
                        chunks.append(dict(at_s=time.perf_counter()-sent, audio_s=len(block)/48000, bytes=len(block)))
                        audio.extend(block)
                assert audio
                first_at = chunks[0]["at_s"]
                finished = time.perf_counter()
                return bytes(audio), first_at, finished - sent, sent + first_at, finished, chunks

            selected = (None, "default", "alternate") if args.workers > 1 else (None, "default")
            with ThreadPoolExecutor(max_workers=len(selected)) as pool:
                results = list(pool.map(framed, selected))
            a, b = results[:2]
            alternate = results[2] if args.workers > 1 else None
            if alternate is not None:
                assert alternate[3] < a[4] and a[3] < alternate[4], "两个不同任务没有并行输出音频"
            assert a[0] == b[0], "共享任务结果不一致"
            cached = framed("default")
            assert cached[0] == a[0]
            for route, fmt in (("/v1/audio/speech", "pcm"), ("/v1/audio/speech/sse", "pcm_framed")):
                req = urllib.request.Request(base + route, data=json.dumps(dict(request, response_format=fmt)).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as response:
                    data = response.read()
                if fmt == "pcm":
                    assert data == a[0]
                else:
                    import base64
                    decoded = bytearray()
                    events = [json.loads(line[6:]) for line in data.decode().splitlines() if line.startswith("data: ")]
                    assert events[-1]["data"]["type"] == "done"
                    for event in events[:-1]:
                        assert event["data"]["type"] == "chunk"
                        with wave.open(io.BytesIO(base64.b64decode(event["data"]["audio_b64"]))) as wav:
                            assert wav.getframerate() == 24000
                            decoded.extend(wav.readframes(wav.getnframes()))
                    assert decoded == a[0]
            with wave.open(str(output / "speech.wav"), "wb") as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000)
                wav.writeframes(a[0])
            if alternate is None:
                alternate = framed("alternate")
            assert alternate[0] == a[0], "同一参考音频与种子的音频应一致"
            # Alias references intentionally match; different voice names still generate separate cache entries.
            backend_log.flush()
            logs = (output / "backend.log").read_text(encoding="utf-8")
            assert logs.count("M5 请求结束") == 2, logs
            assert logs.count("音色预加载完成") == 2 * args.workers, logs
            result = dict(ready_s=ready, first_audio_s=a[1], complete_s=a[2],
                          cache_first_audio_s=cached[1], audio_s=len(a[0])/48000,
                          shared_requests=2, preloaded_voices_per_worker=2, workers=args.workers, inference_runs=2,
                          transports=["pcm_framed", "pcm", "sse"],
                          identity=health["data"]["backend"]["identity"], input=args.text,
                          startup_buffer_ms=args.startup_buffer_ms,
                          requests=[dict(voice=voice or "default", complete_s=row[2],
                                         **playback(row[5]), chunks=row[5]) for voice, row in zip(selected, results)])
            (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            for process in processes:
                process.stop()


if __name__ == "__main__":
    main()
