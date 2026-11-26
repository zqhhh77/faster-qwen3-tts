"""Benchmark isolated M5 gRPC services and recommend workers/threads for a chosen load."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
import platform
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

import grpc
import psutil

from .generated import inference_pb2 as pb
from .generated import inference_pb2_grpc as rpc
from .processes import ProcessTree

DEFAULT_TEXTS = (
    "前方船舶请注意安全。", "请保持安全航速。", "请勿在航道内停泊。",
    "前方正在施工，请谨慎驾驶。", "请按规定航线行驶。", "请保持通信畅通。",
)


def read_number(path):
    try:
        value = Path(path).read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def cpu_capacity():
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, math.floor(int(quota) / int(period))))
    except (OSError, ValueError):
        quota = read_number("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = read_number("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota and quota > 0 and period:
            count = min(count, max(1, quota // period))
    return count


def available_memory():
    available = psutil.virtual_memory().available
    for limit_path, used_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        limit, used = read_number(limit_path), read_number(used_path)
        if limit is not None and used is not None:
            available = min(available, max(0, limit - used))
    return available


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def integers(value):
    numbers = sorted({int(item) for item in value.split(",")})
    if not numbers or any(number < 1 for number in numbers):
        raise ValueError("候选值必须为正整数列表")
    return numbers


def combinations(concurrency, cpus, worker_values=None, thread_values=None):
    maximum = min(concurrency, cpus, 32)
    workers = worker_values or sorted({1, min(2, maximum), math.ceil(maximum / 2), maximum})
    result = []
    for count in workers:
        if count > min(concurrency, 32):
            continue
        budget = min(6, cpus // count)
        threads = thread_values or sorted({1, min(2, max(1, budget)), min(4, max(1, budget)), max(1, budget)})
        for number in threads:
            if number <= 256 and count * number <= cpus:
                result.append((count, number))
    return result


def corpus(path, count, request_path=None):
    if request_path:
        requests = json.loads(Path(request_path).read_text(encoding="utf-8-sig"))
        if not isinstance(requests, list):
            raise ValueError("任务文件必须为 JSON 数组")
        for request in requests:
            if (not isinstance(request, dict) or set(request) - {"input", "voice", "language"}
                    or not isinstance(request.get("input"), str) or not request["input"].strip()
                    or any(not isinstance(value, str) or not value for value in request.values())):
                raise ValueError("任务必须包含非空 input，可选 voice、language 字符串")
        if len(requests) < count:
            raise ValueError(f"任务文件至少需要 {count} 条任务")
        requests = requests[:count]
    elif path:
        texts = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len(set(texts)) < count:
            raise ValueError(f"文本文件必须至少有 {count} 条不同文本，一行一条")
        texts = list(dict.fromkeys(texts))[:count]
        requests = [{"input": text} for text in texts]
    else:
        texts = [DEFAULT_TEXTS[index % len(DEFAULT_TEXTS)] +
                 (f"这是第{index + 1}项测试。" if index >= len(DEFAULT_TEXTS) else "") for index in range(count)]
        requests = [{"input": text} for text in texts]
    if any(len(request["input"]) > 2048 for request in requests):
        raise ValueError("测试文本不能超过 2048 字符")
    return requests


def resolve_requests(requests, info, voice, language):
    resolved, seen = [], set()
    for request in requests:
        item = dict(input=request["input"], voice=request.get("voice", voice or info.default_voice),
                    language=request.get("language", language))
        matched = next((value for value in info.languages if value.lower() == item["language"].lower()), None)
        if item["voice"] not in info.voices or matched is None:
            raise ValueError("测试音色或语言不存在")
        item["language"] = matched
        key = (item["input"], item["voice"], item["language"])
        if key in seen:
            raise ValueError("任务重复：解析默认音色后，文本、音色和语言必须组成不同任务")
        seen.add(key)
        resolved.append(item)
    return resolved


def playback(chunks, prebuffer_ms=0, complete_s=None):
    if not math.isfinite(prebuffer_ms) or not 0 <= prebuffer_ms <= 1000:
        raise ValueError("开播缓冲必须为 0～1000 毫秒")
    first = chunks[0]["at_s"] + prebuffer_ms / 1000
    if complete_s is not None:
        first = min(first, complete_s)
    end = first
    underruns = 0
    stall_s = max_stall_s = 0.0
    for chunk in chunks:
        arrived = max(first, chunk["at_s"])
        gap = max(0.0, arrived - end)
        if gap > 0:
            underruns += 1
            stall_s += gap
            max_stall_s = max(max_stall_s, gap)
        end = max(end, arrived) + chunk["audio_s"]
    return dict(first_s=first, underruns=underruns, stall_s=stall_s, max_stall_s=max_stall_s)


def stream_request(stub, info, text, voice, language, timeout, *, trace=False, prebuffer_ms=None):
    if prebuffer_ms is not None and (not math.isfinite(prebuffer_ms) or not 0 <= prebuffer_ms <= 1000):
        raise ValueError("开播缓冲必须为 0～1000 毫秒")
    started = time.perf_counter()
    first, samples, done = None, 0, False
    chunks = []
    call = stub.Synthesize(pb.SynthesizeRequest(input=text, voice=voice, language=language,
                            speed=1, expected_identity=info.identity), timeout=timeout)
    try:
        for event in call:
            if done:
                raise ValueError("完成后仍收到数据")
            kind = event.WhichOneof("event")
            if kind == "audio":
                if not event.audio or len(event.audio) % 2 or len(event.audio) > 262144:
                    raise ValueError("音频块无效")
                arrived = time.perf_counter()
                if first is None:
                    first = arrived
                chunks.append(dict(at_s=arrived-started, audio_s=len(event.audio)/48000, bytes=len(event.audio)))
                samples += len(event.audio) // 2
            elif kind == "done":
                if not samples or samples != event.done.samples:
                    raise ValueError("完成样本数不一致")
                done = True
            else:
                raise ValueError("推理事件无效")
        if not done or call.code() != grpc.StatusCode.OK:
            raise ValueError("缺少成功完成标记")
        finished = time.perf_counter()
        timing = playback(chunks)
        timing.pop("first_s")
        result = dict(first_at=first, finished_at=finished, audio_s=samples / 24000,
                      inference_s=finished-started, **timing)
        if trace:
            result["chunks"] = chunks
        if prebuffer_ms is not None:
            result["buffered"] = dict(prebuffer_ms=prebuffer_ms,
                                      **playback(chunks, prebuffer_ms, finished-started))
            result["buffered"]["first_at"] = started + result["buffered"].pop("first_s")
        return result
    finally:
        call.cancel()


def batch(stub, info, requests, timeout, cancel=None, *, trace=False, prebuffer_ms=None):
    # Mirror the gateway's bounded dispatch: excess requests wait, and queue time counts toward latency/deadline.
    slots = threading.Semaphore(info.concurrency)
    ready = threading.Barrier(len(requests) + 1)

    def task(request):
        ready.wait()
        submitted = time.perf_counter()
        if not slots.acquire(timeout=timeout):
            return dict(ok=False, error="等待推理槽位超时")
        try:
            dispatched = time.perf_counter()
            remaining = timeout - (dispatched - submitted)
            if remaining <= 0:
                raise TimeoutError("等待推理槽位超时")
            result = stream_request(stub, info, request["input"], request["voice"], request["language"], remaining,
                                    trace=trace, prebuffer_ms=prebuffer_ms)
            row = dict(ok=True, first_s=result["first_at"] - submitted,
                        complete_s=result["finished_at"] - submitted, queue_s=dispatched - submitted,
                        audio_s=result["audio_s"], inference_s=result["inference_s"], underruns=result["underruns"],
                        stall_s=result["stall_s"], max_stall_s=result["max_stall_s"])
            if trace:
                row.update(voice=request["voice"], input=request["input"], chunks=result["chunks"])
            if "buffered" in result:
                row["buffered"] = result["buffered"]
                row["buffered"]["first_s"] = row["buffered"].pop("first_at") - submitted
            return row
        except (grpc.RpcError, ValueError, TimeoutError) as exc:
            return dict(ok=False, error=str(exc))
        finally:
            slots.release()

    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        try:
            futures = [pool.submit(task, request) for request in requests]
            started = time.perf_counter()
            ready.wait()
            rows = [future.result() for future in futures]
        except KeyboardInterrupt:
            ready.abort()
            if cancel is not None:
                cancel()
            raise
    return dict(wall_s=time.perf_counter() - started, requests=rows)


def summarize(workers, threads, batches, peak_bytes, startup_s):
    rows = [row for item in batches for row in item["requests"]]
    good = [row for row in rows if row["ok"]]
    wall = sum(item["wall_s"] for item in batches)
    return dict(workers=workers, threads=threads, ok=len(good) == len(rows),
                successes=len(good), failures=len(rows) - len(good),
                stalled_requests=sum(row["underruns"] > 0 for row in good),
                underruns=sum(row["underruns"] for row in good), stall_s=sum(row["stall_s"] for row in good),
                max_stall_s=max((row["max_stall_s"] for row in good), default=0.0),
                requests_per_s=len(good) / wall, audio_seconds_per_s=sum(row["audio_s"] for row in good) / wall,
                first_p95_s=percentile([row["first_s"] for row in good], .95) if good else None,
                complete_p95_s=percentile([row["complete_s"] for row in good], .95) if good else None,
                batch_mean_s=wall / len(batches), peak_rss_mb=peak_bytes / 1024**2,
                startup_s=startup_s, batches=batches)


def recommend(results, require_continuous=False):
    valid = [row for row in results if row.get("ok") and row.get("requests_per_s", 0) > 0
             and (not require_continuous or row.get("underruns") == 0)]
    return min(valid, key=lambda row: (-row["requests_per_s"], row["first_p95_s"], row["peak_rss_mb"])) if valid else None


def run_candidate(args, workers, threads, requests, directory, memory_limit):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    # 预热和多轮测量使用相同正文，调线程时每轮都需实际推理动态字段。
    env = dict(os.environ, M5_ALERT_FIELD_CACHE_MB="0", M5_MODEL=str(Path(args.model).resolve()), M5_WORKERS=str(workers),
               M5_THREADS=str(threads), M5_LISTEN=f"127.0.0.1:{port}", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    if args.voices:
        env["M5_VOICES_FILE"] = str(Path(args.voices).resolve())
    stopped, exceeded = threading.Event(), threading.Event()
    peak = [0]
    monitor_errors = []
    with (directory / f"w{workers}-t{threads}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "backend.m5.server"], env=env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        tree = ProcessTree(process)
        def monitor():
            while not stopped.is_set():
                try:
                    usage = tree.resident_bytes()
                except psutil.Error as exc:
                    monitor_errors.append(str(exc))
                    tree.stop(force=True)
                    return
                peak[0] = max(peak[0], usage)
                if memory_limit and usage > memory_limit:
                    exceeded.set()
                    tree.stop(force=True)
                    return
                stopped.wait(.1)
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        started = time.perf_counter()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = rpc.InferenceServiceStub(channel)
                deadline = started + args.startup_timeout
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("模型服务退出，详见该组日志")
                    try:
                        info = stub.GetInfo(pb.GetInfoRequest(), timeout=.5)
                        break
                    except grpc.RpcError:
                        if time.perf_counter() > deadline:
                            raise TimeoutError("模型启动超时")
                        time.sleep(.1)
                if (info.concurrency != workers or info.sample_rate != 24000 or info.channels != 1
                        or info.sample_format != "s16le"):
                    raise ValueError("模型服务能力与测试配置不一致")
                resolved = resolve_requests(requests, info, args.voice, args.language)
                startup = time.perf_counter() - started
                warmup = batch(stub, info, [resolved[index % len(resolved)] for index in range(workers)],
                               args.request_timeout, lambda: tree.stop(force=True))
                if not all(row["ok"] for row in warmup["requests"]):
                    raise RuntimeError("完整推理预热失败: " + str(warmup["requests"]))
                batches = [batch(stub, info, resolved, args.request_timeout, lambda: tree.stop(force=True))
                           for _ in range(args.rounds)]
                if exceeded.is_set() or monitor_errors:
                    raise RuntimeError("内存预算或统计失败")
                result = summarize(workers, threads, batches, peak[0], startup)
                result["identity"] = info.identity
                return result
        except (grpc.RpcError, ValueError, RuntimeError, TimeoutError) as exc:
            return dict(workers=workers, threads=threads, ok=False,
                        error="超过测试内存预算" if exceeded.is_set() else
                              "无法统计进程树内存: " + monitor_errors[0] if monitor_errors else str(exc),
                        peak_rss_mb=peak[0] / 1024**2)
        finally:
            stopped.set()
            watcher.join()
            tree.stop()


def save_report(directory, report):
    (directory / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ("workers", "threads", "ok", "requests_per_s", "audio_seconds_per_s", "first_p95_s",
              "complete_p95_s", "stalled_requests", "underruns", "stall_s", "max_stall_s",
              "batch_mean_s", "peak_rss_mb", "startup_s", "error")
    with (directory / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["results"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="在目标 CPU 上测试多个不同任务，按整批吞吐推荐 M5 workers/threads。")
    parser.add_argument("--model", default=os.environ.get("M5_MODEL", "models/M5-TTS-50M-ONNX"))
    parser.add_argument("--concurrency", type=int, default=6, help="同时到达的不同任务数，默认 6")
    parser.add_argument("--workers", help="候选实例数，如 1,2,3,6；默认按并发和 CPU 自动选择")
    parser.add_argument("--threads", help="候选算子线程数，如 1,2,3,4；默认自动选择")
    parser.add_argument("--cpu-budget", type=int, default=cpu_capacity(), help="候选 workers×threads 的上限；默认识别 CPU/容器配额")
    parser.add_argument("--rounds", type=int, default=3, help="每组预热后测试轮数，默认 3")
    parser.add_argument("--require-continuous", action="store_true",
                        help="只推荐所有正式测试请求都成功且首块即播零中断的组合，无合格组合时返回退出码 1")
    workload = parser.add_mutually_exclusive_group()
    workload.add_argument("--texts", help="UTF-8 文件，一行一条真实业务文本，至少 concurrency 条不同文本")
    workload.add_argument("--requests", help="JSON 任务数组，每项包含 input，可选 voice、language；允许同文本不同音色")
    parser.add_argument("--voices", help="音色配置文件，默认使用环境变量或镜像内配置")
    parser.add_argument("--voice", help="测试音色，默认使用服务默认音色")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--max-memory-gb", type=float, help="测试子进程 RSS 预算；默认当前可用内存的 70%%")
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float,
                        default=float(os.environ.get("TTS_TASK_TIMEOUT_SECONDS", "120")),
                        help="每个任务时限，包含排队；默认与网关相同的 120 秒")
    parser.add_argument("--output", help="结果目录，默认 .cache/m5-tune 下的独立目录")
    parser.add_argument("--list", action="store_true", help="只列出候选组合，不加载模型")
    args = parser.parse_args(argv)
    if not 1 <= args.concurrency <= 128 or args.cpu_budget < 1 or args.rounds < 1:
        parser.error("concurrency 必须为 1～128，cpu-budget 和 rounds 必须为正数")
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.startup_timeout, args.request_timeout, *([args.max_memory_gb] if args.max_memory_gb is not None else []))):
        parser.error("超时和内存预算必须大于 0")
    try:
        candidates = combinations(args.concurrency, args.cpu_budget,
                                  integers(args.workers) if args.workers else None,
                                  integers(args.threads) if args.threads else None)
        requests = corpus(args.texts, args.concurrency, args.requests)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not candidates:
        parser.error("没有候选组合满足并发数、实例上限与 CPU 预算")
    print(f"不同任务并发数={args.concurrency}，CPU 预算={args.cpu_budget}，候选组数={len(candidates)}", flush=True)
    print("候选: " + ", ".join(f"{w} workers × {t} threads" for w, t in candidates), flush=True)
    if args.list:
        return 0
    if not Path(args.model).is_dir():
        parser.error("模型目录不存在")
    available = available_memory()
    if available <= 0 and args.max_memory_gb is None:
        parser.error("当前没有可用内存，无法自动设置测试预算")
    memory_limit = int(args.max_memory_gb * 1024**3) if args.max_memory_gb else int(available * .7) if available else 0
    if args.output:
        directory = Path(args.output).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if any((directory / name).exists() for name in ("results.json", "recommended.env")):
            parser.error("结果目录已有报告，请选择新目录")
    else:
        base = Path(".cache/m5-tune")
        base.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=base)).resolve()
    print(f"结果目录: {directory}", flush=True)
    print(f"子进程 RSS 预算: {memory_limit / 1024**3:.2f} GiB" if memory_limit else "无法自动识别内存预算，建议显式指定 --max-memory-gb", flush=True)
    report = dict(status="running", concurrency=args.concurrency, rounds=args.rounds, cpu_budget=args.cpu_budget,
                  require_continuous=args.require_continuous,
                  memory_limit_bytes=memory_limit, requests=requests,
                  environment=dict(platform=platform.platform(), python=sys.version,
                                   logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False),
                                   detected_cpu_capacity=cpu_capacity()),
                  scope="独立 gRPC 模型服务，关闭动态字段缓存。客户端按实例数派发并计入排队时间，不经过 HTTP/磁盘缓存", results=[])
    try:
        for index, (workers, threads) in enumerate(candidates, 1):
            print(f"[{index}/{len(candidates)}] workers={workers}, threads={threads}：加载、预热并测试…", flush=True)
            result = run_candidate(args, workers, threads, requests, directory, memory_limit)
            report["results"].append(result)
            save_report(directory, report)
            if "requests_per_s" in result:
                print(f"  成功={result['successes']} 失败={result['failures']} 吞吐={result['requests_per_s']:.3f} 请求/秒 "
                      f"首块P95={result['first_p95_s']}秒 完成P95={result['complete_p95_s']}秒 RSS={result['peak_rss_mb']:.0f}MiB\n"
                      f"  成功请求中播放中断={result['underruns']} 次，涉及 {result['stalled_requests']} 条请求，"
                      f"累计 {result['stall_s']:.3f} 秒，最长 {result['max_stall_s']:.3f} 秒", flush=True)
            else:
                print(f"  跳过: {result['error']}", flush=True)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        save_report(directory, report)
        print("测试已停止，已完成的组保存在报告中。", flush=True)
        return 130
    best = recommend(report["results"], args.require_continuous)
    report.update(status="complete", recommendation={key: best[key] for key in (
        "workers", "threads", "requests_per_s", "first_p95_s", "complete_p95_s", "peak_rss_mb", "underruns")} if best else None)
    save_report(directory, report)
    criteria = "全部请求成功且播放零中断" if args.require_continuous else "全部请求成功"
    if best is None:
        print(f"没有满足「{criteria}」的候选组合，详细结果见 results.json。", flush=True)
        return 1
    config = (f"M5_WORKERS={best['workers']}\nM5_THREADS={best['threads']}\n"
              f"TTS_MAX_CONCURRENT_TASKS={best['workers']}\n"
              f"TTS_MAX_PENDING_TASKS={max(0, args.concurrency - best['workers'])}\n")
    (directory / "recommended.env").write_text(config, encoding="utf-8")
    print(f"\n推荐（已测候选中{criteria}、吞吐最高）：\n" + config, flush=True)
    print("结果仅适用于本次文本、并发、CPU/容器配额和机器负载；配置未自动应用。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
