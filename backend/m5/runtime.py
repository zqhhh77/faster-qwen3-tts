"""One CPU/CUDA model instance; the service holds its slot for the whole generator."""
from __future__ import annotations

import hashlib
import copy
from contextlib import closing
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort

from .alert_phrases import (ALERT_ENDINGS, ALERT_PHRASES, AlertFieldCache, alert_parts,
                           silence_threshold, split_ship_number, stretch_number_tail, trim_alert_edges)
from .cuda_graph import CudaGraph

VENDOR = Path(__file__).with_name("vendor")
sys.path.insert(0, str(VENDOR))
from infer_onnx import FILES, M5Onnx, conditioned_text, create_session, execution_providers, read_reference
import text_normalization

LANGUAGES = ("Chinese", "English", "German", "French", "Japanese", "Korean")


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_bundle(root: Path) -> dict:
    checks = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    names = (*FILES[:-1], "depth_frame_pair.onnx") if manifest.get("depth_frame_pair") else FILES[:-1]
    for name in names:
        path = root / name
        expected = checks[name]
        digest = digest_file(path)
        size = path.stat().st_size
        if path.suffix == ".json" and digest != expected["sha256"]:
            # Git on Windows may change JSON line endings; verify the original LF bytes.
            normalized = path.read_bytes().replace(b"\r\n", b"\n")
            digest, size = hashlib.sha256(normalized).hexdigest(), len(normalized)
        if digest != expected["sha256"] or size != expected["bytes"]:
            raise ValueError(f"模型文件校验失败: {name}")
    return {name: checks[name]["sha256"] for name in names}


def pcm16(chunk) -> bytes:
    values = np.asarray(chunk, dtype=np.float32)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("模型输出的音频无效")
    return np.clip(values * 32767.0, -32768, 32767).astype("<i2").tobytes()


class Runtime:
    def __init__(self, *, shared_alert_phrases=None, shared_field_cache=None):
        root = Path(os.environ.get("M5_MODEL", "/models/m5")).resolve(strict=True)
        assets = verify_bundle(root)
        self.device = os.environ.get("M5_DEVICE", "cpu")
        self.device_id = int(os.environ.get("M5_DEVICE_ID", "0"))
        if self.device not in ("cpu", "cuda") or self.device_id < 0 or (self.device == "cpu" and self.device_id != 0):
            raise ValueError("M5_DEVICE 必须为 cpu 或 cuda；M5_DEVICE_ID 必须为非负整数，CPU 使用 0")
        if self.device == "cuda":
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("depth_frame_precision", "fp32") != "fp32" or manifest.get("codec_decoder_fast_conv"):
                raise ValueError("CUDA 请使用原始 FP32、FP32 查表或 FP32 合批模型，不能使用 CPU INT8 模型")
        self.threads = int(os.environ.get("M5_THREADS", "2"))
        batch_mode = os.environ.get("M5_DEPTH_BATCH", "off")
        if batch_mode not in ("on", "off"):
            raise ValueError("M5_DEPTH_BATCH 必须为 on 或 off")
        self.depth_batch = batch_mode == "on"
        graph_mode = os.environ.get("M5_CUDA_GRAPH", "on")
        if graph_mode not in ("on", "off"):
            raise ValueError("M5_CUDA_GRAPH 必须为 on 或 off")
        self.cuda_graph_enabled = self.device == "cuda" and graph_mode == "on" and not self.depth_batch
        if self.depth_batch and "depth_frame_pair.onnx" not in assets:
            raise ValueError("开启深度合批需要带 depth_frame_pair.onnx 的模型目录")
        self.max_frames = int(os.environ.get("M5_MAX_FRAMES", "1024"))
        if not 1 <= self.threads <= 256 or not 1 <= self.max_frames <= 2048:
            raise ValueError("M5_THREADS 或 M5_MAX_FRAMES 超出允许范围")
        mode = os.environ.get("M5_TEXT_NORMALIZATION", "off")
        if mode not in ("on", "off"):
            raise ValueError("M5_TEXT_NORMALIZATION 只能为 on 或 off")
        self.normalization = mode == "on"
        if self.normalization:
            # Fail at startup instead of silently changing pronunciation between requests.
            import pynini  # noqa: F401
            for language in LANGUAGES:
                text_normalization.normalize_text("test", lang=language)
        else:
            text_normalization.normalize_text = lambda text, **kwargs: text
        self.model = M5Onnx(str(root), threads=self.threads, codec_mode="cached",
                            device=self.device, device_id=self.device_id, local_files_only=True)
        logging.info("M5 推理设备 device=%s device_id=%d", self.device, self.device_id)
        voice_file = Path(os.environ.get("M5_VOICES_FILE", str(Path(__file__).with_name("voices.json")))).resolve(strict=True)
        definitions = json.loads(voice_file.read_text(encoding="utf-8"))
        if not isinstance(definitions, dict) or not 1 <= len(definitions) <= 64:
            raise ValueError("音色配置必须包含 1 到 64 个音色")
        self.voices = {}
        identity_voices = {}
        use_int8 = self.model.meta.get("depth_frame_precision") == "int8"
        # FP32 首两帧扣除保留尾音后仅有 60 ms，CUDA 首块使用 6 帧正文。
        self.settings = dict(seed=1234, first_frames=6 if use_int8 or self.device == "cuda" else 2,
                             chunk_frames=4, left_context=8)
        for name, definition in definitions.items():
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name):
                raise ValueError(f"音色名称无效: {name}")
            path = (voice_file.parent / definition["wav"]).resolve(strict=True)
            text = definition["text"]
            if not isinstance(text, str) or not text.strip() or len(text) > 2048:
                raise ValueError(f"音色参考转写无效: {name}")
            started = time.perf_counter()
            codes = self.model.encode_reference(read_reference(str(path), self.model.sample_rate))
            prefix = self.model.prepare_decoder_prefix(codes[-self.settings["left_context"]:]) if use_int8 else None
            logging.info("音色预加载完成 voice=%s frames=%d elapsed_s=%.3f", name, len(codes),
                         time.perf_counter() - started)
            self.voices[name] = {"wav": str(path), "text": text, "codes": codes, "decoder_prefix": prefix}
            identity_voices[name] = {"audio": digest_file(path), "text": text}
        if self.device == "cuda":
            # 全部音色已编码，请求复用 codes；释放只在启动时使用的显存。
            del self.model.sessions["codec_encoder"]
        self.default_voice = os.environ.get("M5_DEFAULT_VOICE") or next(iter(self.voices))
        if self.default_voice not in self.voices:
            raise ValueError("默认音色不在配置列表中")
        phrase_mode = os.environ.get("M5_ALERT_PHRASES", "on")
        if phrase_mode not in ("on", "off"):
            raise ValueError("M5_ALERT_PHRASES 必须为 on 或 off")
        ship_mode = os.environ.get("M5_ALERT_SHIP_PARTS", "off")
        if ship_mode not in ("on", "off"):
            raise ValueError("M5_ALERT_SHIP_PARTS 必须为 on 或 off")
        self.ship_parts = ship_mode == "on"
        self.number_tail_speed = float(os.environ.get("M5_ALERT_NUMBER_TAIL_SPEED", "0.65"))
        if self.number_tail_speed not in (0.5, 0.65, 1):
            raise ValueError("M5_ALERT_NUMBER_TAIL_SPEED 必须为 0.5、0.65 或 1")
        self.ffmpeg = None
        ffmpeg_version = None
        if self.ship_parts:
            if phrase_mode != "on" or int(os.environ.get("M5_ALERT_FIELD_CACHE_MB", "0")) <= 0:
                raise ValueError("船名前缀复用需要 M5_ALERT_PHRASES=on 且 M5_ALERT_FIELD_CACHE_MB 大于 0")
            if self.number_tail_speed < 1:
                self.ffmpeg = shutil.which("ffmpeg")
                if not self.ffmpeg:
                    raise ValueError("编号尾音延长需要安装 FFmpeg")
                ffmpeg_version = subprocess.run([self.ffmpeg, "-version"], capture_output=True, check=True,
                                                timeout=5, text=True).stdout.splitlines()[0]
        identity = {
            "backend": "m5-" + self.device, "device_id": self.device_id,
            "assets": assets, "voices": identity_voices,
            "default_voice": self.default_voice, "threads": self.threads, "depth_batch": self.depth_batch,
            "cuda_graph": self.cuda_graph_enabled,
            "max_frames": self.max_frames, "settings": self.settings,
            "normalization": self.normalization, "alert_phrases": phrase_mode,
            "ship_parts": self.ship_parts, "number_tail_speed": self.number_tail_speed if self.ship_parts else 1,
            "ffmpeg": ffmpeg_version,
            "source": {str(p.relative_to(Path(__file__).parent)): digest_file(p)
                       for p in (Path(__file__), Path(__file__).with_name("server.py"),
                                 Path(__file__).with_name("alert_phrases.py"),
                                 Path(__file__).with_name("batching.py"),
                                 Path(__file__).with_name("cuda_graph.py"),
                                 VENDOR / "infer_onnx.py", VENDOR / "text_normalization.py")},
            "packages": {"onnxruntime": ort.__version__, **{name: importlib.metadata.version(name) for name in (
                "numpy", "scipy", "tokenizers", "soundfile")}},
        }
        if self.normalization:
            identity["normalizer_packages"] = {name: importlib.metadata.version(name)
                                               for name in ("pynini", "WeTextProcessing")}
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.cache_identities = {}
        for voice, definition in identity_voices.items():
            per_voice = dict(identity, voices={voice: definition})
            per_voice.pop("default_voice")
            self.cache_identities[voice] = hashlib.sha256(json.dumps(per_voice, sort_keys=True).encode()).hexdigest()
        self.alert_phrases = None
        self.field_cache = None
        if phrase_mode == "on":
            directory = Path(os.environ.get("M5_ALERT_CACHE_DIR", "cache/m5-alerts"))
            cache_mb = int(os.environ.get("M5_ALERT_FIELD_CACHE_MB", "0"))
            if cache_mb < 0:
                raise ValueError("M5_ALERT_FIELD_CACHE_MB 必须为非负整数")
            if cache_mb:
                self.field_cache = shared_field_cache or AlertFieldCache(directory / "fields", cache_mb * 1024**2)
            self.alert_phrases = shared_alert_phrases or AlertFieldCache(directory / "phrases")
            if shared_alert_phrases is None:
                self._prepare_alert_phrases()
            logging.info("告警磁盘缓存已启用 path=%s field_cache_mb=%d", directory.resolve(), cache_mb)
        self.model_name = root.name
        self.cuda_graph = None
        if self.cuda_graph_enabled:
            self.cuda_graph = CudaGraph(self.model)
            self.model.sessions["depth_frame"] = self.cuda_graph
            self.model.provider_options["depth_frame"] = self.cuda_graph.session.get_provider_options()
            logging.info("CUDA 深度生成 Graph 已捕获，工作实例共享权重并独立生成音色")
        logging.info("M5 已就绪 identity=%s voices=%s normalization=%s",
                     self.identity, list(self.voices), self.normalization)
        if self.ship_parts:
            logging.info("船名前缀复用已启用 number_tail_speed=%s", self.number_tail_speed)

    def fork_cuda(self):
        """启动时共享 CUDA Session，每个工作实例独立持有生成和解码状态。"""
        if self.device != "cuda" or self.model.stats.get("status") == "running":
            raise ValueError("只能从空闲的 CUDA 实例创建工作实例")
        worker = copy.copy(self)
        worker.settings = self.settings.copy()
        worker.model = copy.copy(self.model)
        worker.model.sessions = self.model.sessions.copy()
        worker.model.stats = {}
        worker.model.__dict__.pop("generated_codes", None)
        worker.model._reset_codec()
        return worker

    def _prepare_alert_phrases(self):
        for voice in self.voices:
            reused = 0
            for phrase in ALERT_PHRASES:
                key = (self.cache_identities[voice], phrase)
                if self.alert_phrases.get(key) is None:
                    self._alert_phrase(voice, phrase)
                else:
                    reused += 1
            logging.info("告警短语预生成完成 voice=%s phrases=%d disk_hits=%d generated=%d",
                         voice, len(ALERT_PHRASES), reused, len(ALERT_PHRASES) - reused)

    def _alert_phrase(self, voice, phrase):
        key = (self.cache_identities[voice], phrase)
        entry = self.alert_phrases.get(key)
        if entry is None:
            spec = self.voices[voice]
            text = phrase + "。" if phrase in ALERT_ENDINGS else phrase
            budget = min(1024, self._frame_budget(text, spec, "Chinese"))
            pcm = b"".join(self._generate_text(text, spec, "Chinese", budget))
            pcm = b"".join(trim_alert_edges([pcm], silence_threshold(pcm)))
            entry = (pcm, self.model.stats["frames"])
            self.alert_phrases.put(key, *entry)
        return entry

    def _frame_budget(self, text, spec, language):
        normalized = text_normalization.normalize_text(text, lang=language)
        ids = self.model.tokenizer.encode(conditioned_text(spec["text"].strip() + normalized.strip(), language)).ids
        available = self.model.meta["max_sequence"] - len(ids) - len(spec["codes"])
        if available <= self.model.meta["min_frames"]:
            raise ValueError("文本和参考音频超出模型上下文容量")
        return available

    def generate(self, text: str, voice: str, language: str):
        spec = self.voices[voice]
        budget = min(self.max_frames, self._frame_budget(text, spec, language))
        if self.alert_phrases is not None and language == "Chinese":
            ending_only = text.strip().removesuffix("。").rstrip()
            if ending_only in ALERT_ENDINGS:
                pcm, frames = self._alert_phrase(voice, ending_only)
                if frames <= budget:
                    yield pcm
                    return
            parts = alert_parts(text)
            if parts:
                ship, area, kind, ending = parts
                ship_parts = split_ship_number(ship) if self.ship_parts else None
                phrases = {p: self._alert_phrase(voice, p) for p in ("发现", "触发", ending)}
                remaining = budget - sum(phrases[p][1] for p in ("发现", "触发", ending))
                if remaining > (2 + bool(area) + bool(ship_parts)) * self.model.meta["min_frames"]:
                    threshold = silence_threshold(phrases["发现"][0])
                    lead = bytearray(phrases["发现"][0])
                    if ship_parts:
                        name, ship = ship_parts
                        with closing(self._alert_field(name, voice, remaining, threshold)) as stream:
                            while True:
                                try:
                                    lead.extend(next(stream))
                                except StopIteration as finished:
                                    remaining -= finished.value
                                    break
                        if remaining <= 0:
                            raise RuntimeError("生成预算耗尽，音频不完整")
                    frames = yield from self._alert_field(ship, voice, remaining, threshold, bytes(lead),
                                                          number_tail=bool(ship_parts))
                    remaining -= frames
                    if area:
                        if remaining <= 0:
                            raise RuntimeError("生成预算耗尽，音频不完整")
                        frames = yield from self._alert_field(area, voice, remaining, threshold)
                        remaining -= frames
                    if remaining <= 0:
                        raise RuntimeError("生成预算耗尽，音频不完整")
                    yield phrases["触发"][0]
                    yield from self._alert_field(kind, voice, remaining, threshold)
                    yield phrases[ending][0]
                    return
        yield from self._generate_text(text, spec, language, budget)

    def _alert_field(self, field, voice, remaining, threshold, prefix=b"", *, number_tail=False):
        spec = self.voices[voice]
        limit = min(remaining, self._frame_budget(field, spec, "Chinese"))
        key = (self.cache_identities[voice], field)
        if number_tail:
            key += ("number_tail", self.number_tail_speed)
        cache = self.field_cache
        entry = cache.get(key) if cache is not None else None
        if entry is not None and entry[1] <= limit:
            pcm, frames = entry
            logging.info("告警字段磁盘缓存命中 voice=%s field=%s", voice, field)
            yield prefix + pcm
            return frames

        captured = bytearray() if cache is not None else None
        first = bytearray(prefix)
        with closing(self._generate_text(field, spec, "Chinese", limit)) as raw:
            chunks = trim_alert_edges(raw, threshold)
            if number_tail:
                chunks = stretch_number_tail(chunks, self.number_tail_speed, self.ffmpeg)
            for chunk in chunks:
                if captured is not None:
                    if len(captured) + len(chunk) <= cache.max_bytes:
                        captured.extend(chunk)
                    else:
                        captured = None
                if first:
                    first.extend(chunk)
                    # 船名开头裁掉静音后，攒够 0.8 秒正文再开播。
                    if len(first) >= 38400:
                        yield bytes(first)
                        first.clear()
                else:
                    yield chunk
        frames = self.model.stats["frames"]
        if captured is not None:
            cache.put(key, bytes(captured), frames)
        if first:
            yield bytes(first)
        return frames

    def _generate_text(self, text, spec, language, budget):
        chunks = self.model.generate(text, spec["wav"], spec["text"], lang=language,
                                     max_frames=budget, stream=True, reference_codes=spec["codes"],
                                     decoder_prefix=spec["decoder_prefix"], **self.settings)
        try:
            for chunk in chunks:
                yield pcm16(chunk)
            if not self.model.stats.get("eos") or self.model.stats.get("truncated"):
                raise RuntimeError("生成预算耗尽，音频不完整")
        finally:
            try:
                chunks.close()
            finally:
                self.model._reset_codec()
