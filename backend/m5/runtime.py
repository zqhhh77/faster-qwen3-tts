"""One CPU model instance; the service holds its slot for the whole generator."""
from __future__ import annotations

import hashlib
from contextlib import closing
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from .alert_phrases import ALERT_ENDINGS, ALERT_PHRASES, AlertFieldCache, alert_parts, silence_threshold, trim_alert_edges

VENDOR = Path(__file__).with_name("vendor")
sys.path.insert(0, str(VENDOR))
from infer_onnx import FILES, M5Onnx, conditioned_text, read_reference
import text_normalization

LANGUAGES = ("Chinese", "English", "German", "French", "Japanese", "Korean")


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_bundle(root: Path) -> dict:
    checks = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    for name in FILES[:-1]:
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
    return {name: checks[name]["sha256"] for name in FILES[:-1]}


def pcm16(chunk) -> bytes:
    values = np.asarray(chunk, dtype=np.float32)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("模型输出的音频无效")
    return np.clip(values * 32767.0, -32768, 32767).astype("<i2").tobytes()


class Runtime:
    def __init__(self, *, shared_alert_phrases=None, shared_field_cache=None):
        root = Path(os.environ.get("M5_MODEL", "/models/m5")).resolve(strict=True)
        assets = verify_bundle(root)
        self.threads = int(os.environ.get("M5_THREADS", "2"))
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
                            device="cpu", local_files_only=True)
        voice_file = Path(os.environ.get("M5_VOICES_FILE", str(Path(__file__).with_name("voices.json")))).resolve(strict=True)
        definitions = json.loads(voice_file.read_text(encoding="utf-8"))
        if not isinstance(definitions, dict) or not 1 <= len(definitions) <= 64:
            raise ValueError("音色配置必须包含 1 到 64 个音色")
        self.voices = {}
        identity_voices = {}
        use_int8 = self.model.meta.get("depth_frame_precision") == "int8"
        self.settings = dict(seed=1234, first_frames=6 if use_int8 else 2, chunk_frames=4, left_context=8)
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
        self.default_voice = os.environ.get("M5_DEFAULT_VOICE") or next(iter(self.voices))
        if self.default_voice not in self.voices:
            raise ValueError("默认音色不在配置列表中")
        phrase_mode = os.environ.get("M5_ALERT_PHRASES", "on")
        if phrase_mode not in ("on", "off"):
            raise ValueError("M5_ALERT_PHRASES 必须为 on 或 off")
        identity = {
            "backend": "m5-cpu", "assets": assets, "voices": identity_voices,
            "default_voice": self.default_voice, "threads": self.threads,
            "max_frames": self.max_frames, "settings": self.settings,
            "normalization": self.normalization, "alert_phrases": phrase_mode,
            "source": {str(p.relative_to(Path(__file__).parent)): digest_file(p)
                       for p in (Path(__file__), Path(__file__).with_name("server.py"),
                                 Path(__file__).with_name("alert_phrases.py"),
                                 VENDOR / "infer_onnx.py", VENDOR / "text_normalization.py")},
            "packages": {name: importlib.metadata.version(name) for name in (
                "numpy", "onnxruntime", "scipy", "tokenizers", "soundfile")},
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
        logging.info("M5 CPU 已就绪 identity=%s voices=%s normalization=%s",
                     self.identity, list(self.voices), self.normalization)

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
                phrases = {p: self._alert_phrase(voice, p) for p in ("发现", "触发", ending)}
                remaining = budget - sum(phrases[p][1] for p in ("发现", "触发", ending))
                if remaining > (3 if area else 2) * self.model.meta["min_frames"]:
                    threshold = silence_threshold(phrases["发现"][0])
                    frames = yield from self._alert_field(ship, voice, remaining, threshold, phrases["发现"][0])
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

    def _alert_field(self, field, voice, remaining, threshold, prefix=b""):
        spec = self.voices[voice]
        limit = min(remaining, self._frame_budget(field, spec, "Chinese"))
        key = (self.cache_identities[voice], field)
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
            for chunk in trim_alert_edges(raw, threshold):
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
