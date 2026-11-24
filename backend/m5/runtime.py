"""One CPU model instance; the service holds its slot for the whole generator."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

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
    def __init__(self):
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
        identity = {
            "backend": "m5-cpu", "assets": assets, "voices": identity_voices,
            "default_voice": self.default_voice, "threads": self.threads,
            "max_frames": self.max_frames, "settings": self.settings,
            "normalization": self.normalization,
            "source": {str(p.relative_to(Path(__file__).parent)): digest_file(p)
                       for p in (Path(__file__), Path(__file__).with_name("server.py"),
                                 VENDOR / "infer_onnx.py", VENDOR / "text_normalization.py")},
            "packages": {name: importlib.metadata.version(name) for name in (
                "numpy", "onnxruntime", "scipy", "tokenizers", "soundfile")},
        }
        if self.normalization:
            identity["normalizer_packages"] = {name: importlib.metadata.version(name)
                                               for name in ("pynini", "WeTextProcessing")}
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.model_name = root.name
        logging.info("M5 CPU 已就绪 identity=%s voices=%s normalization=%s",
                     self.identity, list(self.voices), self.normalization)

    def generate(self, text: str, voice: str, language: str):
        spec = self.voices[voice]
        normalized = text_normalization.normalize_text(text, lang=language)
        ids = self.model.tokenizer.encode(conditioned_text(spec["text"].strip() + normalized.strip(), language)).ids
        available = self.model.meta["max_sequence"] - len(ids) - len(spec["codes"])
        if available <= self.model.meta["min_frames"]:
            raise ValueError("文本和参考音频超出模型上下文容量")
        budget = min(self.max_frames, available)
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
