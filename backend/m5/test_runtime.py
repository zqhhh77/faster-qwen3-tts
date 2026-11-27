import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.m5.alert_phrases import ALERT_PHRASES, AlertFieldCache, alert_parts, trim_alert_edges
from backend.m5.runtime import Runtime, pcm16, verify_bundle


class FakeModel:
    sample_rate = 24000
    meta = {"max_sequence": 2048, "min_frames": 6, "depth_frame_precision": "int8"}
    tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[1, 2]))

    def __init__(self):
        self.sessions = {"codec_encoder": object()}
        self.provider_options = {}
        self.encodes = 0
        self.truncated = False
        self.resets = 0
        self.prefixes = []
        self.used = []

    def encode_reference(self, audio):
        self.encodes += 1
        return np.full((12, 16), self.encodes, dtype=np.int64)

    def prepare_decoder_prefix(self, codes):
        if codes.shape != (8, 16):
            raise AssertionError("unexpected reference context")
        self.prefixes.append(object())
        return self.prefixes[-1]

    def generate(self, *args, **kwargs):
        voice = int(kwargs["reference_codes"][0, 0])
        if kwargs["decoder_prefix"] is not self.prefixes[voice - 1]:
            raise AssertionError("wrong voice decoder prefix")
        self.used.append(voice)
        yield np.array([0.1, -0.1], np.float32)
        self.stats = {"eos": not self.truncated, "truncated": self.truncated, "frames": 7}

    def _reset_codec(self):
        self.resets += 1


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache_root = Path(directory.name)
        environment = patch.dict(os.environ, {"M5_ALERT_CACHE_DIR": str(self.cache_root),
                                               "M5_ALERT_SHIP_PARTS": "off", "M5_ALERT_NUMBER_TAIL_SPEED": "0.65",
                                               "M5_DEPTH_BATCH": "off", "M5_DEVICE": "cpu", "M5_DEVICE_ID": "0"})
        environment.start()
        self.addCleanup(environment.stop)

    def disk_cache(self, name="fields", max_bytes=1 << 20):
        return AlertFieldCache(self.cache_root / name, max_bytes)

    def test_pair_model_asset_is_verified_and_required_when_enabled(self):
        root = self.cache_root
        (root / "manifest.json").write_text('{"depth_frame_pair": true}', encoding="utf-8")
        (root / "depth_frame_pair.onnx").write_bytes(b"paired-model")
        checks = {name: {"bytes": (root / name).stat().st_size,
                         "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
                  for name in ("manifest.json", "depth_frame_pair.onnx")}
        (root / "checksums.json").write_text(json.dumps(checks), encoding="utf-8")
        with patch("backend.m5.runtime.FILES", ("manifest.json", "checksums.json")):
            self.assertIn("depth_frame_pair.onnx", verify_bundle(root))
            (root / "depth_frame_pair.onnx").write_bytes(b"changed-model")
            with self.assertRaisesRegex(ValueError, "模型文件校验失败"):
                verify_bundle(root)
        with patch.dict(os.environ, {"M5_MODEL": str(root), "M5_DEPTH_BATCH": "on"}), \
                patch("backend.m5.runtime.verify_bundle", return_value={}):
            with self.assertRaisesRegex(ValueError, "开启深度合批需要"):
                Runtime()

    def test_startup_defaults_to_four_phrases_per_voice_shared_by_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "voice.wav").write_bytes(b"reference")
            voices = root / "voices.json"
            voices.write_text(json.dumps({name: {"wav": "voice.wav", "text": "reference"}
                                          for name in ("a", "b")}), encoding="utf-8")
            with patch.dict(os.environ, {"M5_MODEL": directory, "M5_VOICES_FILE": str(voices),
                                         "M5_DEFAULT_VOICE": "a", "M5_TEXT_NORMALIZATION": "off"}), \
                    patch("backend.m5.runtime.verify_bundle", return_value={}), \
                    patch("backend.m5.runtime.M5Onnx", side_effect=lambda *a, **kw: FakeModel()), \
                    patch("backend.m5.runtime.read_reference", return_value=np.zeros(100)):
                os.environ.pop("M5_ALERT_PHRASES", None)
                os.environ.pop("M5_ALERT_FIELD_CACHE_MB", None)
                first = Runtime()
                other = Runtime(shared_alert_phrases=first.alert_phrases)
                self.assertEqual(first.model.used, [1] * 4 + [2] * 4)
                self.assertEqual(other.model.used, [])
                self.assertIs(first.alert_phrases, other.alert_phrases)
                self.assertIsNone(first.field_cache)
                restarted = Runtime()
                self.assertEqual(restarted.model.used, [])
                for voice in ("a", "b"):
                    for phrase in ALERT_PHRASES:
                        self.assertEqual(first._alert_phrase(voice, phrase), restarted._alert_phrase(voice, phrase))
                self.assertEqual(len(list((self.cache_root / "phrases").glob("*.pcm"))), 8)
                self.assertEqual(first.alert_phrases._entries, {})
                for invalid in ("nan", "0.49", "0.99", "1.1"):
                    with patch.dict(os.environ, {"M5_ALERT_NUMBER_TAIL_SPEED": invalid}):
                        with self.assertRaisesRegex(ValueError, "M5_ALERT_NUMBER_TAIL_SPEED"):
                            Runtime()
                with patch.dict(os.environ, {"M5_ALERT_SHIP_PARTS": "on"}):
                    with self.assertRaisesRegex(ValueError, "M5_ALERT_FIELD_CACHE_MB"):
                        Runtime()

    def test_reference_and_model_changes_invalidate_disk_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b"):
                (root / (name + ".wav")).write_bytes(name.encode())
            voices = root / "voices.json"
            definitions = {name: {"wav": name + ".wav", "text": "reference"} for name in ("a", "b")}
            voices.write_text(json.dumps(definitions), encoding="utf-8")
            with patch.dict(os.environ, {"M5_MODEL": directory, "M5_VOICES_FILE": str(voices),
                                        "M5_DEFAULT_VOICE": "a", "M5_TEXT_NORMALIZATION": "off",
                                        "M5_ALERT_PHRASES": "on", "M5_ALERT_FIELD_CACHE_MB": "1"}), \
                    patch("backend.m5.runtime.verify_bundle", return_value={}) as assets, \
                    patch("backend.m5.runtime.M5Onnx", side_effect=lambda *a, **kw: FakeModel()), \
                    patch("backend.m5.runtime.read_reference", return_value=np.zeros(100)):
                first = Runtime()
                text = "发现甲船，在北港，触发偏航，请立即关注"
                list(first.generate(text, "b", "Chinese"))
                restarted = Runtime()
                list(restarted.generate(text, "b", "Chinese"))
                self.assertEqual(restarted.model.used, [])
                (root / "a.wav").write_bytes(b"new reference")
                changed = Runtime()
                self.assertEqual(changed.model.used, [1] * 4)
                list(changed.generate(text, "b", "Chinese"))
                self.assertEqual(changed.model.used, [1] * 4)
                definitions["b"]["text"] = "new transcript"
                voices.write_text(json.dumps(definitions), encoding="utf-8")
                self.assertEqual(Runtime().model.used, [2] * 4)
                assets.return_value = {"model": "new"}
                self.assertEqual(Runtime().model.used, [1] * 4 + [2] * 4)

    def test_all_voices_preloaded_and_reused_without_reencoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "voice.wav").write_bytes(b"reference")
            voice_file = root / "voices.json"
            voice_file.write_text(json.dumps({name: {"wav": "voice.wav", "text": "reference"}
                                             for name in ("a", "b")}), encoding="utf-8")
            model = FakeModel()
            original_encode = model.encode_reference
            with patch.dict(os.environ, {"M5_MODEL": str(root), "M5_VOICES_FILE": str(voice_file),
                                        "M5_DEFAULT_VOICE": "b", "M5_TEXT_NORMALIZATION": "off",
                                        "M5_ALERT_PHRASES": "off"}), \
                    patch("backend.m5.runtime.verify_bundle", return_value={}), \
                    patch("backend.m5.runtime.M5Onnx", return_value=model), \
                    patch("backend.m5.runtime.read_reference", return_value=np.zeros(100)):
                runtime = Runtime()
                self.assertEqual(model.encodes, 2)
                self.assertEqual(runtime.default_voice, "b")
                interrupted = runtime.generate("text", "b", "Chinese")
                next(interrupted)
                interrupted.close()
                for name in ("a", "b", "a"):
                    self.assertTrue(list(runtime.generate("text", name, "Chinese")))
                self.assertEqual(model.used, [2, 1, 2, 1])
                self.assertEqual(model.encodes, 2)
                self.assertEqual(model.encode_reference, original_encode)
                model.truncated = True
                with self.assertRaises(RuntimeError):
                    list(runtime.generate("text", "a", "Chinese"))
                self.assertEqual(model.encode_reference, original_encode)
                self.assertEqual(model.resets, 5)

    def test_alert_fields_and_fallback(self):
        for text in ("发现海巡一号,触发偏航,请保持关注", "发现海巡一号，，触发偏航，请保持关注。"):
            self.assertEqual(alert_parts(text), ("海巡一号，", "", "偏航，", "请保持关注"))
        self.assertEqual(alert_parts("发现海巡一号，在北港，触发超速，请立即关注。"),
                         ("海巡一号，", "在北港，", "超速，", "请立即关注"))
        for text in ("发现了一条船", "发现，在东海，触发偏航，请立即关注",
                     "发现甲船，触发，请立即关注", "发现甲船，在，触发偏航，请保持关注",
                     "发现甲船，触发偏航，请保持关注。请回复。", "发现甲船，触发偏航，请注意"):
            self.assertIsNone(alert_parts(text))

    def test_alert_audio_uses_selected_voice_and_only_generates_fields(self):
        runtime = Runtime.__new__(Runtime)
        runtime.ship_parts = False
        runtime.field_cache = self.disk_cache()
        runtime.alert_phrases = self.disk_cache("phrases", None)
        runtime.cache_identities = {"a": "a", "b": "b"}
        runtime.model = SimpleNamespace(meta={"min_frames": 6}, stats={})
        runtime.max_frames = 100
        runtime.voices = {"a": "a", "b": "b"}
        runtime._frame_budget = lambda text, spec, lang: 100
        calls = []
        tones = {}

        def audio(voice, text):
            tone = tones.setdefault((voice, text), len(tones) + 1)
            return np.full(240, tone * 100, dtype="<i2").tobytes()

        def generate(text, voice, language, budget):
            calls.append((text, voice, language, budget))
            yield audio(voice, text)
            runtime.model.stats["frames"] = 10

        runtime._generate_text = generate
        runtime._prepare_alert_phrases()
        self.assertEqual(len(calls), 2 * len(ALERT_PHRASES))
        calls.clear()
        for ending in ("请立即关注", "请保持关注"):
            self.assertEqual(list(runtime.generate(ending, "a", "Chinese")), [audio("a", ending + "。")])
        self.assertFalse(calls)
        text = "发现海巡一号，在北港，触发偏航，请保持关注。"
        self.assertEqual(b"".join(runtime.generate(text, "b", "Chinese")),
                         b"".join(audio("b", part) for part in
                                  ("发现", "海巡一号，", "在北港，", "触发", "偏航，", "请保持关注。")))
        self.assertEqual(calls, [("海巡一号，", "b", "Chinese", 70), ("在北港，", "b", "Chinese", 60),
                                 ("偏航，", "b", "Chinese", 50)])
        calls.clear()
        changed = text.replace("保持", "立即")
        actual = b"".join(runtime.generate(changed, "b", "Chinese"))
        self.assertEqual(calls, [])
        self.assertTrue(actual.endswith(audio("b", "请立即关注。")))
        new_ship = changed.replace("海巡一号", "海巡二号")
        list(runtime.generate(new_ship, "b", "Chinese"))
        self.assertEqual([call[0] for call in calls], ["海巡二号，"])
        calls.clear()
        list(runtime.generate(text, "a", "Chinese"))
        self.assertEqual([call[0] for call in calls], ["海巡一号，", "在北港，", "偏航，"])
        calls.clear()
        self.assertEqual(list(runtime._alert_field("偏航，", "a", 10, 0)), [audio("a", "偏航，")])
        self.assertEqual(calls, [])
        list(runtime.generate(text, "a", "English"))
        self.assertEqual(calls, [(text, "a", "English", 100)])
        runtime.max_frames = 20
        calls.clear()
        list(runtime.generate(text, "a", "Chinese"))
        self.assertEqual(calls, [(text, "a", "Chinese", 20)])

        runtime.max_frames = 100
        runtime.ship_parts = True
        runtime.number_tail_speed = 1
        runtime.ffmpeg = None
        calls.clear()
        numbered = text.replace("海巡一号", "鲁威渔一五二四零九")
        list(runtime.generate(numbered, "b", "Chinese"))
        self.assertEqual([call[0] for call in calls], ["鲁威渔", "一五二四零九，"])
        calls.clear()
        list(runtime.generate(numbered.replace("零九", "零八"), "b", "Chinese"))
        self.assertEqual([call[0] for call in calls], ["一五二四零八，"])
        calls.clear()
        list(runtime.generate(numbered.replace("鲁威渔", "浙普渔"), "b", "Chinese"))
        self.assertEqual([call[0] for call in calls], ["浙普渔"])
        calls.clear()
        list(runtime.generate(numbered, "b", "Chinese"))
        self.assertEqual(calls, [])

    def test_alert_silence_trim_preserves_interior_and_soft_clips(self):
        quiet = np.zeros(2400, dtype="<i2").tobytes()
        spoken = np.full(240, -3456, dtype="<i2").tobytes()
        result = b"".join(trim_alert_edges([quiet, spoken, quiet, quiet, spoken + quiet], 30))
        self.assertEqual(result, quiet[-2880:] + spoken + quiet * 2 + spoken + quiet[:2880])
        soft = np.full(2400, -10, dtype="<i2").tobytes()
        self.assertEqual(b"".join(trim_alert_edges([soft, soft], 30)), soft * 2)

    def test_alert_prefix_streams_before_subject_finishes_and_releases_generator_on_cancel(self):
        runtime = Runtime.__new__(Runtime)
        runtime.ship_parts = False
        runtime.field_cache = self.disk_cache()
        runtime.alert_phrases = self.disk_cache("phrases", None)
        runtime.cache_identities = {"a": "a"}
        runtime.model = SimpleNamespace(meta={"min_frames": 6})
        runtime.max_frames = 100
        runtime.voices = {"a": "a"}
        runtime._frame_budget = lambda *args: 100
        fixed = np.full(2400, 1000, dtype="<i2").tobytes()
        field = np.full(12000, 2000, dtype="<i2").tobytes()
        for phrase in ALERT_PHRASES:
            runtime.alert_phrases.put(("a", phrase), fixed, 7)
        closed = []

        def generate(text, *args):
            try:
                yield field
                yield field
                self.fail("首包应在船名段生成完毕前发出")
            finally:
                closed.append(text)

        runtime._generate_text = generate
        stream = runtime.generate("发现甲船，触发偏航，请立即关注", "a", "Chinese")
        self.assertEqual(next(stream), fixed + field * 2)
        stream.close()
        self.assertEqual(closed, ["甲船，"])
        self.assertIsNone(runtime.field_cache.get(("a", "甲船，")))

        def failed(*args):
            yield field
            raise RuntimeError("truncated")

        runtime._generate_text = failed
        with self.assertRaises(RuntimeError):
            list(runtime._alert_field("甲船，", "a", 100, 0))
        self.assertIsNone(runtime.field_cache.get(("a", "甲船，")))

    def test_disk_cache_persists_lru_quota_and_ignores_duplicates(self):
        capacity = 2 * AlertFieldCache._header.size + 6
        cache = self.disk_cache(max_bytes=capacity)
        cache.put("a", b"aa", 1)
        cache.put("b", b"bb", 2)
        os.utime(cache._path("b"), (1, 1))
        self.assertEqual(cache.get("a"), (b"aa", 1))
        cache = self.disk_cache(max_bytes=capacity)
        cache.put("c", b"cccc", 3)
        cache.put("a", b"other!", 4)
        cache.put("large", b"12" * capacity, 5)
        self.assertEqual(cache.get("a"), (b"aa", 1))
        self.assertIsNone(cache.get("b"))
        self.assertIsNone(cache.get("large"))
        self.assertEqual(cache.get("c"), (b"cccc", 3))
        self.assertLessEqual(sum(p.stat().st_size for p in cache.directory.glob("*.pcm")), capacity)
        self.assertTrue(all(isinstance(size, int) for size in cache._entries.values()))

    def test_corrupt_and_incomplete_disk_entries_are_not_reused(self):
        cache = self.disk_cache()
        for corrupted in (b"partial", cache._header.pack(cache._magic, 7, 2, b"0" * 32) + b"aa"):
            cache._path("a").write_bytes(corrupted)
            self.assertIsNone(cache.get("a"))
            self.assertFalse(cache._path("a").exists())
        with patch("backend.m5.alert_phrases.os.replace", side_effect=OSError("disk full")):
            cache.put("a", b"aa", 1)
        self.assertIsNone(cache.get("a"))
        self.assertEqual(list(cache.directory.glob("*.tmp")), [])
        cache.put("a", b"aa", 1)
        self.assertEqual(cache.get("a"), (b"aa", 1))

    def test_workers_share_complete_disk_entries_without_pcm_in_index(self):
        cache = self.disk_cache()

        def write_and_read(index):
            key = ("voice" + str(index % 2), "在北港，")
            pcm = bytes([index % 2, 0]) * 24000
            cache.put(key, pcm, 16)
            self.assertEqual(cache.get(key), (pcm, 16))

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(write_and_read, range(20)))
        self.assertEqual(len(cache._entries), 2)
        self.assertEqual(list(cache.directory.glob("*.tmp")), [])
        restored = self.disk_cache()
        for voice in ("voice0", "voice1"):
            self.assertEqual(cache.get((voice, "在北港，")), restored.get((voice, "在北港，")))

    def test_pcm16_clips_and_rejects_nonfinite(self):
        values = np.frombuffer(pcm16(np.array([-2, -1, 0, 1, 2], np.float32)), dtype="<i2")
        self.assertEqual(values.tolist(), [-32768, -32767, 0, 32767, 32767])
        for data in ([float("nan")], [float("inf")], [], [[1, 2]]):
            with self.assertRaises(ValueError):
                pcm16(data)

    def test_bundle_allows_only_line_ending_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = b'{\n  "sample_rate": 24000\n}\n'
            (root / "manifest.json").write_bytes(original.replace(b"\n", b"\r\n"))
            checks = {"manifest.json": {"bytes": len(original), "sha256": hashlib.sha256(original).hexdigest()}}
            (root / "checksums.json").write_text(json.dumps(checks))
            with patch("backend.m5.runtime.FILES", ("manifest.json", "checksums.json")):
                verify_bundle(root)
                (root / "manifest.json").write_bytes(original.replace(b"24000", b"16000"))
                with self.assertRaises(ValueError):
                    verify_bundle(root)


if __name__ == "__main__":
    unittest.main()
