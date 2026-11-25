import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from backend.m5.alert_phrases import ALERT_ENDINGS, AlertFieldCache
from backend.m5.runtime import Runtime
from backend.m5.test_runtime import FakeModel


class MergedAlertTests(unittest.TestCase):
    def runtime(self):
        runtime = Runtime.__new__(Runtime)
        runtime.alert_mode = "merged"
        runtime.voices = {"a": "a", "b": "b"}
        runtime.model = SimpleNamespace(meta={"min_frames": 6}, stats={})
        runtime.max_frames = 100
        runtime._frame_budget = lambda *args: 100
        runtime.field_cache = AlertFieldCache(1024 * 1024)
        runtime.alert_phrases = {
            voice: {phrase: ((index + offset).to_bytes(2, "little") * 2400, 7)
                    for index, phrase in enumerate(("发现", *ALERT_ENDINGS), 1000)}
            for voice, offset in (("a", 0), ("b", 100))
        }
        self.calls = []
        self.body = b"\x20\x4e" * 24000

        def raw(text, spec, language, budget):
            self.calls.append((text, spec, language, budget))
            yield self.body
            runtime.model.stats["frames"] = 20

        runtime._generate_text = raw
        return runtime

    def test_startup_prepares_three_phrases_once_and_separates_mode_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "voice.wav").write_bytes(b"reference")
            voices = root / "voices.json"
            voices.write_text(json.dumps({name: {"wav": "voice.wav", "text": "reference"}
                                          for name in ("a", "b")}), encoding="utf-8")
            with patch.dict(os.environ, {"M5_MODEL": directory, "M5_VOICES_FILE": str(voices),
                                         "M5_DEFAULT_VOICE": "a", "M5_TEXT_NORMALIZATION": "off",
                                         "M5_ALERT_PHRASES": "merged", "M5_ALERT_FIELD_CACHE_MB": "1"}), \
                    patch("backend.m5.runtime.verify_bundle", return_value={}), \
                    patch("backend.m5.runtime.M5Onnx", side_effect=lambda *args, **kwargs: FakeModel()), \
                    patch("backend.m5.runtime.read_reference", return_value=np.zeros(100)):
                first = Runtime()
                other = Runtime(shared_alert_phrases=first.alert_phrases, shared_field_cache=first.field_cache)
                self.assertEqual(first.alert_mode, "merged")
                self.assertEqual(first.model.used, [1] * 3 + [2] * 3)
                self.assertEqual(other.model.used, [])
                self.assertIs(first.alert_phrases, other.alert_phrases)
                self.assertIs(first.field_cache, other.field_cache)
                for voice in ("a", "b"):
                    self.assertEqual(set(first.alert_phrases[voice]), {"发现", *ALERT_ENDINGS})
                os.environ["M5_ALERT_PHRASES"] = "on"
                original = Runtime()
                self.assertNotEqual(first.identity, original.identity)

    def test_one_generation_preserves_all_words_and_selected_voice(self):
        runtime = self.runtime()
        text = "发现海巡一号，在北港，触发超速告警，请立即关注。"
        audio = b"".join(runtime.generate(text, "b", "Chinese"))
        self.assertEqual(self.calls, [("海巡一号，在北港，触发超速告警，", "b", "Chinese", 86)])
        self.assertEqual(audio, runtime.alert_phrases["b"]["发现"][0] + self.body
                         + runtime.alert_phrases["b"]["请立即关注"][0])

    def test_body_cache_reuses_changed_ending_but_isolates_voices(self):
        runtime = self.runtime()
        text = "发现海巡一号，，触发偏航，请保持关注。"
        list(runtime.generate(text, "a", "Chinese"))
        self.calls.clear()
        list(runtime.generate(text.replace("保持", "立即"), "a", "Chinese"))
        self.assertEqual(self.calls, [])
        list(runtime.generate(text, "b", "Chinese"))
        self.assertEqual(self.calls, [("海巡一号，触发偏航，", "b", "Chinese", 86)])

    def test_low_budget_and_non_template_keep_original_fallback(self):
        runtime = self.runtime()
        runtime.max_frames = 20
        text = "发现甲船，触发偏航，请立即关注"
        list(runtime.generate(text, "a", "Chinese"))
        self.assertEqual(self.calls, [(text, "a", "Chinese", 20)])
        self.calls.clear()
        list(runtime.generate("普通正文", "a", "Chinese"))
        self.assertEqual(self.calls, [("普通正文", "a", "Chinese", 20)])
        self.calls.clear()
        list(runtime.generate(text, "a", "English"))
        self.assertEqual(self.calls, [(text, "a", "English", 20)])

    def test_cached_ending_still_returns_without_inference(self):
        runtime = self.runtime()
        audio = list(runtime.generate("请保持关注。", "a", "Chinese"))
        self.assertEqual(audio, [runtime.alert_phrases["a"]["请保持关注"][0]])
        self.assertEqual(self.calls, [])

    def test_first_audio_precedes_body_completion_and_cancel_does_not_cache(self):
        runtime = self.runtime()
        closed = []

        def raw(text, *args):
            try:
                yield self.body[:24000]
                yield self.body[:24000]
                self.fail("首包前不应等待动态正文全部生成")
            finally:
                closed.append(text)

        runtime._generate_text = raw
        stream = runtime.generate("发现甲船，触发偏航，请立即关注", "a", "Chinese")
        self.assertTrue(next(stream))
        stream.close()
        self.assertEqual(closed, ["甲船，触发偏航，"])
        self.assertIsNone(runtime.field_cache.get(("a", "甲船，触发偏航，")))

    def test_incomplete_body_raises_without_caching_or_appending_ending(self):
        runtime = self.runtime()

        def raw(*args):
            yield self.body
            raise RuntimeError("生成预算耗尽，音频不完整")

        runtime._generate_text = raw
        stream = runtime.generate("发现甲船，触发偏航，请立即关注", "a", "Chinese")
        self.assertEqual(next(stream), runtime.alert_phrases["a"]["发现"][0] + self.body)
        with self.assertRaisesRegex(RuntimeError, "音频不完整"):
            next(stream)
        self.assertIsNone(runtime.field_cache.get(("a", "甲船，触发偏航，")))


if __name__ == "__main__":
    unittest.main()
