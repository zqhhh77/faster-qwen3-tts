import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.m5.runtime import M5Onnx


class StreamTailTests(unittest.TestCase):
    def model(self, frames, invalid_tail=False):
        model = M5Onnx.__new__(M5Onnx)
        model.meta = dict(depth_frame_precision="int8", layers=1, kv_heads=1, head_dim=1,
                          max_sequence=64, audio_vocab=2051, min_frames=6, hop_length=1920)
        model.sample_rate = 24000
        model.provider, model.device, model.device_id = "CPUExecutionProvider", "cpu", 0
        model.session_providers = model.provider_options = {}
        model.threads, model.codec_mode = 1, "cached"
        model.tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[0]))
        model.text_embeddings = np.zeros((1, 1), dtype=np.float32)
        model.audio_embeddings = np.zeros((16 * 2051, 1), dtype=np.float32)
        model.offsets = np.arange(16) * 2051
        state = dict(generated=0, decoded=0)

        def backbone(outputs, feed):
            return np.zeros((1, 1), dtype=np.float32), feed["past"]

        def depth(outputs, feed):
            eos = state["generated"] == frames
            state["generated"] += 1
            return np.full((1, 16), 0 if eos else 1, dtype=np.int64), np.zeros((1, 1, 1), np.float32)

        def decode(codes):
            state["decoded"] += len(codes)
            model.codec_offset += len(codes)
            pcm = np.full(len(codes) * 1920, 0.2, dtype=np.float32)
            if invalid_tail and state["decoded"] == frames:
                pcm[-2400:] = np.nan
            return pcm

        model.sessions = {"backbone": SimpleNamespace(run=backbone), "depth_frame": SimpleNamespace(run=depth)}
        model._decode = decode
        return model

    def generate(self, model, max_frames=32):
        return model.generate("一五二四零九，", "unused.wav", "参考文本", stream=True,
                              first_frames=6, chunk_frames=4, max_frames=max_frames,
                              reference_codes=np.zeros((8, 16), np.int64), decoder_prefix=(8, None, []))

    @patch("text_normalization.normalize_text", side_effect=lambda text, **kwargs: text)
    def test_eos_keeps_last_100ms_for_full_and_partial_batches(self, normalize):
        for frames in (9, 10):
            with self.subTest(frames=frames):
                model = self.model(frames)
                chunks = list(self.generate(model))
                self.assertEqual(len(chunks[0]), 6 * 1920 - 2400)
                self.assertEqual(sum(map(len, chunks)), frames * 1920)
                self.assertEqual(len(chunks[-1]), 2400)
                self.assertTrue(model.stats["eos"])
                self.assertAlmostEqual(model.stats["audio_s"], frames * 0.08)

    @patch("text_normalization.normalize_text", side_effect=lambda text, **kwargs: text)
    def test_invalid_tail_is_rejected(self, normalize):
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            list(self.generate(self.model(9, invalid_tail=True)))

    @patch("text_normalization.normalize_text", side_effect=lambda text, **kwargs: text)
    def test_truncation_does_not_flush_tail_or_report_success(self, normalize):
        model = self.model(20)
        with self.assertWarnsRegex(RuntimeWarning, "max_frames"):
            chunks = list(self.generate(model, max_frames=10))
        self.assertFalse(model.stats["eos"])
        self.assertTrue(model.stats["truncated"])
        self.assertEqual(sum(map(len, chunks)), 10 * 1920 - 2400)

    @patch("text_normalization.normalize_text", side_effect=lambda text, **kwargs: text)
    def test_fp32_without_decoder_prefix_overlaps_generation_and_decoding(self, normalize):
        model = self.model(9)
        model.meta["depth_frame_precision"] = "fp32"
        model.codec_meta = dict(layers=1, kv_heads=1, head_dim=1, states=[])
        generating, decoding = Event(), Event()
        backbone = model.sessions["backbone"].run
        decode = model._decode
        calls = 0

        def next_frame(outputs, feed):
            nonlocal calls
            calls += 1
            if calls == 3:
                generating.set()
                self.assertTrue(decoding.wait(2), "下一批生成没有与解码重叠执行")
            return backbone(outputs, feed)

        def decode_chunk(codes):
            decoding.set()
            self.assertTrue(generating.wait(2), "解码前没有启动下一批生成")
            return decode(codes)

        model.sessions["backbone"].run = next_frame
        model._decode = decode_chunk
        chunks = list(model.generate("请保持关注。", "unused.wav", "参考文本", stream=True,
                                     first_frames=2, chunk_frames=4, max_frames=32,
                                     reference_codes=np.zeros((8, 16), np.int64)))
        self.assertTrue(model.stats["eos"])
        self.assertEqual(sum(map(len, chunks)), 9 * 1920)


if __name__ == "__main__":
    unittest.main()
