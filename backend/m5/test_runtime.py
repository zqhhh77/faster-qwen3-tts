import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.m5.runtime import Runtime, pcm16, verify_bundle


class FakeModel:
    sample_rate = 24000
    meta = {"max_sequence": 2048, "min_frames": 6, "depth_frame_precision": "int8"}
    tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[1, 2]))

    def __init__(self):
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
        self.stats = {"eos": not self.truncated, "truncated": self.truncated}

    def _reset_codec(self):
        self.resets += 1


class RuntimeTests(unittest.TestCase):
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
                                        "M5_DEFAULT_VOICE": "b", "M5_TEXT_NORMALIZATION": "off"}), \
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
