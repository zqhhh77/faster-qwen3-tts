import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .prepare_int8 import prepare
from .runtime import FILES, digest_file, verify_bundle


@unittest.skipUnless(importlib.util.find_spec("onnx"), "离线转换测试需要 onnx")
class PrepareTests(unittest.TestCase):
    def test_lookup_preserves_source_precision_and_decoder_assets(self):
        for precision in (None, "fp32", "int8"):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "original"
                source.mkdir()
                manifest = {} if precision is None else {"depth_frame_precision": precision}
                for name in FILES[:-1]:
                    (source / name).write_bytes(name.encode())
                (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                checks = {name: {"sha256": digest_file(source / name), "bytes": (source / name).stat().st_size}
                          for name in FILES[:-1]}
                (source / "checksums.json").write_text(json.dumps(checks), encoding="utf-8")
                destination = Path(directory) / "lookup"

                def lookup(before, after, metadata):
                    self.assertEqual(metadata, manifest)
                    after.write_bytes(b"lookup")

                with patch("backend.m5.depth_lut.precompute_depth", side_effect=lookup), \
                        patch("backend.m5.prepare_int8.quantize_depth") as quantize:
                    prepare(source, destination, depth_lut=True)
                    quantize.assert_not_called()

                result = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(result["depth_frame_precision"], precision or "fp32")
                self.assertTrue(result["depth_frame_lut"])
                verify_bundle(destination)
                self.assertEqual(verify_bundle(source), {name: value["sha256"] for name, value in checks.items()})
                for name in FILES[:-1]:
                    if name not in ("manifest.json", "depth_frame.onnx"):
                        self.assertEqual((source / name).read_bytes(), (destination / name).read_bytes())
                with self.assertRaisesRegex(ValueError, "尚未预计算"):
                    prepare(destination, Path(directory) / "second-lookup", depth_lut=True)


if __name__ == "__main__":
    unittest.main()
