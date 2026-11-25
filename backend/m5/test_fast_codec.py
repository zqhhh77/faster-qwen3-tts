import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("onnx"), "离线转换测试需要 onnx")
class FastCodecTests(unittest.TestCase):
    def test_dilated_taps_pointwise_bias_and_short_tail(self):
        import onnx
        import onnxruntime as ort
        from onnx import helper, numpy_helper, TensorProto
        from .fast_codec import pack_convolutions

        rng = np.random.default_rng(123)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "before.onnx"
            target = Path(directory) / "after.onnx"
            for kernel, dilation in [(7, 9), (1, 1)]:
                weight = rng.normal(0, .1, (32, 32, kernel)).astype(np.float32)
                weight[0] = 0
                bias = rng.normal(0, .1, 32).astype(np.float32)
                graph = helper.make_graph(
                    [helper.make_node("Conv", ["x", "w", "b"], ["y"], dilations=[dilation])],
                    "codec", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32, "time"])],
                    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32, "samples"])],
                    [numpy_helper.from_array(weight, "w"), numpy_helper.from_array(bias, "b")])
                onnx.save(helper.make_model(graph, ir_version=9, opset_imports=[helper.make_opsetid("", 17)]), source)
                pack_convolutions(source, target)
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                sessions = [ort.InferenceSession(str(p), options, providers=["CPUExecutionProvider"])
                            for p in (source, target)]
                for frames in (1, 4, 31):
                    x = rng.normal(size=(1, 32, frames + dilation * (kernel - 1))).astype(np.float32)
                    before, after = (session.run(None, {"x": x})[0] for session in sessions)
                    self.assertEqual(after.shape, (1, 32, frames))
                    self.assertLess(np.linalg.norm(after - before), .025 * np.linalg.norm(before))
                    np.testing.assert_array_equal(after[0, 0], before[0, 0])
