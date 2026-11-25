import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("onnx"), "离线转换测试需要 onnx")
class FastUpsamplingTests(unittest.TestCase):
    def test_two_phase_upsampling_layers_preserve_shape_and_values(self):
        import onnx
        import onnxruntime as ort
        from onnx import helper, numpy_helper, TensorProto
        from .fast_upsampling import pack_upsampling

        rng = np.random.default_rng(456)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "before.onnx"
            target = Path(directory) / "after.onnx"
            initializers = []
            nodes = []
            outputs = []
            for index, (channels_out, stride) in enumerate(((3, 4), (2, 3))):
                weight = rng.normal(0, .1, (4, channels_out, 2 * stride)).astype(np.float32)
                bias = rng.normal(0, .1, channels_out).astype(np.float32)
                initializers.extend((numpy_helper.from_array(weight, f"w{index}"),
                                     numpy_helper.from_array(bias, f"b{index}")))
                nodes.append(helper.make_node("ConvTranspose", ["x", f"w{index}", f"b{index}"],
                                              [f"y{index}"], strides=[stride], pads=[0, 0]))
                outputs.append(helper.make_tensor_value_info(f"y{index}", TensorProto.FLOAT,
                                                              [1, channels_out, "samples"]))
            graph = helper.make_graph(nodes, "upsampling",
                                      [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, "time"])],
                                      outputs, initializers)
            onnx.save(helper.make_model(graph, ir_version=9,
                                        opset_imports=[helper.make_opsetid("", 17)]), source)
            pack_upsampling(source, target)
            sessions = [ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                        for path in (source, target)]
            for frames in (1, 4, 17):
                x = rng.normal(size=(1, 4, frames)).astype(np.float32)
                before, after = (session.run(None, {"x": x}) for session in sessions)
                for expected, actual in zip(before, after):
                    self.assertEqual(expected.shape, actual.shape)
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
