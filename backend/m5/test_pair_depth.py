import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("onnx"), "离线转换测试需要 onnx")
class PairDepthTests(unittest.TestCase):
    def test_shared_weight_preserves_two_independent_inputs_and_eos_masks(self):
        import onnx
        from onnx import helper, numpy_helper, TensorProto
        import onnxruntime as ort
        from .pair_depth import prepare_pair

        nodes = [
            helper.make_node("Identity", ["weight"], ["alias"]),
            helper.make_node("MatMul", ["hidden", "alias"], ["projected"]),
            helper.make_node("Unsqueeze", ["projected", "axis"], ["next_hidden"]),
            helper.make_node("Gather", ["noise", "zero"], ["first_noise"], axis=0),
            helper.make_node("ArgMax", ["first_noise"], ["token"], axis=1),
            helper.make_node("Tile", ["token", "repeats"], ["tokens"]),
            helper.make_node("Where", ["allow_eos", "tokens", "zeros"], ["codes"]),
        ]
        arrays = {"weight": np.eye(512, dtype=np.float32), "axis": np.array([1], np.int64),
                  "zero": np.array(0, np.int64), "repeats": np.array([1, 16], np.int64),
                  "zeros": np.zeros((1, 16), np.int64)}
        graph = helper.make_graph(nodes, "single",
            [helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, 512]),
             helper.make_tensor_value_info("noise", TensorProto.FLOAT, [16, 1, 2051]),
             helper.make_tensor_value_info("allow_eos", TensorProto.BOOL, [1])],
            [helper.make_tensor_value_info("codes", TensorProto.INT64, [1, 16]),
             helper.make_tensor_value_info("next_hidden", TensorProto.FLOAT, [1, 1, 512])],
            [numpy_helper.from_array(value, name) for name, value in arrays.items()])
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "single.onnx", Path(directory) / "paired.onnx"
            onnx.save(helper.make_model(graph, ir_version=9, opset_imports=[helper.make_opsetid("", 17)]), source)
            prepare_pair(source, target)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(str(target), options, providers=["CPUExecutionProvider"])
            feeds = {}
            for branch in range(2):
                feeds[f"r{branch}/hidden"] = np.full((1, 512), branch + 1, np.float32)
                noise = np.zeros((16, 1, 2051), np.float32)
                noise[:, :, 123] = 1
                feeds[f"r{branch}/noise"] = noise
                feeds[f"r{branch}/allow_eos"] = np.array([bool(branch)])
            codes_a, hidden_a, codes_b, hidden_b = session.run(None, feeds)
            np.testing.assert_array_equal(codes_a, np.zeros((1, 16), np.int64))
            np.testing.assert_array_equal(codes_b, np.full((1, 16), 123, np.int64))
            np.testing.assert_array_equal(hidden_a, np.ones((1, 1, 512), np.float32))
            np.testing.assert_array_equal(hidden_b, np.full((1, 1, 512), 2, np.float32))
