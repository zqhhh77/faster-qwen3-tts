"""将解码器最后两层转置卷积改写为按采样相位排列的 FP32 矩阵乘法。"""
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def pack_upsampling(source: Path, destination: Path) -> None:
    model = onnx.load(source)
    graph = model.graph
    weights = {tensor.name: tensor for tensor in graph.initializer}
    nodes = []
    converted = 0

    def constant(prefix, name, value):
        result = prefix + name
        graph.initializer.append(numpy_helper.from_array(np.asarray(value), result))
        return result

    for node in graph.node:
        tensor = weights.get(node.input[1]) if len(node.input) > 1 else None
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if (node.op_type != "ConvTranspose" or tensor is None or len(tensor.dims) != 3
                or tensor.data_type != onnx.TensorProto.FLOAT or attrs.get("group", 1) != 1
                or attrs.get("dilations", [1]) != [1] or attrs.get("pads", [0, 0]) != [0, 0]
                or len(attrs.get("strides", [])) != 1):
            nodes.append(node)
            continue
        channels_in, channels_out, kernel = tensor.dims
        stride = attrs["strides"][0]
        if kernel != 2 * stride or channels_out > 192:
            nodes.append(node)
            continue

        converted += 1
        prefix = node.output[0] + "_phase/"
        transposed = prefix + "time_channels"
        nodes.append(helper.make_node("Transpose", [node.input[0]], [transposed], perm=[0, 2, 1]))
        current = prefix + "current"
        previous = prefix + "previous"
        nodes.append(helper.make_node("Pad", [transposed, constant(prefix, "pad_end",
                                                                     np.array([0, 0, 0, 0, 1, 0], np.int64))],
                                      [current]))
        nodes.append(helper.make_node("Pad", [transposed, constant(prefix, "pad_start",
                                                                     np.array([0, 1, 0, 0, 0, 0], np.int64))],
                                      [previous]))
        window = prefix + "window"
        nodes.append(helper.make_node("Concat", [current, previous], [window], axis=2))

        matrix = numpy_helper.to_array(tensor).reshape(channels_in, channels_out, 2, stride)
        matrix = matrix.transpose(2, 0, 3, 1).reshape(2 * channels_in, stride * channels_out)
        projected = prefix + "projected"
        nodes.append(helper.make_node("MatMul", [window, constant(prefix, "weights", matrix)], [projected]))
        biased = prefix + "biased"
        bias = np.tile(numpy_helper.to_array(weights[node.input[2]]), stride)
        nodes.append(helper.make_node("Add", [projected, constant(prefix, "bias", bias)], [biased]))
        sequence = prefix + "sequence"
        nodes.append(helper.make_node("Reshape", [biased, constant(prefix, "shape",
                                                                    np.array([0, -1, channels_out], np.int64))],
                                      [sequence]))
        nodes.append(helper.make_node("Transpose", [sequence], list(node.output), perm=[0, 2, 1]))

    if converted != 2:
        raise ValueError(f"流式解码器应包含两个支持转换的末级上采样层，实际找到 {converted} 个")
    used = {name for node in nodes for name in node.input} | {value.name for value in graph.output}
    kept = [tensor for tensor in graph.initializer if tensor.name in used]
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    onnx.checker.check_model(model)
    onnx.save(model, destination)
