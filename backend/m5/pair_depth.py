"""将两路独立深度生成中的 FP32 常量权重投影合批。"""
import copy

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort


def prepare_pair(source, destination):
    model = onnx.load(source)
    graph = model.graph
    inputs = {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim] for value in graph.input}
    if inputs != {"hidden": [1, 512], "noise": [16, 1, 2051], "allow_eos": [1]}:
        raise ValueError("合批转换需要 M5 单请求深度生成模型")
    weights = {weight.name: weight for weight in graph.initializer}
    constant_weights = dict(weights)
    for node in graph.node:
        if node.op_type == "Identity" and node.input[0] in constant_weights:
            constant_weights[node.output[0]] = constant_weights[node.input[0]]

    def mapped(name, branch):
        return name if not name or name in weights else f"r{branch}/{name}"

    nodes = []
    count = 0
    for ordinal, node in enumerate(graph.node):
        weight = constant_weights.get(node.input[1]) if len(node.input) > 1 else None
        if (node.op_type == "MatMul" and weight is not None and len(weight.dims) == 2
                and weight.data_type == onnx.TensorProto.FLOAT):
            prefix = f"paired_projection_{ordinal}"
            nodes.extend([
                helper.make_node("Concat", [mapped(node.input[0], i) for i in range(2)],
                                 [prefix + "/inputs"], name=prefix + "/concat", axis=0),
                helper.make_node("MatMul", [prefix + "/inputs", weight.name],
                                 [prefix + "/outputs"], name=prefix + "/matmul"),
                helper.make_node("Split", [prefix + "/outputs", "paired_split_sizes"],
                                 [mapped(node.output[0], i) for i in range(2)], name=prefix + "/split", axis=0),
            ])
            count += 1
        else:
            for branch in range(2):
                cloned = copy.deepcopy(node)
                cloned.name = f"r{branch}/{node.name or ordinal}"
                for index, value in enumerate(cloned.input):
                    cloned.input[index] = mapped(value, branch)
                for index, value in enumerate(cloned.output):
                    cloned.output[index] = mapped(value, branch)
                nodes.append(cloned)
    if not count:
        raise ValueError("模型中没有可合批的 FP32 常量权重投影")
    inputs, outputs = [], []
    for branch in range(2):
        for before, after in ((graph.input, inputs), (graph.output, outputs)):
            for value in before:
                cloned = copy.deepcopy(value)
                cloned.name = mapped(value.name, branch)
                after.append(cloned)
    paired = helper.make_model(helper.make_graph(
        nodes, "paired_depth", inputs, outputs,
        [*graph.initializer, numpy_helper.from_array(np.array([1, 1], np.int64), "paired_split_sizes")]),
        ir_version=model.ir_version, opset_imports=model.opset_import)
    onnx.checker.check_model(paired)
    onnx.save(paired, destination)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    original, prepared = [ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
                          for path in (source, destination)]
    rng = np.random.default_rng(2314)
    for index in range(64):
        feeds = [dict(hidden=rng.normal(size=(1, 512)).astype(np.float32),
                      noise=rng.exponential(size=(16, 1, 2051)).astype(np.float32),
                      allow_eos=np.array([bool((index + branch) % 2)])) for branch in range(2)]
        expected = [value for feed in feeds for value in original.run(None, feed)]
        actual = prepared.run(None, {f"r{i}/{key}": value for i, feed in enumerate(feeds) for key, value in feed.items()})
        if len(actual) != len(expected) or any(not np.array_equal(a, b) for a, b in zip(expected, actual)):
            raise ValueError("合批前后的音频 token 或下一帧隐状态不一致")
    print(f"双请求深度生成转换完成，共 {count} 组投影，64 组输出一致性校验通过", flush=True)
