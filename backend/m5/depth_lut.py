"""预计算 M5 INT8 深度解码器中仅依赖上一个码本 token 的首层 Q/K/V 投影。"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


def dependencies(nodes, outputs, inputs):
    needed = set(outputs)
    selected = []
    for node in reversed(nodes):
        if needed.intersection(node.output) and not inputs.intersection(node.output):
            selected.append(node)
            needed.difference_update(node.output)
            needed.update(node.input)
    return list(reversed(selected)), needed


def precompute_depth(source, destination, manifest):
    model = onnx.load(source)
    graph = model.graph
    nodes = list(graph.node)
    weights = {w.name: w for w in graph.initializer}
    producers = {out: node for node in nodes for out in node.output}
    replacements = {}
    tables = []
    vocab = manifest["audio_vocab"]
    if (manifest["codebooks"], vocab) != (16, 2051):
        raise ValueError("查表转换仅支持 M5 的 16 码本、2051 词表导出版本")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    for stage in range(1, 15):
        # 每步展开五层。第一步同时依赖文本隐状态，后续十四步的首层投影可以枚举。
        code = f"/ArgMax_{stage}_output_0"
        outputs = [f"/core/{kind}_proj_{5 * stage}/MatMul_output_0" for kind in "qkv"]
        if not all(name in producers for name in [code, *outputs]):
            raise ValueError(f"第 {stage} 步缺少所需 ONNX 节点")
        subnodes, leaves = dependencies(nodes, outputs, {code})
        if code not in leaves or not (leaves - {code}) <= weights.keys():
            raise ValueError(f"第 {stage} 步投影还依赖其他动态输入，无法预计算")
        subgraph = helper.make_graph(
            subnodes, "depth_projection",
            [helper.make_tensor_value_info(code, TensorProto.INT64, [1, 1])],
            [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, None]) for name in outputs],
            [weights[name] for name in sorted(leaves - {code})])
        submodel = helper.make_model(subgraph, ir_version=model.ir_version, opset_imports=model.opset_import)
        session = ort.InferenceSession(submodel.SerializeToString(), options, providers=providers)
        token = np.zeros((1, 1), np.int64)
        first = session.run(None, {code: token})
        if any(x.dtype != np.float32 or x.shape[:2] != (1, 1) or x.ndim != 3 for x in first):
            raise ValueError("首层投影的类型或形状与模型不符")
        values = [np.empty((vocab, x.shape[2]), np.float32) for x in first]
        for token_id in range(vocab):
            token[0, 0] = token_id
            for table, projection in zip(values, session.run(None, {code: token})):
                table[token_id] = projection[0, 0]
        for name, table in zip(outputs, values):
            if not np.isfinite(table).all():
                raise ValueError(f"预计算投影含无效值 {name}")
            table_name = name + "_lut"
            tables.append(numpy_helper.from_array(table, table_name))
            replacements[producers[name].name] = helper.make_node("Gather", [table_name, code], [name], axis=0)

    rewritten = [replacements.get(n.name, n) for n in nodes]
    keep, leaves = dependencies(rewritten, [v.name for v in graph.output], {v.name for v in graph.input})
    constants = [w for w in [*graph.initializer, *tables] if w.name in leaves]
    del graph.node[:]
    graph.node.extend(keep)
    del graph.initializer[:]
    graph.initializer.extend(constants)
    onnx.checker.check_model(model)
    onnx.save(model, destination)

    original, prepared = [ort.InferenceSession(str(path), options, providers=providers)
                          for path in (source, destination)]
    rng = np.random.default_rng(1234)
    for index in range(16):
        feed = dict(hidden=rng.normal(size=(1, manifest["hidden_size"])).astype(np.float32),
                    noise=rng.exponential(size=(16, 1, vocab)).astype(np.float32),
                    allow_eos=np.array([bool(index % 2)]))
        for before, after in zip(original.run(None, feed), prepared.run(None, feed)):
            if not np.array_equal(before, after):
                raise ValueError("查表前后的音频 token 或下一帧隐状态不一致")
