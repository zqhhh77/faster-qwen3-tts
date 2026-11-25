"""将流式解码器中的稠密一维卷积改写为按输出通道量化的 CPU 矩阵乘法。"""
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

from .runtime import read_reference


def pack_convolutions(source, destination):
    model = onnx.load(source)
    graph = model.graph
    weights = {tensor.name: tensor for tensor in graph.initializer}
    nodes = []
    converted = 0

    def constant(name, value):
        graph.initializer.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    for node in graph.node:
        if node.op_type != "Conv" or node.input[1] not in weights:
            nodes.append(node)
            continue
        tensor = weights[node.input[1]]
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if (len(tensor.dims) != 3 or tensor.data_type != onnx.TensorProto.FLOAT
                or attrs.get("group", 1) != 1 or attrs.get("strides", [1]) != [1]
                or any(attrs.get("pads", [])) or attrs.get("auto_pad", b"NOTSET") != b"NOTSET"):
            nodes.append(node)
            continue
        channels_out, channels_in, kernel = tensor.dims
        if not (kernel == 7 and channels_out >= 32 or kernel == 1 and channels_in == channels_out):
            nodes.append(node)
            continue
        converted += 1
        prefix = node.output[0] + "_fast/"
        q, scale, zero = (prefix + suffix for suffix in ("quant", "scale", "zero"))
        nodes.append(helper.make_node("DynamicQuantizeLinear", [node.input[0]], [q, scale, zero]))
        data = prefix + "time_channels"
        nodes.append(helper.make_node("Transpose", [q], [data], perm=[0, 2, 1]))
        if kernel > 1:
            # 先量化再展开卷积窗口，相邻窗口复用同一批 uint8 样本。
            dilation = attrs.get("dilations", [1])[0]
            axis = constant(prefix + "axis", np.array([1], np.int64))
            columns = []
            for tap in range(kernel):
                start = constant(prefix + f"start{tap}", np.array([dilation * tap], np.int64))
                stop = -dilation * (kernel - 1 - tap) if tap < kernel - 1 else np.iinfo(np.int64).max
                end = constant(prefix + f"end{tap}", np.array([stop], np.int64))
                column = prefix + f"tap{tap}"
                nodes.append(helper.make_node("Slice", [data, start, end, axis], [column]))
                columns.append(column)
            data = prefix + "windows"
            nodes.append(helper.make_node("Concat", columns, [data], axis=2))

        matrix = numpy_helper.to_array(tensor).transpose(2, 1, 0).reshape(-1, channels_out)
        weight_scale = np.maximum(np.abs(matrix).max(axis=0) / 127, np.float32(1e-12))
        # 使用无符号输入和权重，避免依赖较新 CPU 的有符号点积指令。
        packed = (np.rint(matrix / weight_scale).astype(np.int16) + 128).astype(np.uint8)
        inputs = [data, constant(prefix + "weights", packed), scale,
                  constant(prefix + "weight_scale", weight_scale), zero,
                  constant(prefix + "weight_zero", np.uint8(128)), *node.input[2:]]
        projected = prefix + "projected"
        nodes.append(helper.make_node("MatMulIntegerToFloat", inputs, [projected], domain="com.microsoft"))
        nodes.append(helper.make_node("Transpose", [projected], list(node.output), perm=[0, 2, 1]))

    if not converted:
        raise ValueError("模型中未找到支持转换的流式解码卷积")
    if not any(opset.domain == "com.microsoft" for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    used = {name for node in nodes for name in node.input} | {value.name for value in graph.output}
    kept = [tensor for tensor in graph.initializer if tensor.name in used]
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    onnx.checker.check_model(model)
    onnx.save(model, destination)


def check_audio(source, candidate):
    """用真实语音编码验证连续分块解码、短尾块、输出长度和波形误差。"""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    providers = ["CPUExecutionProvider"]
    reference = read_reference(Path(__file__).with_name("default.wav"))
    encoder = ort.InferenceSession(str(source / "codec_encoder.onnx"), opts, providers=providers)
    codes = encoder.run(None, {"audio": reference[None]})[0][:, :, :int(np.ceil(len(reference) / 1920))]
    del encoder
    meta = json.loads((source / "codec_cached.json").read_text(encoding="utf-8"))

    def decode(path):
        session = ort.InferenceSession(str(path), opts, providers=providers)
        past = np.empty((meta["layers"], 2, 1, meta["kv_heads"], 0, meta["head_dim"]), np.float32)
        states = [np.zeros(spec["shape"], np.float32) for spec in meta["states"]]
        pcm = []
        offset = 0
        while offset < codes.shape[-1]:
            size = 1 if offset == 0 else 4
            part = np.ascontiguousarray(codes[:, :, offset:offset + size])
            feed = dict(zip(meta["inputs"], [part, np.array([offset], np.int64), past, *states]))
            audio, past, *states = session.run(None, feed)
            if audio.size != part.shape[-1] * 1920 or not np.isfinite(audio).all():
                raise ValueError("流式解码器返回的音频长度或数值无效")
            pcm.append(audio.ravel())
            offset += part.shape[-1]
        return np.concatenate(pcm)

    before = decode(source / "codec_decoder_cached.onnx")
    after = decode(candidate)
    power = float(np.dot(before, before))
    error = before - after
    snr = 10 * np.log10(max(power, 1e-20) / max(float(np.dot(error, error)), 1e-20))
    if power <= 0 or snr < 20:
        raise ValueError(f"参考语音的解码对照信噪比过低 {snr:.2f} dB")
    return snr
