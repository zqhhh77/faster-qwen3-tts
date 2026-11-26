"""生成 M5 CPU INT8 模型，或为 FP32/INT8 模型预计算投影查找表。"""

import argparse
import json
from pathlib import Path
import shutil
from uuid import uuid4

from .runtime import FILES, digest_file, verify_bundle


def quantize_depth(source: Path, staging: Path) -> None:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = onnx.load(source / "depth_frame.onnx")
    weights = {value.name for value in model.graph.initializer}
    aliases = {}
    kept = []
    outputs = {value.name for value in model.graph.output}
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            if value in aliases:
                node.input[index] = aliases[value]
        if node.op_type == "Identity" and node.input[0] in weights and node.output[0] not in outputs:
            aliases[node.output[0]] = node.input[0]
        else:
            kept.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    folded = staging / "folded.onnx"
    onnx.save(model, folded)
    del model, kept
    quantized = staging / "depth_frame.onnx"
    quantize_dynamic(folded, quantized, per_channel=True, weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"], extra_options={"MatMulConstBOnly": True})
    folded.unlink()
    graph = onnx.load(quantized)
    onnx.checker.check_model(graph)
    if not any(node.op_type == "MatMulInteger" for node in graph.graph.node):
        raise ValueError("depth_frame 模型没有量化矩阵乘法")


def prepare(source: Path, destination: Path, *, depth_lut: bool = False, fast_codec: bool = False,
            fast_upsampling: bool = False, depth_pair: bool = False) -> None:
    if destination.exists():
        raise FileExistsError(f"目标目录已存在: {destination}")
    verify_bundle(source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    precision = manifest.get("depth_frame_precision", "fp32")
    if manifest.get("depth_frame_pair"):
        raise ValueError("请使用尚未合批的模型目录进行转换")
    if depth_pair and precision != "fp32":
        raise ValueError("双请求合批需要 FP32 深度生成模型")
    if depth_lut and (precision not in ("fp32", "int8") or manifest.get("depth_frame_lut")):
        raise ValueError("查表转换需要尚未预计算的 FP32 或 INT8 模型目录")
    if fast_codec and (precision != "int8" or manifest.get("codec_decoder_fast_conv")):
        raise ValueError("解码器转换需要已有 INT8 模型和原版流式音频解码器")
    if fast_upsampling and (precision != "int8" or manifest.get("codec_upsampling_layout")):
        raise ValueError("上采样转换需要尚未改写末级上采样的 INT8 模型")
    if not (depth_lut or fast_codec or fast_upsampling or depth_pair) and precision != "fp32":
        raise ValueError("需要原版 FP32 depth_frame 模型")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The published directory must inherit its parent's deployment permissions.
    staging = destination.with_name(f".{destination.name}-{uuid4().hex}")
    staging.mkdir()
    try:
        changed = {"manifest.json"}
        if depth_lut:
            from .depth_lut import precompute_depth
            precompute_depth(source / "depth_frame.onnx", staging / "depth_frame.onnx", manifest)
            manifest["depth_frame_lut"] = True
            changed.add("depth_frame.onnx")
        elif precision == "fp32" and not depth_pair:
            quantize_depth(source, staging)
            precision = "int8"
            changed.add("depth_frame.onnx")
        if depth_pair:
            from .pair_depth import prepare_pair
            depth_source = staging / "depth_frame.onnx" if depth_lut else source / "depth_frame.onnx"
            prepare_pair(depth_source, staging / "depth_frame_pair.onnx")
            manifest["depth_frame_pair"] = True
            changed.add("depth_frame_pair.onnx")
        if fast_codec or fast_upsampling:
            from .fast_codec import check_audio
            decoder = "codec_decoder_cached.onnx"
            decoder_source = source / decoder
            temporary = staging / "codec_decoder_intermediate.onnx"
            if fast_codec:
                from .fast_codec import pack_convolutions
                pack_convolutions(decoder_source, temporary if fast_upsampling else staging / decoder)
                decoder_source = temporary
                manifest["codec_decoder_fast_conv"] = True
            if fast_upsampling:
                from .fast_upsampling import pack_upsampling
                pack_upsampling(decoder_source, staging / decoder)
                manifest["codec_upsampling_layout"] = "phase_fp32_last_two"
                temporary.unlink(missing_ok=True)
            snr = check_audio(source, staging / decoder)
            print(f"参考语音流式解码校验通过，波形信噪比 {snr:.2f} dB", flush=True)
            changed.add(decoder)
        for name in FILES[:-1]:
            if name not in changed:
                shutil.copyfile(source / name, staging / name)
        manifest["depth_frame_precision"] = precision
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        checks = json.loads((source / "checksums.json").read_text(encoding="utf-8"))
        for name in changed:
            checks[name] = {"bytes": (staging / name).stat().st_size, "sha256": digest_file(staging / name)}
        (staging / "checksums.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
        verify_bundle(staging)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--depth-lut", action="store_true", help="预计算首层投影查找表，保留源模型的 FP32 或 INT8 精度")
    parser.add_argument("--depth-pair", action="store_true", help="为 FP32 模型生成双请求深度合批图")
    parser.add_argument("--fast-codec", action="store_true", help="将 INT8 模型的流式音频解码器转换为混合精度快速版本")
    parser.add_argument("--fast-upsampling", action="store_true", help="将解码器最后两层上采样改写为 FP32 相位矩阵")
    args = parser.parse_args()
    prepare(args.source.resolve(strict=True), args.destination.resolve(),
            depth_lut=args.depth_lut, fast_codec=args.fast_codec, fast_upsampling=args.fast_upsampling,
            depth_pair=args.depth_pair)
