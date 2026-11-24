"""从已校验的 M5 FP32 模型生成可独立挂载的 CPU INT8 模型目录。"""

import argparse
import json
from pathlib import Path
import shutil
from uuid import uuid4

from .runtime import FILES, digest_file, verify_bundle


def prepare(source: Path, destination: Path) -> None:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if destination.exists():
        raise FileExistsError(f"目标目录已存在: {destination}")
    verify_bundle(source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("depth_frame_precision", "fp32") != "fp32":
        raise ValueError("需要原版 FP32 depth_frame 模型")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The published directory must inherit its parent's deployment permissions.
    staging = destination.with_name(f".{destination.name}-{uuid4().hex}")
    staging.mkdir()
    try:
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
        del graph
        for name in FILES[:-1]:
            if name not in ("manifest.json", "depth_frame.onnx"):
                shutil.copyfile(source / name, staging / name)
        manifest["depth_frame_precision"] = "int8"
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        checks = json.loads((source / "checksums.json").read_text(encoding="utf-8"))
        for name in ("manifest.json", "depth_frame.onnx"):
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
    args = parser.parse_args()
    prepare(args.source.resolve(strict=True), args.destination.resolve())
