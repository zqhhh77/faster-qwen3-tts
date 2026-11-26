# M5 ONNX runtime source

Source: https://github.com/ASLP-lab/M5-TTS

Commit: `4e7cc91545ad5ecd368d7726129cefb41fc6e1d7`

`infer_onnx.py` includes local support for precomputed reference codes and pipelined streaming with a per-voice decoder prefix. `text_normalization.py` comes from the commit above.
正常生成结束时会输出最后保留的 100 ms 音频，避免上游固定裁尾截掉短编号的尾音；异常和预算耗尽仍按失败处理。
`../default.wav` is upstream `assets/aslp_default_voice.wav`; its transcript is
from upstream `app.py`. Only the CPU ONNX inference files and this demo voice are
included; no model weights are included in the image.
