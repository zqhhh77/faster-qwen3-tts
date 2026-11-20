# tdt-tts 流式网关

tdt-tts 支持 Qwen3-TTS（vLLM-Omni）和 M5（ONNX Runtime CPU / gRPC），每次部署选择一个模型后端。两者共用 HTTP 接口、命名音色、增量音频、共享生成与磁盘缓存。M5 的部署、本地 uv 环境及音色配置见 [M5 CPU 使用说明](docs/m5.md)。

M5 支持在目标机器运行 `uv run --frozen python -m backend.m5.tune --concurrency N`，自动对比 workers/threads，并输出吞吐、P95 延迟、进程树内存及推荐配置；也可直接在 CPU 镜像中运行。用法见 [一键调优](docs/m5.md#一键寻找-workers--threads-组合)。

推理镜像固定为 vLLM-Omni v0.28.0 官方发布版及其摘要。旧 faster 推理框架、Python 网关、GGML、Gradio 和 Python 包发布流程已移除。

## 工程目录

```text
cmd/gateway/            Go 进程入口、启动与退出
internal/config/       配置解析与校验
internal/engine/       vLLM-Omni HTTP/SSE 客户端、模型身份、请求定义
internal/cache/        磁盘缓存、任务队列、共享生成、补读和过期清理
internal/httpapi/      对外 HTTP 接口、参数校验、PCM/SSE 协议
backend/               官方推理服务启动适配、命名音色预计算
deploy/                两端 Dockerfile、Nginx 流式代理示例
docs/                  架构、接口、部署和音色使用文档
examples/              独立客户端示例
references/            参考录音及配置模板
.github/workflows/     云端测试、镜像构建与 GHCR 发布
compose.yaml           从源码构建并部署
compose.images.yaml    拉取已发布镜像部署
.env.example           部署配置模板
```

Go 测试与实现放在同一包中；模块名为 `tts`。Go 通过标准 HTTP 接入 Qwen，通过 gRPC/protobuf 接入 M5。`backend/launch.py` 保留 Qwen 启动逻辑，`backend/m5/` 提供独立 CPU 推理服务，`compose.m5.yaml` 部署 M5 与同一个网关。

## Qwen3-TTS 后端快速部署

准备 NVIDIA 驱动、NVIDIA Container Toolkit、Docker Engine 和支持 `gpus: all` 的 Docker Compose v2.30+。GPU/驱动需要兼容固定的官方 vLLM-Omni 镜像；请按 [部署文档](docs/deployment.md) 做启动检查。

```bash
git clone -b feat/vllm-omni-backend <你的仓库地址> tdt-tts
cd tdt-tts
cp .env.example .env
# 编辑 .env，将 TTS_IMAGE_PREFIX 填为实际镜像前缀，例如 ghcr.io/owner/repo（全部小写）。
mkdir -p cache references /data/qwen3-tts/models
# 把完整模型下载到 /data/qwen3-tts/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
docker compose -f compose.images.yaml pull
docker compose -f compose.images.yaml up -d
docker compose -f compose.images.yaml logs -f backend
```

`.env` 默认使用本地 0.6B CustomVoice 模型，音色 `vivian`，最多 4 个不同任务并行推理、8 个等待任务。相同文本、音色、语言和指令的请求共享一次推理，输出协议不同也能共享。

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/audio/voices
python examples/http_streaming_client.py --url http://127.0.0.1:8000 \
  --text '前方船舶请注意安全。' --voice vivian --output speech.wav
```

## 文档

- [Linux 部署、hfd 下载、升级与配置](docs/deployment.md)
- [内置音色与多个自定义音色](docs/voices.md)
- [HTTP、PCM 帧和 SSE 接口](docs/api.md)
- [模块职责、并发与缓存语义](docs/architecture.md)

## 验证与发布

本地可通过 uv 运行 M5 Python 测试和真实 CPU 流式检查，并执行 Go 测试。新工作流 `M5 CPU and gateway images` 在 Linux 上检查后构建并发布 M5 CPU 与网关镜像；旧 Qwen 构建工作流仅保留手动触发，稳定后端无需重新打包。

新 CI 按当前 GitHub 仓库发布 `ghcr.io/<owner>/<repo>-m5` 和 `ghcr.io/<owner>/<repo>-gateway`（全部小写）。部署时设置 `TTS_IMAGE_PREFIX` 和相同的提交标签。既有 Qwen 的 `-backend` 镜像继续使用原来已发布的版本。

旧工作流手动执行时仍支持按内容复用 Qwen 后端镜像。本次不自动触发该构建。M5 权重通过挂载提供，CPU 镜像不包含模型权重。

M5 迁移分支发布 `m5` 标签，`main` 发布 `latest`。M5 和网关两个镜像准备成功后才移动标签；固定版本使用 `sha-完整提交SHA`。既有 Qwen 部署继续使用原来的 `vllm-omni` 或固定版本标签。

CPU CI 不验证 GPU 显存、模型启动、音色效果或实时性。这些需要在远端实际 GPU 上验收。
