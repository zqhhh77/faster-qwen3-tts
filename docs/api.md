# tdt-tts 流式网关 HTTP API

网关默认地址为 `http://127.0.0.1:8000`。由服务端选择 Qwen3、M5 或 Edge TTS 后端，客户端通过 Gateway 调用统一接口。音色配置见 [M5 CPU 后端](m5.md) 和 [Edge TTS 后端](edge.md)。

所有普通 JSON 响应使用：

```json
{"code":200,"message":"操作成功","data":null}
```

code 与 HTTP 状态码一致，message 为中文。列表返回顶层 rows，不使用 data.items。参数错误 400，音色不存在 404，方法不允许 405，队列满/后端不可用/缓存容量不足 503。流开始前的错误也是该 JSON 格式。音频流按下面的二进制或 SSE 协议返回。

## 查询

- `GET /`：返回网关名称 `tdt-tts 流式网关` 及各接口路径。
- `GET /health`：推理健康状态、模型身份、音色信息、缓存条目数、reader 数、未完成任务数、running_tasks、pending_tasks 和缓存预算占用。
- `GET /v1/audio/voices`：音色列表。每项有 voice、type、default。

## 合成请求

`POST /v1/audio/speech` 和 `POST /v1/audio/speech/sse` 使用相同请求字段：

```json
{
  "input": "前方船舶请注意安全。",
  "voice": "vivian",
  "language": "Chinese",
  "instruct": "",
  "speed": 1.0,
  "response_format": "pcm_framed"
}
```

| 字段 | 规则 |
| --- | --- |
| input | 必填，1–2048 字符，不能全为空白；原文本参与缓存键 |
| voice | 可省略或 null，使用默认音色；指定值须与查询列表一致 |
| language | 默认 Chinese，忽略大小写；可查询 health 中的 languages |
| instruct | 默认空，最多 512 字符；仅 1.7B CustomVoice 接受非空值 |
| speed | 当前只支持 1.0 |
| stream | 默认 true。对 `/v1/audio/speech` 传 false 可获取完整 WAV。SSE 接口要求 true |
| response_format | 流式 PCM 支持 pcm_framed（默认）和 pcm。非流式请求省略该字段 |

未知字段、重复 JSON 字段、非法类型和超过 32 KiB 的请求体被拒绝。使用 `voice` 字段，不是 `viose`。

## 非流式 WAV

`POST /v1/audio/speech` 请求体示例

```json
{"input":"前方船舶请注意安全。","stream":false}
```

完整合成后返回 HTTP 200。

```json
{"code":200,"message":"语音生成成功","data":{"format":"wav","sample_rate":24000,"audio_b64":"BASE64_WAV"}}
```

`data.audio_b64` 解码后是完整的 24 kHz 单声道 16 位 PCM WAV。生成失败时返回 HTTP 503 和中文 JSON，`data` 为 `null`。相同合成参数的流式和非流式请求可以共享生成任务和缓存。

## 带完成标记的 PCM

`response_format=pcm_framed`，Content-Type 为 `application/vnd.qwen.pcm-stream`。响应头：

```text
X-Audio-Protocol: qwen-pcm-v1
X-Audio-Sample-Rate: 24000
X-Audio-Channels: 1
X-Audio-Sample-Format: s16le
```

每帧是 1 字节类型 + 4 字节大端长度 + payload：

| 类型 | payload |
| --- | --- |
| 1 | 非空、偶数字节的 PCM，最多 262144 字节 |
| 2 | 长度必须为 0，表示显式成功完成 |
| 3 | 中文错误 JSON，code=500、data=null，丢弃已收到的部分音频 |

帧内 code 描述流中错误；此时 HTTP 头已发出，HTTP 状态仍是 200。不能把 EOF 当成完成，必须接收到类型 2。TCP/HTTP 读取边界不等于帧边界，需要按长度重组。

[Python 客户端](../examples/http_streaming_client.py) 能保存 WAV，使用 `--play` 时逐块播放；播放依赖 sounddevice，保存只需标准库。

## 原始 PCM

`response_format=pcm` 返回 `audio/pcm`、协议 raw，音频格式仍是 24 kHz / mono / s16le。流中失败时中断连接，不把 JSON 混进 PCM。因为没有业务完成帧，需要强完成校验的调用方应选 pcm_framed。

## SSE 分块 WAV

`POST /v1/audio/speech/sse` 返回 `text/event-stream`：

```text
data: {"code":200,"message":"音频片段已生成","data":{"type":"chunk","sample_rate":24000,"audio_b64":"BASE64_WAV"}}

data: {"code":200,"message":"语音生成完成","data":{"type":"done"}}
```

audio_b64 每块都是独立、完整的 WAV，解码后提取 PCM 顺序播放；不能直接把多个 WAV 文件的全部字节拼成一个 WAV 文件。

流中出错：

```text
data: {"code":200,"message":"语音生成失败，请丢弃已接收音频","data":{"type":"error"}}
```

SSE 的 HTTP 状态已是 200，因此事件顶层 code 也为 200，data.type=error 表示生成失败，此后不会发送 done。完成前连接断开同样视为失败。

代理须关闭响应缓冲，见 [Nginx 示例](../deploy/nginx-streaming.conf)。首块即时发送，后续块随推理生成而到达；缓存命中则从磁盘读取。
