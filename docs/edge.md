# Edge TTS 后端

Edge TTS 通过微软在线语音服务合成，需要部署环境能访问该服务。后端读取本地 JSON 音色映射表，键是对外名称，值是实际用于合成的 Edge 预设音色短名称。使用 H5 参考录音试听选择预设音色。

复制 [映射表示例](../references/edge-voices.example.json) 并填写实际配置。表中必须包含 `default`，可以删除示例中的 `broadcast` 或添加其他需要公开的音色。仅配置 `default` 时，客户只会查询到这一项。

```powershell
Copy-Item references/edge-voices.example.json references/edge-voices.json
$env:EDGE_TTS_VOICES_FILE = './references/edge-voices.json'
docker compose -f compose.edge.yaml up -d --pull always --no-build
```

`EDGE_TTS_WORKERS` 默认为 4，Compose 将同一个值设为 Gateway 合成并发。对外端口默认为 `127.0.0.1:8000`。修改映射表后运行 `docker compose -f compose.edge.yaml up -d --force-recreate edge-backend`，Gateway 下次查询会读取更新后的公开名称，缓存也会按新映射重新计算。

## 出网代理

服务器可以访问微软语音服务时，使用基础 Compose 即可直连。`EDGE_TTS_PROXY` 可留空。

如果已经部署了 HTTP CONNECT 代理，在本项目的 `.env` 配置代理地址，然后重新运行基础 Compose。HTTP 和 HTTPS 代理地址均可配置。

```dotenv
EDGE_TTS_PROXY=http://192.168.10.20:9006
```

如果只在一台可出网电脑部署 Moto，使用 TCP 转发。下面是 Moto 配置示例，监听地址填中转电脑的内网 IP，`allowlist` 填 Edge 服务器访问 Moto 时的来源 IP。

```json
{
  "rules": [
    {
      "name": "Edge TTS",
      "listen": "192.168.10.20:9010",
      "mode": "normal",
      "protocol": "tcp",
      "allowlist": ["192.168.10.30/32"],
      "targets": [{"address": "speech.platform.bing.com:443"}]
    }
  ]
}
```

在 Edge 服务器上的 `.env` 设置 `EDGE_TTS_MOTO_ADDR=192.168.10.20:9010`，将 `EDGE_TTS_PROXY` 留空，再叠加 Moto 配置启动。

```sh
docker compose -f compose.edge.yaml -f compose.edge.moto.yaml up -d --pull always --no-build
```

该配置启动一个容器内的 TCP 转发服务，把 Edge 的连接送到配置的 Moto 地址。Edge 继续使用微软原域名进行 TLS 校验，Moto 转发到微软的 443 端口。Moto 的 HTTP CONNECT 模式用于接入已有的 H2/H3 CONNECT 上游代理。参见 [Moto 部署文档](https://github.com/cppla/moto)。

两种部署方式都从 GHCR 拉取配套的 `linux/amd64` 镜像。默认使用 `edge` 标签。设置 `TTS_IMAGE_TAG=sha-<完整提交 SHA>` 可固定版本。

## 调用示例

在 PowerShell 查询、保存流式 WAV 和保存非流式 WAV。

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/audio/voices
python examples/http_streaming_client.py --text '前方船舶请注意安全。' --voice default --output stream.wav
$body = @{input='前方船舶请注意安全。'; voice='default'; stream=$false} | ConvertTo-Json
$result = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/audio/speech -ContentType 'application/json; charset=utf-8' -Body $body
[IO.File]::WriteAllBytes((Join-Path $PWD 'complete.wav'), [Convert]::FromBase64String($result.data.audio_b64))
```

其他调用方式见 [HTTP API](api.md)。
