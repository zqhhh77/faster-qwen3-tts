$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    $env:GOBIN = Join-Path $PWD '.cache/proto-bin'
    go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.6
    if ($LASTEXITCODE) { throw 'protoc-gen-go install failed' }
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5.1
    if ($LASTEXITCODE) { throw 'protoc-gen-go-grpc install failed' }
    uv run python -m grpc_tools.protoc -I proto `
        --python_out=backend/m5/generated --grpc_python_out=backend/m5/generated `
        --plugin="protoc-gen-go=$env:GOBIN/protoc-gen-go.exe" `
        --plugin="protoc-gen-go-grpc=$env:GOBIN/protoc-gen-go-grpc.exe" `
        --go_out=. --go_opt=module=tts --go-grpc_out=. --go-grpc_opt=module=tts proto/inference.proto
    if ($LASTEXITCODE) { throw 'protobuf generation failed' }
    $generatedPath = Join-Path $PWD 'backend/m5/generated/inference_pb2_grpc.py'
    $generatedText = [IO.File]::ReadAllText($generatedPath).Replace('import inference_pb2 as', 'from . import inference_pb2 as')
    [IO.File]::WriteAllText($generatedPath, $generatedText, [Text.UTF8Encoding]::new($false))
} finally { Pop-Location }
