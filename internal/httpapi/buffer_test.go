package httpapi

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"tts/internal/cache"
	"tts/internal/config"
	"tts/internal/engine"
)

type bufferBackend struct {
	info     *engine.Info
	generate func(context.Context, engine.Request, *engine.Info, func([]byte) error) error
}

func (b *bufferBackend) Info(context.Context) (*engine.Info, error) { return b.info, nil }
func (b *bufferBackend) CachedInfo() *engine.Info                   { return b.info }
func (b *bufferBackend) Generate(ctx context.Context, r engine.Request, i *engine.Info, emit func([]byte) error) error {
	return b.generate(ctx, r, i, emit)
}
func (b *bufferBackend) Close() error { return nil }

func TestStartupBufferKeepsGenerationRunningAndStreamsBeforeCompletion(t *testing.T) {
	for _, format := range []string{"pcm", "pcm_framed", "sse"} {
		t.Run(format, func(t *testing.T) {
			first, second := make(chan time.Time, 1), make(chan struct{})
			next, finish := make(chan struct{}), make(chan struct{})
			b := &bufferBackend{info: &engine.Info{Identity: strings.Repeat("a", 64), Voices: []string{"default"}, DefaultVoice: "default", Languages: []string{"Chinese"}}}
			b.generate = func(ctx context.Context, _ engine.Request, _ *engine.Info, emit func([]byte) error) error {
				if err := emit([]byte{1, 2}); err != nil {
					return err
				}
				first <- time.Now()
				select {
				case <-next:
				case <-ctx.Done():
					return ctx.Err()
				}
				if err := emit([]byte{3, 4}); err != nil {
					return err
				}
				close(second)
				select {
				case <-finish:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			}
			cfg := config.Config{Directory: t.TempDir(), Concurrency: 1, Readers: 4, Capacity: 32768, Output: 4096,
				TTL: time.Hour, Task: 5 * time.Second, Send: time.Second, Sweep: time.Hour, StartupBuffer: 200 * time.Millisecond}
			store, err := cache.New(cfg, b.Generate)
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			server := httptest.NewServer(New(cfg, b, store))
			defer server.Close()
			defer store.Cancel()
			path, responseFormat := "/v1/audio/speech", format
			if format == "sse" {
				path, responseFormat = "/v1/audio/speech/sse", "pcm_framed"
			}
			body := `{"input":"测试","response_format":"` + responseFormat + `"}`
			type response struct {
				value *http.Response
				err   error
			}
			responses := make(chan response, 1)
			client := http.Client{Timeout: 3 * time.Second}
			go func() {
				r, e := client.Post(server.URL+path, "application/json", strings.NewReader(body))
				responses <- response{r, e}
			}()
			var began time.Time
			select {
			case began = <-first:
			case <-time.After(time.Second):
				t.Fatal("推理未开始")
			}
			close(next)
			select {
			case <-second:
			case <-time.After(time.Second):
				t.Fatal("缓冲阻塞了模型继续生成")
			}
			var resp *http.Response
			select {
			case got := <-responses:
				if got.err != nil {
					t.Fatal(got.err)
				}
				resp = got.value
			case <-time.After(time.Second):
				t.Fatal("开播仍在等待整句结束")
			}
			defer resp.Body.Close()
			if time.Since(began) < 150*time.Millisecond {
				t.Fatal("没有等待开播缓冲")
			}
			if resp.StatusCode != 200 {
				t.Fatal(resp.Status)
			}
			close(finish)
			data, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}
			switch format {
			case "pcm":
				if !bytes.Equal(data, []byte{1, 2, 3, 4}) {
					t.Fatal(data)
				}
			case "pcm_framed":
				if len(data) != 14 || data[0] != 1 || binary.BigEndian.Uint32(data[1:5]) != 4 || !bytes.Equal(data[5:9], []byte{1, 2, 3, 4}) || data[9] != 2 {
					t.Fatal(data)
				}
			case "sse":
				if !bytes.Equal(data, append(sse("chunk", "音频片段已生成", []byte{1, 2, 3, 4}), sse("done", "语音生成完成", nil)...)) {
					t.Fatal(string(data))
				}
			}
			// 相同请求已完成，命中缓存时跳过开播等待。
			ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
			defer cancel()
			req, _ := http.NewRequestWithContext(ctx, "POST", server.URL+path, strings.NewReader(body))
			cached, err := client.Do(req)
			if err != nil {
				t.Fatal("完整缓存仍被延迟", err)
			}
			defer cached.Body.Close()
			cachedData, err := io.ReadAll(cached.Body)
			if err != nil || !bytes.Equal(data, cachedData) {
				t.Fatal("缓存音频不一致", err)
			}
		})
	}
}

func TestStartupBufferFailureReturnsJSONBeforeAudio(t *testing.T) {
	b := &bufferBackend{info: &engine.Info{Identity: strings.Repeat("a", 64), Voices: []string{"default"}, DefaultVoice: "default", Languages: []string{"Chinese"}}}
	b.generate = func(_ context.Context, _ engine.Request, _ *engine.Info, emit func([]byte) error) error {
		if err := emit([]byte{1, 2}); err != nil {
			return err
		}
		time.Sleep(20 * time.Millisecond)
		return errors.New("模拟推理失败")
	}
	cfg := config.Config{Directory: t.TempDir(), Concurrency: 1, Readers: 2, Capacity: 32768, Output: 4096,
		TTL: time.Hour, Task: time.Second, Send: time.Second, Sweep: time.Hour, StartupBuffer: 200 * time.Millisecond}
	store, err := cache.New(cfg, b.Generate)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	server := httptest.NewServer(New(cfg, b, store))
	defer server.Close()
	resp, err := server.Client().Post(server.URL+"/v1/audio/speech", "application/json", strings.NewReader(`{"input":"测试"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var body envelope
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil || resp.StatusCode != 503 || body.Code != 503 || body.Message == "" || body.Data != nil {
		t.Fatalf("status=%d body=%+v err=%v", resp.StatusCode, body, err)
	}
}
