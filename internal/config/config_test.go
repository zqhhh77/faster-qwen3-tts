package config

import "testing"

func TestQueueZeroUnitsAndInvalidConfig(t *testing.T) {
	t.Setenv("TTS_MAX_PENDING_TASKS", "0")
	t.Setenv("TTS_CACHE_MAX_MB", "3")
	c, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if c.Pending != 0 || c.Capacity != 3*1024*1024 || c.Output != c.Capacity {
		t.Fatal(c)
	}
	t.Setenv("TTS_MAX_CONCURRENT_TASKS", "0")
	if _, err = Load(); err == nil {
		t.Fatal("accepted zero concurrency")
	}
	t.Setenv("TTS_MAX_CONCURRENT_TASKS", "4")
	t.Setenv("TTS_BACKEND_URL", ":invalid")
	if _, err = Load(); err == nil {
		t.Fatal("accepted invalid backend address")
	}
}

func TestM5Config(t *testing.T) {
	t.Setenv("TTS_BACKEND_TYPE", "m5")
	t.Setenv("TTS_MAX_CONCURRENT_TASKS", "")
	t.Setenv("TTS_BACKEND_GRPC_ADDR", "127.0.0.1:50051")
	c, err := Load()
	if err != nil || c.Concurrency != 1 {
		t.Fatalf("config=%+v err=%v", c, err)
	}
	t.Setenv("TTS_MAX_CONCURRENT_TASKS", "33")
	if _, err := Load(); err == nil {
		t.Fatal("accepted unsafe M5 concurrency")
	}
	t.Setenv("TTS_MAX_CONCURRENT_TASKS", "1")
	t.Setenv("TTS_BACKEND_GRPC_ADDR", "http://localhost:50051")
	if _, err := Load(); err == nil {
		t.Fatal("accepted invalid gRPC address")
	}
}
