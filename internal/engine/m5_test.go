package engine

import (
	"context"
	"errors"
	"net"
	"strings"
	"testing"
	"time"
	"tts/internal/config"
	"tts/internal/engine/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type testInference struct {
	pb.UnimplementedInferenceServiceServer
	mode     string
	identity string
}

func (s *testInference) GetInfo(context.Context, *pb.GetInfoRequest) (*pb.ModelInfo, error) {
	return &pb.ModelInfo{Identity: s.identity, Model: "m5-test", Voices: []string{"default"}, DefaultVoice: "default",
		Languages: []string{"Chinese"}, SampleRate: 24000, Channels: 1, SampleFormat: "s16le", Concurrency: 1}, nil
}

func (s *testInference) Synthesize(req *pb.SynthesizeRequest, stream grpc.ServerStreamingServer[pb.SynthesizeEvent]) error {
	if req.ExpectedIdentity != s.identity {
		return status.Error(codes.FailedPrecondition, "changed")
	}
	if s.mode == "cancel" {
		<-stream.Context().Done()
		return stream.Context().Err()
	}
	audio := []byte{1, 0, 2, 0}
	if s.mode == "odd" {
		audio = []byte{1}
	}
	if s.mode == "oversize" {
		audio = make([]byte, 262146)
	}
	if s.mode != "empty" {
		if err := stream.Send(&pb.SynthesizeEvent{Event: &pb.SynthesizeEvent_Audio{Audio: audio}}); err != nil {
			return err
		}
	}
	if s.mode == "missing-done" {
		return nil
	}
	count := uint64(2)
	if s.mode == "bad-count" {
		count = 3
	}
	if err := stream.Send(&pb.SynthesizeEvent{Event: &pb.SynthesizeEvent_Done{Done: &pb.Completion{Samples: count}}}); err != nil {
		return err
	}
	if s.mode == "trailing-error" {
		return status.Error(codes.Internal, "truncated")
	}
	if s.mode == "after-done" {
		return stream.Send(&pb.SynthesizeEvent{Event: &pb.SynthesizeEvent_Audio{Audio: audio}})
	}
	return nil
}

func TestM5StreamingCompletion(t *testing.T) {
	for _, mode := range []string{"ok", "missing-done", "bad-count", "trailing-error", "after-done", "odd", "oversize", "empty", "cancel", "emit-error", "identity-changed", "capacity-mismatch"} {
		t.Run(mode, func(t *testing.T) {
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			service := &testInference{mode: mode, identity: strings.Repeat("a", 64)}
			server := grpc.NewServer()
			pb.RegisterInferenceServiceServer(server, service)
			go server.Serve(listener)
			t.Cleanup(server.Stop)
			client, err := newGRPC(config.Config{GRPCAddr: listener.Addr().String(), Concurrency: 1})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { client.Close() })
			if mode == "capacity-mismatch" {
				client.concurrency = 2
			}
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			info, err := client.Info(ctx)
			if mode == "capacity-mismatch" {
				if !errors.Is(err, ErrIdentity) || client.CachedInfo() != nil {
					t.Fatalf("capacity mismatch accepted: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if mode == "identity-changed" {
				info = &Info{Identity: strings.Repeat("b", 64)}
			}
			if mode == "cancel" {
				var stop context.CancelFunc
				ctx, stop = context.WithTimeout(ctx, 50*time.Millisecond)
				defer stop()
			}
			bytes := 0
			err = client.Generate(ctx, Request{Input: "你好", Voice: "default", Language: "Chinese", Speed: 1}, info, func(p []byte) error {
				bytes += len(p)
				if mode == "emit-error" {
					return errors.New("disk full")
				}
				return nil
			})
			if mode == "ok" {
				if err != nil || bytes != 4 {
					t.Fatalf("bytes=%d err=%v", bytes, err)
				}
			} else if err == nil {
				t.Fatal("invalid stream accepted")
			}
		})
	}
}
