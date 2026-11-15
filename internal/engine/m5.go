package engine

import (
	"context"
	"encoding/hex"
	"errors"
	"io"
	"sync"
	"time"
	"tts/internal/config"
	"tts/internal/engine/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type m5Client struct {
	conn        *grpc.ClientConn
	rpc         pb.InferenceServiceClient
	mu          sync.Mutex
	last        *Info
	concurrency int
}

func newM5(c config.Config) (*m5Client, error) {
	conn, err := grpc.NewClient("passthrough:///"+c.GRPCAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(1<<20)))
	if err != nil {
		return nil, err
	}
	return &m5Client{conn: conn, rpc: pb.NewInferenceServiceClient(conn), concurrency: c.Concurrency}, nil
}
func (b *m5Client) Close() error { return b.conn.Close() }
func (b *m5Client) Info(ctx context.Context) (*Info, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	m, err := b.rpc.GetInfo(ctx, &pb.GetInfoRequest{})
	if err != nil {
		return nil, err
	}
	_, digestErr := hex.DecodeString(m.Identity)
	valid := digestErr == nil && len(m.Identity) == 64 && m.Model != "" && m.SampleRate == 24000 &&
		m.Channels == 1 && m.SampleFormat == "s16le" && m.Concurrency > 0 && int(m.Concurrency) == b.concurrency && !m.SupportsInstruct &&
		len(m.Voices) > 0 && len(m.Languages) > 0
	voices := map[string]bool{}
	for _, voice := range m.Voices {
		valid = valid && voice != "" && !voices[voice]
		voices[voice] = true
	}
	if !valid || !voices[m.DefaultVoice] {
		b.mu.Lock()
		b.last = nil
		b.mu.Unlock()
		return nil, ErrIdentity
	}
	info := &Info{Identity: m.Identity, Model: m.Model, ServedModel: "tts-" + m.Identity, TaskType: "Base",
		Voices: m.Voices, DefaultVoice: m.DefaultVoice, Languages: m.Languages, SampleRate: 24000, Concurrency: int(m.Concurrency)}
	b.mu.Lock()
	b.last = info
	b.mu.Unlock()
	return info, nil
}
func (b *m5Client) CachedInfo() *Info {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.last
}
func (b *m5Client) Generate(ctx context.Context, req Request, info *Info, emit func([]byte) error) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	live, err := b.Info(ctx)
	if err != nil || live.Identity != info.Identity {
		return ErrIdentity
	}
	stream, err := b.rpc.Synthesize(ctx, &pb.SynthesizeRequest{Input: req.Input, Voice: req.Voice,
		Language: req.Language, Instruct: req.Instruct, Speed: req.Speed, ExpectedIdentity: info.Identity})
	if err != nil {
		return err
	}
	done := false
	var samples uint64
	for {
		event, err := stream.Recv()
		if err == io.EOF {
			if !done || samples == 0 {
				return errors.New("推理流缺少成功完成标记")
			}
			return nil
		}
		if err != nil {
			return err
		}
		if done {
			return errors.New("推理流在完成后继续返回数据")
		}
		switch item := event.Event.(type) {
		case *pb.SynthesizeEvent_Audio:
			if len(item.Audio) == 0 || len(item.Audio)%2 != 0 || len(item.Audio) > 262144 {
				return errors.New("推理音频块无效")
			}
			if err := emit(item.Audio); err != nil {
				return err
			}
			samples += uint64(len(item.Audio) / 2)
		case *pb.SynthesizeEvent_Done:
			if item.Done == nil || item.Done.Samples != samples || samples == 0 {
				return errors.New("推理音频完成长度不一致")
			}
			done = true
		default:
			return errors.New("推理流事件无效")
		}
	}
}
