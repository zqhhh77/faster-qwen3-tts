package engine

import (
	"context"
	"tts/internal/config"
)

// Backend emits only 24 kHz mono s16le PCM and returns nil only after verified completion.
type Backend interface {
	Info(context.Context) (*Info, error)
	CachedInfo() *Info
	Generate(context.Context, Request, *Info, func([]byte) error) error
	Close() error
}

func Open(c config.Config) (Backend, error) {
	if c.BackendType == "m5" {
		return newM5(c)
	}
	return New(c), nil
}

func (b *Client) Close() error {
	b.client.CloseIdleConnections()
	return nil
}
