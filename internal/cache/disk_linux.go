package cache

import (
	"errors"
	"os"
	"syscall"
)

func diskAvailable(path string, required int64) error {
	var s syscall.Statfs_t
	if err := syscall.Statfs(path, &s); err != nil {
		return err
	}
	if s.Bavail*uint64(s.Bsize) < uint64(required) {
		return errors.New("磁盘空间不足")
	}
	return nil
}

func syncDirectory(path string) error {
	d, err := os.Open(path)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}
