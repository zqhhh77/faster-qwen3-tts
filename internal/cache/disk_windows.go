package cache

import (
	"errors"
	"golang.org/x/sys/windows"
)

func diskAvailable(path string, required int64) error {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return err
	}
	var available uint64
	if err := windows.GetDiskFreeSpaceEx(p, &available, nil, nil); err != nil {
		return err
	}
	if available < uint64(required) {
		return errors.New("磁盘空间不足")
	}
	return nil
}

// Windows does not support fsync on directory handles; production images run Linux.
func syncDirectory(path string) error { return nil }
