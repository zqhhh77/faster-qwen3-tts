"""船舶告警模板匹配和 24 kHz 单声道 PCM 片段衔接。"""
import re
import hashlib
import json
import logging
import os
from pathlib import Path
import struct
import tempfile
from collections import OrderedDict
from threading import Lock

import numpy as np


ALERT_ENDINGS = ("请立即关注", "请保持关注")
ALERT_PHRASES = ("发现", "触发", *ALERT_ENDINGS)


class AlertFieldCache:
    """磁盘保存完整 PCM，内存只保存文件大小和最近访问顺序。"""

    _header = struct.Struct("<8sII32s")
    _magic = b"M5PCM01\0"
    _max_pcm_bytes = 2048 * 3840

    def __init__(self, directory, max_bytes=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = Lock()
        if max_bytes is not None:
            files = [(p, p.stat()) for p in self.directory.glob("*.pcm")]
            for path, stat in sorted(files, key=lambda item: item[1].st_mtime_ns):
                self._entries[path] = stat.st_size
                self._bytes += stat.st_size
            self._evict()

    def _path(self, key):
        digest = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()
        return self.directory / (digest + ".pcm")

    def _discard(self, path):
        path.unlink(missing_ok=True)
        self._bytes -= self._entries.pop(path, 0)

    def _evict(self):
        while self.max_bytes is not None and self._bytes > self.max_bytes:
            self._discard(next(iter(self._entries)))

    def get(self, key):
        with self._lock:
            path = self._path(key)
            try:
                with path.open("rb") as stream:
                    magic, frames, size, digest = self._header.unpack(stream.read(self._header.size))
                    if (magic != self._magic or not 1 <= frames <= 2048 or not size
                            or size % 2 or size > self._max_pcm_bytes
                            or os.fstat(stream.fileno()).st_size != size + self._header.size):
                        raise ValueError("音频缓存头无效")
                    pcm = stream.read(size)
                if hashlib.sha256(struct.pack("<I", frames) + pcm).digest() != digest:
                    raise ValueError("音频缓存校验失败")
                os.utime(path, None)
                if path in self._entries:
                    self._entries.move_to_end(path)
                return pcm, frames
            except FileNotFoundError:
                self._bytes -= self._entries.pop(path, 0)
            except (ValueError, struct.error):
                logging.warning("告警音频缓存损坏，将重新生成 path=%s", path)
                try:
                    self._discard(path)
                except OSError:
                    logging.warning("清理损坏的告警缓存失败 path=%s", path, exc_info=True)
            except OSError:
                logging.warning("读取告警音频缓存失败，将实时生成 path=%s", path, exc_info=True)
            return None

    def put(self, key, pcm, frames):
        size = len(pcm) + self._header.size
        if (not pcm or len(pcm) % 2 or len(pcm) > self._max_pcm_bytes
                or not 1 <= frames <= 2048
                or (self.max_bytes is not None and size > self.max_bytes)):
            return
        with self._lock:
            path = self._path(key)
            temporary = None
            try:
                if path.exists():
                    return
                digest = hashlib.sha256(struct.pack("<I", frames) + pcm).digest()
                with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(self._header.pack(self._magic, frames, len(pcm), digest))
                    stream.write(pcm)
                os.replace(temporary, path)
                if self.max_bytes is not None:
                    self._bytes -= self._entries.pop(path, 0)
                    self._entries[path] = size
                    self._bytes += size
                    self._evict()
            except OSError:
                logging.warning("写入告警音频缓存失败，本次音频仍可播放 path=%s", path, exc_info=True)
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        logging.warning("清理告警缓存临时文件失败 path=%s", temporary, exc_info=True)


def alert_parts(text: str):
    body = text.strip().removesuffix("。").rstrip()
    parts = [part.strip() for part in re.split("[，,]", body)]
    if len(parts) not in (3, 4) or parts[-1] not in ALERT_ENDINGS:
        return None
    if not parts[0].startswith("发现") or not parts[-2].startswith("触发"):
        return None
    ship, kind = parts[0][2:].strip(), parts[-2][2:].strip()
    area = parts[1] if len(parts) == 4 else ""
    if not ship or not kind or (area and not (area.startswith("在") and area[1:].strip())):
        return None
    if re.search(r"[。！？!?\r\n]", body):
        return None
    return ship + "，", area + "，" if area else "", kind + "，", parts[-1]


def silence_threshold(pcm):
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
    return np.max(np.abs(samples)) * 0.009


def trim_alert_edges(chunks, threshold):
    # 用 10 ms 均方能量识别边界，保留 60 ms 余量以保护轻辅音。
    frame = 240
    margin = 1440 * 2
    started = False
    pending = bytearray()
    for chunk in chunks:
        samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32)
        if not samples.size:
            continue
        starts = np.arange(0, samples.size, frame)
        power = np.add.reduceat(samples * samples, starts)
        active = np.flatnonzero(power > threshold ** 2 * np.minimum(frame, samples.size - starts))
        if not active.size:
            pending.extend(chunk)
            continue
        first, last = int(active[0]) * frame * 2, (int(active[-1]) + 1) * frame * 2
        pending.extend(chunk[:first])
        lead = bytes(pending if started else pending[-margin:])
        yield lead + chunk[first:last]
        pending = bytearray(chunk[last:])
        started = True
    if pending:
        yield bytes(pending[:margin] if started else pending)
