"""船舶告警模板匹配和 24 kHz 单声道 PCM 片段衔接。"""
import re
from collections import OrderedDict
from threading import Lock

import numpy as np


ALERT_ENDINGS = ("请立即关注", "请保持关注")
ALERT_PHRASES = ("发现", "触发", *ALERT_ENDINGS)


class AlertFieldCache:
    """按 PCM 字节数淘汰最久未访问的字段。所有推理实例共享已完成的音频。"""

    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = Lock()

    def get(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def put(self, key, pcm, frames):
        if not pcm or len(pcm) > self.max_bytes:
            return
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return
            while self._bytes + len(pcm) > self.max_bytes:
                _, (old, _) = self._entries.popitem(last=False)
                self._bytes -= len(old)
            self._entries[key] = (pcm, frames)
            self._bytes += len(pcm)


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
    subject = f"{ship}，{area}" if area else ship
    return subject + "，", kind + "，", parts[-1]


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
