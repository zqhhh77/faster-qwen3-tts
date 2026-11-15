"""Track only a subprocess we started, including Windows venv redirector children."""
import threading

import psutil


class ProcessTree:
    def __init__(self, process):
        self.process = process
        self.root = psutil.Process(process.pid)
        self.known = {self.root}
        self.lock = threading.Lock()

    def members(self):
        with self.lock:
            try:
                self.known.update(self.root.children(recursive=True))
            except psutil.NoSuchProcess:
                pass
            return list(self.known)

    def resident_bytes(self):
        total = 0
        for process in self.members():
            try:
                total += process.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return total

    def stop(self, force=False):
        members = self.members()
        for process in members:
            try:
                process.kill() if force else process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(members, timeout=3 if force else 15)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=3)
        self.process.wait(timeout=3)
