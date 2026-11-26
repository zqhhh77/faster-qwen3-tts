"""最多两路深度生成合批，音色状态及音频解码由各自 Runtime 持有。"""
from concurrent.futures import Future
from queue import Empty, Queue
from threading import Lock, Thread

import onnxruntime as ort


class DepthBatcher:
    def __init__(self, single, paired):
        self.single = single
        self.paired = paired
        self.queue = Queue()
        self.lock = Lock()
        self.closed = False
        self.active = 0
        self.worker = Thread(target=self._work, name="m5-depth-batch", daemon=True)
        self.worker.start()

    @classmethod
    def load(cls, model):
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(256, model.threads * 2)
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        paired = ort.InferenceSession(str(model.root / "depth_frame_pair.onnx"), options,
                                     providers=["CPUExecutionProvider"])
        paired.disable_fallback()
        return cls(model.sessions["depth_frame"], paired)

    def run(self, outputs, feed):
        future = Future()
        with self.lock:
            if self.closed:
                raise RuntimeError("深度合批服务已停止")
            self.queue.put((feed, future))
        return future.result()

    def enter(self):
        with self.lock:
            self.active += 1

    def leave(self):
        with self.lock:
            self.active -= 1

    def _work(self):
        while True:
            first = self.queue.get()
            if first is None:
                return
            jobs = [first]
            stopping = False
            try:
                with self.lock:
                    wait = 0.003 if self.active >= 2 else 0
                second = self.queue.get(timeout=wait)
                if second is None:
                    stopping = True
                else:
                    jobs.append(second)
            except Empty:
                pass
            try:
                if len(jobs) == 2:
                    inputs = {f"r{i}/{key}": value for i, (feed, _) in enumerate(jobs) for key, value in feed.items()}
                    outputs = self.paired.run(None, inputs)
                    if len(outputs) != 4:
                        raise RuntimeError("深度合批模型返回的输出数量无效")
                    for index, (_, future) in enumerate(jobs):
                        future.set_result(outputs[index * 2:index * 2 + 2])
                else:
                    first[1].set_result(self.single.run(None, first[0]))
            except Exception as error:
                for _, future in jobs:
                    if not future.done():
                        future.set_exception(error)
            if stopping:
                return

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self.queue.put(None)
        self.worker.join()
