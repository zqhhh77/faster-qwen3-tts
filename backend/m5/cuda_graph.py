"""固定形状的深度生成使用 CUDA Graph，所有调用由同一线程执行。"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort


class CudaGraph:
    def __init__(self, model):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="m5-cuda-graph")
        try:
            self.executor.submit(self._initialize, model).result()
        except BaseException:
            self.executor.shutdown()
            raise

    def _initialize(self, model):
        from .runtime import create_session, execution_providers

        options = ort.SessionOptions()
        options.intra_op_num_threads = model.threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        providers = execution_providers("cuda", model.device_id)
        providers[0][1]["enable_cuda_graph"] = "1"
        self.session = create_session(model.root / "depth_frame.onnx", options, providers, "CUDAExecutionProvider")
        self.io = self.session.io_binding()
        types = {"tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(bool)": np.bool_}
        self.inputs, self.outputs = {}, []
        for meta in (*self.session.get_inputs(), *self.session.get_outputs()):
            if meta.type not in types or any(not isinstance(d, int) or d <= 0 for d in meta.shape):
                raise ValueError("CUDA Graph 需要固定形状的 FP32 深度生成模型")
        for meta in self.session.get_inputs():
            value = ort.OrtValue.ortvalue_from_numpy(np.ones(meta.shape, types[meta.type]), "cuda", model.device_id)
            self.inputs[meta.name] = value
            self.io.bind_ortvalue_input(meta.name, value)
        for meta in self.session.get_outputs():
            value = ort.OrtValue.ortvalue_from_shape_and_type(meta.shape, types[meta.type], "cuda", model.device_id)
            self.io.bind_ortvalue_output(meta.name, value)
            self.outputs.append(value)
        # 启动时完成捕获，后续调用更新同一组缓冲区并重放。
        self.session.run_with_iobinding(self.io)

    def _run(self, feed):
        if feed.keys() != self.inputs.keys():
            raise ValueError("CUDA Graph 输入字段不匹配")
        for name, value in feed.items():
            self.inputs[name].update_inplace(value)
        self.session.run_with_iobinding(self.io)
        return [value.numpy() for value in self.outputs]

    def run(self, outputs, feed):
        return self.executor.submit(self._run, feed).result()

    def close(self):
        self.executor.shutdown()
