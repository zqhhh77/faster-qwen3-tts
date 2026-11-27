import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from .runtime import Runtime, create_session, execution_providers
from .batching import DepthBatcher
from .test_runtime import FakeModel


class DeviceTests(unittest.TestCase):
    def test_cuda_workers_share_sessions_but_not_mutable_generation_state(self):
        runtime = Runtime.__new__(Runtime)
        runtime.device = "cuda"
        runtime.settings = {"first_frames": 6}
        runtime.voices = {"default": {"codes": np.zeros((8,16),np.int64)}}
        from infer_onnx import M5Onnx
        runtime.model = M5Onnx.__new__(M5Onnx)
        runtime.model.codec_meta = dict(layers=1, kv_heads=1, head_dim=1, states=[{"shape":[1,2]}])
        runtime.model.sessions = {name: object() for name in ("backbone","depth_frame","codec_decoder")}
        runtime.model.stats = {"status":"complete"}
        runtime.model.generated_codes = np.ones((8,16), np.int64)
        runtime.model._reset_codec()
        worker = runtime.fork_cuda()
        self.assertIsNot(worker.model, runtime.model)
        self.assertIsNot(worker.model.sessions, runtime.model.sessions)
        for name in runtime.model.sessions:
            self.assertIs(worker.model.sessions[name], runtime.model.sessions[name])
        self.assertIs(worker.voices, runtime.voices)
        self.assertFalse(hasattr(worker.model, "generated_codes"))
        worker.model.codec_states[0][0,0] = 9
        worker.model.codec_offset = 10
        worker.model.stats["status"] = "running"
        worker.settings["first_frames"] = 10
        self.assertEqual(runtime.model.codec_states[0][0,0], 0)
        self.assertEqual(runtime.model.codec_offset, 0)
        self.assertEqual(runtime.model.stats["status"], "complete")
        self.assertEqual(runtime.settings["first_frames"], 6)
        with self.assertRaisesRegex(ValueError, "空闲"):
            worker.fork_cuda()
        runtime.device = "cpu"
        with self.assertRaisesRegex(ValueError, "CUDA"):
            runtime.fork_cuda()

    def test_runtime_routes_device_and_isolates_audio_cache(self):
        def fake_model(*args, **kwargs):
            model = FakeModel()
            model.meta = dict(model.meta, depth_frame_precision="fp32")
            return model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text('{"depth_frame_precision":"fp32"}', encoding="utf-8")
            (root / "voice.wav").write_bytes(b"reference")
            (root / "voices.json").write_text(json.dumps({"default": {"wav": "voice.wav", "text": "参考文本"}}), encoding="utf-8")
            with patch.dict(os.environ, {"M5_MODEL": directory, "M5_VOICES_FILE": str(root / "voices.json"),
                                        "M5_DEFAULT_VOICE": "default", "M5_ALERT_PHRASES": "off",
                                        "M5_ALERT_SHIP_PARTS": "off", "M5_DEPTH_BATCH": "off", "M5_CUDA_GRAPH": "on",
                                        "M5_TEXT_NORMALIZATION": "off", "M5_ALERT_NUMBER_TAIL_SPEED": "1"}), \
                    patch("backend.m5.runtime.verify_bundle", return_value={}), \
                    patch("backend.m5.runtime.CudaGraph") as graph, \
                    patch("backend.m5.runtime.M5Onnx", side_effect=fake_model) as model, \
                    patch("backend.m5.runtime.read_reference", return_value=np.zeros(100)):
                identities, voice_identities = [], []
                for device, device_id in (("cpu", 0), ("cuda", 0), ("cuda", 1)):
                    with patch.dict(os.environ, M5_DEVICE=device, M5_DEVICE_ID=str(device_id)):
                        runtime = Runtime()
                        self.assertEqual(model.call_args.kwargs["device"], device)
                        self.assertEqual(model.call_args.kwargs["device_id"], device_id)
                        self.assertEqual(runtime.settings["first_frames"], 6 if device == "cuda" else 2)
                        self.assertEqual("codec_encoder" in runtime.model.sessions, device == "cpu")
                        if device == "cuda":
                            self.assertIs(runtime.model.sessions["depth_frame"], graph.return_value)
                        identities.append(runtime.identity)
                        voice_identities.append(runtime.cache_identities["default"])
                self.assertEqual(len(set(identities)), 3)
                self.assertEqual(len(set(voice_identities)), 3)
                with patch.dict(os.environ, M5_DEVICE="cuda", M5_DEVICE_ID="0", M5_CUDA_GRAPH="off"):
                    disabled = Runtime()
                    self.assertIsNone(disabled.cuda_graph)
                    self.assertNotEqual(disabled.identity, identities[1])
                with patch.dict(os.environ, M5_DEVICE="cpu", M5_DEVICE_ID="0", M5_CUDA_GRAPH="auto"):
                    with self.assertRaisesRegex(ValueError, "M5_CUDA_GRAPH"):
                        Runtime()
                for device, device_id in (("auto", "0"), ("cuda", "-1"), ("cpu", "1")):
                    with patch.dict(os.environ, M5_DEVICE=device, M5_DEVICE_ID=device_id):
                        with self.assertRaisesRegex(ValueError, "M5_DEVICE"):
                            Runtime()
                (root / "manifest.json").write_text('{"depth_frame_precision":"int8"}', encoding="utf-8")
                with patch.dict(os.environ, M5_DEVICE="cuda", M5_DEVICE_ID="0"):
                    with self.assertRaisesRegex(ValueError, "FP32"):
                        Runtime()

    def test_cuda_provider_requires_gpu_package_and_disables_tf32(self):
        with patch("infer_onnx.ort.get_available_providers", return_value=["CPUExecutionProvider"]):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                execution_providers("cuda", 0)
        with patch("infer_onnx.ort.get_available_providers", return_value=["CUDAExecutionProvider"]), \
                patch("infer_onnx.ort.preload_dlls") as preload:
            providers = execution_providers("cuda", 2)
            self.assertEqual(providers[0], ("CUDAExecutionProvider", {"device_id": 2, "use_tf32": 0}))
            preload.assert_called_once()

    def test_requested_cuda_rejects_session_falling_back_to_cpu(self):
        session = Mock()
        session.get_providers.return_value = ["CPUExecutionProvider"]
        with patch("infer_onnx.ort.InferenceSession", return_value=session):
            with self.assertRaisesRegex(RuntimeError, "refusing silent backend fallback"):
                create_session(Path("depth_frame.onnx"), None, [], "CUDAExecutionProvider")
            session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.assertIs(create_session(Path("depth_frame.onnx"), None, [], "CUDAExecutionProvider"), session)
            session.disable_fallback.assert_called_once()

    def test_depth_batch_uses_same_device_and_strict_session_loader(self):
        for device, provider in (("cpu", "CPUExecutionProvider"), ("cuda", "CUDAExecutionProvider")):
            model = SimpleNamespace(root=Path("model"), threads=2, device=device, device_id=0,
                                    provider=provider, sessions={"depth_frame": object()})
            with patch("backend.m5.runtime.execution_providers", return_value=[provider]) as providers, \
                    patch("backend.m5.runtime.create_session") as create:
                batcher = DepthBatcher.load(model)
                try:
                    providers.assert_called_once_with(device, 0)
                    self.assertEqual(create.call_args.args[0], model.root / "depth_frame_pair.onnx")
                    self.assertEqual(create.call_args.args[2:], ([provider], provider))
                    self.assertIs(batcher.single, model.sessions["depth_frame"])
                    self.assertIs(batcher.paired, create.return_value)
                finally:
                    batcher.close()
