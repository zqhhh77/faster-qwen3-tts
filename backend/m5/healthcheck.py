import os

import grpc
from .generated import inference_pb2 as pb
from .generated import inference_pb2_grpc as rpc

with grpc.insecure_channel(os.environ.get("M5_HEALTH_ADDR", "127.0.0.1:50051")) as channel:
    info = rpc.InferenceServiceStub(channel).GetInfo(pb.GetInfoRequest(), timeout=5)
    if not info.identity or not info.voices:
        raise SystemExit(1)
