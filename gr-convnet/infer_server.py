"""FastAPI grasp inference 서버 (운영용).

모델을 startup 시 1회 로드 후 매 요청에 재사용 → cold start 회피.
infer.py 와 동일한 후처리/출력 포맷.

기동:
  uvicorn infer_server:app --host 0.0.0.0 --port 8080
  # 또는
  python infer_server.py --port 8080

설정 (환경변수):
  JIPCHAK_CHECKPOINT=trained-models/.../epoch_30_iou_0.97
  JIPCHAK_DEVICE=cuda:0    # 또는 cpu

요청 (multipart/form-data):
  POST /infer
    rgb=@color.png  (필수)
    depth=@depth.png (필수)
    factor_depth=1000  (선택, default 1000)
    top_k=5  (선택, default 5)

응답:
  {
    "model": "...",
    "device": "...",
    "inference_ms": 3.1,
    "n_candidates": 5,
    "candidates": [{"x": ..., "y": ..., "z": ..., "score": ...}, ...]
  }

헬스체크:
  GET /health → {"status": "ok", "device": "...", "checkpoint": "..."}
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from infer import DEFAULT_CHECKPOINT, GraspInfer, candidates_to_dict  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="jipchak-ai grasp inference")

_state: dict = {"infer": None, "device": None, "checkpoint": None}


@app.on_event("startup")
def load_model() -> None:
    checkpoint = os.environ.get("JIPCHAK_CHECKPOINT", DEFAULT_CHECKPOINT)
    device = os.environ.get(
        "JIPCHAK_DEVICE",
        "cuda:0" if torch.cuda.is_available() else "cpu",
    )
    logger.info(f"Loading model: {checkpoint} on {device}")
    t0 = time.perf_counter()
    _state["infer"] = GraspInfer(checkpoint=checkpoint, device=device)
    _state["device"] = device
    _state["checkpoint"] = checkpoint
    logger.info(f"Model loaded in {(time.perf_counter() - t0) * 1000:.1f} ms")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _state["infer"] is not None else "not_ready",
        "device": _state["device"],
        "checkpoint": (
            os.path.basename(_state["checkpoint"]) if _state["checkpoint"] else None
        ),
    }


@app.post("/infer")
async def infer_endpoint(
    rgb: UploadFile = File(...),
    depth: UploadFile = File(...),
    factor_depth: float = Form(1000.0),
    top_k: int = Form(5),
) -> JSONResponse:
    if _state["infer"] is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    rgb_bytes = await rgb.read()
    depth_bytes = await depth.read()

    try:
        rgb_arr = np.array(Image.open(io.BytesIO(rgb_bytes)))
        depth_raw = np.array(Image.open(io.BytesIO(depth_bytes))).astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"image decode failed: {e}")

    depth_m = depth_raw / factor_depth

    t0 = time.perf_counter()
    candidates = _state["infer"].predict(
        rgb_arr, depth_m, depth_raw=depth_raw, top_k=top_k
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    return JSONResponse({
        "model": os.path.basename(os.path.dirname(_state["checkpoint"])),
        "device": _state["device"],
        "inference_ms": round(inference_ms, 2),
        "n_candidates": len(candidates),
        "candidates": candidates_to_dict(candidates),
    })


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
