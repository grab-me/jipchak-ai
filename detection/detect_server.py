"""FastAPI Detection 서버 (운영용).

모델 startup 시 1회 로드 → cold start 회피.
Grasp 의 `gr-convnet/infer_server.py` 와 동일 패턴.

기동:
    uvicorn detect_server:app --host 0.0.0.0 --port 8081
    # 또는
    python detect_server.py --port 8081

설정 (환경변수):
    JIPCHAK_DETECT_DEVICE=cuda:0   # 또는 cpu

요청:
    POST /detect (multipart/form-data)
        rgb=@color.png
        conf=0.05  (선택)
        top_k=20   (선택)
        class_filter=88,47  (선택, 콤마 구분 COCO id)

응답: detect.py 와 동일 JSON

헬스: GET /health
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from detect import DetectInfer, detections_to_dict  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="jipchak-ai detection inference")

_state: dict = {"detector": None, "device": None}


@app.on_event("startup")
def load_model() -> None:
    device = os.environ.get(
        "JIPCHAK_DETECT_DEVICE",
        "cuda:0" if torch.cuda.is_available() else "cpu",
    )
    logger.info(f"Loading SSDlite on {device}")
    t0 = time.perf_counter()
    _state["detector"] = DetectInfer(device=device)
    _state["device"] = device
    logger.info(f"Model loaded in {(time.perf_counter() - t0) * 1000:.1f} ms")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _state["detector"] is not None else "not_ready",
        "model": "ssdlite320_mobilenet_v3_large",
        "device": _state["device"],
    }


@app.post("/detect")
async def detect_endpoint(
    rgb: UploadFile = File(...),
    conf: float = Form(0.05),
    top_k: int = Form(20),
    class_filter: Optional[str] = Form(None),
) -> JSONResponse:
    if _state["detector"] is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    try:
        rgb_bytes = await rgb.read()
        rgb_arr = np.array(Image.open(io.BytesIO(rgb_bytes)).convert("RGB"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"image decode failed: {e}")

    cls_set = None
    if class_filter:
        try:
            cls_set = set(int(x.strip()) for x in class_filter.split(",") if x.strip())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"class_filter parse error: {e}")

    t0 = time.perf_counter()
    detections = _state["detector"].predict(
        rgb_arr, conf=conf, top_k=top_k, class_filter=cls_set
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    return JSONResponse({
        "model": "ssdlite320_mobilenet_v3_large",
        "device": _state["device"],
        "inference_ms": round(inference_ms, 2),
        "n_detections": len(detections),
        "detections": detections_to_dict(detections),
    })


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
