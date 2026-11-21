"""FastAPI service: oil palm fresh fruit bunch (FFB) ripeness detection."""
from __future__ import annotations

import io
import os
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from .db import DetectionLog
from .detector import Detector

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "ffb-detector-yolo26n.onnx"))
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "detections.db"))
MAX_BYTES = 10 * 1024 * 1024

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
detector = Detector(MODEL_PATH)
log = DetectionLog(DB_PATH)

app = FastAPI(
    title="Sawit Vision API",
    description="Detects oil palm fresh fruit bunches and grades ripeness B1 (ripe) to B4 (unripe) with a YOLO26n ONNX model.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_PATH.name, "provider": detector.provider, "classes": detector.names}


@app.post("/detect")
async def detect(
    file: UploadFile = File(..., description="JPEG/PNG image"),
    conf: float = Query(0.25, ge=0.01, le=0.99),
    iou: float = Query(0.45, ge=0.1, le=0.9),
    agnostic: bool = Query(False, description="Detection only: label every box 'Bunch', no ripeness class"),
) -> dict:
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 10 MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "File is not a readable image")
    dets, latency = detector.predict(img, conf=conf, iou=iou, agnostic=agnostic)
    counts = dict(Counter(d.label for d in dets))
    log_id = log.add(file.filename, img.width, img.height, counts, latency, conf)
    return {
        "id": log_id,
        "image": {"width": img.width, "height": img.height},
        "count": len(dets),
        "counts": counts,
        "latency_ms": round(latency, 1),
        "detections": [d.as_dict() for d in dets],
    }


@app.get("/history")
def history(limit: int = Query(20, ge=1, le=200)) -> list[dict]:
    return log.recent(limit)


@app.get("/stats")
def stats() -> dict:
    return log.stats()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
