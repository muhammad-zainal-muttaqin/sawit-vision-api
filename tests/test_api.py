import io
import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image

os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.detector import nms  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)
SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def _jpeg(color=(120, 160, 90), size=(320, 240)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def test_health_lists_four_ripeness_grades():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["classes"] == ["B1", "B2", "B3", "B4"]


def test_detect_returns_schema_and_is_logged():
    r = client.post("/detect", files={"file": ("plain.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["image"] == {"width": 320, "height": 240}
    assert body["count"] == len(body["detections"]) == sum(body["counts"].values())
    assert body["latency_ms"] > 0
    hist = client.get("/history?limit=1").json()
    assert hist[0]["id"] == body["id"] and hist[0]["filename"] == "plain.jpg"
    assert client.get("/stats").json()["requests"] >= 1


def test_rejects_non_image():
    r = client.post("/detect", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_rejects_bad_conf():
    r = client.post("/detect?conf=2", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 422


def test_nms_keeps_best_of_overlapping_boxes():
    import numpy as np
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    assert nms(boxes, scores, 0.5) == [0, 2]


def test_finds_bunches_on_real_field_photo():
    data = (SAMPLES / "palm-2.jpg").read_bytes()
    body = client.post("/detect?conf=0.3", files={"file": ("palm-2.jpg", data, "image/jpeg")}).json()
    assert body["count"] >= 2
    for d in body["detections"]:
        x1, y1, x2, y2 = d["box"]
        assert 0 <= x1 < x2 <= body["image"]["width"] and 0 <= y1 < y2 <= body["image"]["height"]


def test_agnostic_mode_labels_every_box_bunch():
    data = (SAMPLES / "palm-1.jpg").read_bytes()
    body = client.post("/detect?agnostic=true&conf=0.1", files={"file": ("s.jpg", data, "image/jpeg")}).json()
    assert body["count"] >= 1 and set(body["counts"]) == {"Bunch"}
