# sawit-vision-api

Finds oil palm fresh fruit bunches in a photo and grades each one from `B1` (ripe) to `B4` (unripe), the grading used in the
[SawitMVC dataset](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO). Served with FastAPI and ONNX Runtime; every request is logged to SQLite.

The same model runs in the browser on my site: https://muhammad-zainal-muttaqin.github.io/#demo

![detections](docs/detections.jpg)

## Model

`models/ffb-detector-yolo26n.onnx` is the YOLO26n detector our lab ships in the
[PalmAnnotate](https://github.com/ULM-SawitMVC/PalmAnnotate-Kotlin) field app, trained on SawitMVC (953 trees, 3,992 photos), 640 px input, 9.8 MB.

Inference does not need PyTorch. Letterbox pre-processing and NMS are written in NumPy ([`app/detector.py`](app/detector.py)), the same steps as the
Android app, so the runtime only needs `onnxruntime`, `numpy` and `Pillow`.

An older YOLOv8n model (5 classes, Dec 2025) is kept in `models/legacy-yolov8n/` for comparison.

## Speed

One 810×1080 photo, Apple M1 (8 GB), median of 50 runs including pre/post-processing (`python scripts/benchmark.py IMAGE`):

| Runtime | Median | p95 |
|---|---|---|
| ONNX Runtime + CoreML | 37 ms | 48 ms |
| ONNX Runtime, CPU only | 100 ms | 128 ms |

In the browser (ONNX Runtime Web with WebGPU) the same model takes about 40–70 ms per photo after warm-up.

## Field test: new model vs old model

568 new phone photos (71 trees × 8 sides, Damimas block, Nov 2026), not used in training. They have no labels yet, so this compares
the two models rather than measuring mAP (`scripts/predict_folder.py`, confidence 0.25, detection only):

| | YOLO26n (this repo) | YOLOv8n (legacy) |
|---|---|---|
| Boxes | 717 | 691 |
| Boxes centred in the bottom 40% of the photo (ground, trunk) | 2 | 216 |
| Boxes touching the photo edge | 0 | 167 |

Looking at the photos: the new model puts almost nothing on ground or background trees, grades ripe red bunches as `B1` where the old
model called them `Underripe`, and still mixes up neighbouring grades (a red bunch graded `B2`, for example). Small bunches near the top of
the crown in photos taken from far away are still missed.

## API

| Method | Path | |
|---|---|---|
| `POST` | `/detect?conf=0.25&iou=0.45&agnostic=false` | Multipart `file`. Returns boxes, grade, confidence, count per grade and latency. `agnostic=true` gives detection only (every box is `Bunch`) |
| `GET` | `/history?limit=20` | Latest requests from the SQLite log |
| `GET` | `/stats` | Request count, objects per grade, median latency |
| `GET` | `/health` | Model file, execution provider, class names |
| `GET` | `/` | Upload page that draws the boxes |
| `GET` | `/docs` | Swagger UI |

```bash
curl -F "file=@samples/palm-5.jpg" "http://localhost:8000/detect?conf=0.3"
# response (only the first of the 6 detections shown)
```

```json
{
  "id": 2,
  "image": {"width": 810, "height": 1080},
  "count": 6,
  "counts": {"B1": 1, "B2": 2, "B3": 2, "B4": 1},
  "latency_ms": 12.9,
  "detections": [{"label": "B1", "confidence": 0.6777, "box": [315.1, 422.3, 429.0, 595.3]}]
}
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Docker (`DB_PATH` points to the `/data` volume):

```bash
docker build -t sawit-vision-api .
docker run -p 8000:8000 -v sawit-data:/data sawit-vision-api
```

Tests (7, including two on real field photos in `samples/`):

```bash
pip install -r requirements.txt pytest httpx
pytest -q
```

Use another model with `MODEL_PATH=path/to/model.onnx`; export Ultralytics weights with `python scripts/export_onnx.py WEIGHTS.pt`.

## Notes

- The models were trained with Ultralytics YOLO, which is AGPL-3.0.
- Sample photos in `samples/` are field captures from the Damimas block, shared with permission.
