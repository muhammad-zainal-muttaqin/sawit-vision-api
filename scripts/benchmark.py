"""Compare PyTorch (.pt) vs ONNX Runtime latency on CPU. Usage: python scripts/benchmark.py IMAGE [RUNS]"""
import statistics
import sys
import time

from PIL import Image

img_path = sys.argv[1]
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
img = Image.open(img_path).convert("RGB")


def bench(fn):
    for _ in range(5):
        fn()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1000)
    return statistics.median(t), sorted(t)[int(len(t) * 0.95) - 1]


sys.path.insert(0, ".")
from app.detector import Detector  # noqa: E402

import onnxruntime as ort  # noqa: E402

for prov in ("CoreMLExecutionProvider", "CPUExecutionProvider"):
    if prov not in ort.get_available_providers():
        continue
    det = Detector("models/best.onnx", providers=[prov, "CPUExecutionProvider"])
    med, p95 = bench(lambda: det.predict(img))
    print(f"ONNX Runtime {prov.replace('ExecutionProvider',''):7s} (incl. pre/post): median {med:.1f} ms, p95 {p95:.1f} ms, {1000/med:.1f} FPS")

try:
    from ultralytics import YOLO
    m = YOLO("models/best.pt")
    pt_med, pt_p95 = bench(lambda: m.predict(img, verbose=False, device="cpu"))
    print(f"PyTorch (ultralytics predict):         median {pt_med:.1f} ms, p95 {pt_p95:.1f} ms, {1000/pt_med:.1f} FPS")
except ImportError:
    print("ultralytics not installed; skipped PyTorch baseline")
