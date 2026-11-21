"""Export Ultralytics weights to ONNX (needs `pip install ultralytics`). Usage: python scripts/export_onnx.py WEIGHTS.pt"""
import sys

from ultralytics import YOLO

YOLO(sys.argv[1]).export(format="onnx", imgsz=640, simplify=True, opset=17)
