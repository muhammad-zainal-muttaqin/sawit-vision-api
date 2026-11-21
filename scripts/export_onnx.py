"""Export the trained YOLOv8 weights to ONNX (needs `pip install ultralytics`)."""
from ultralytics import YOLO

YOLO("models/best.pt").export(format="onnx", imgsz=640, simplify=True, opset=17)
