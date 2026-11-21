"""Run the ONNX detector over a folder of images (read-only) and write results elsewhere.

Usage: python scripts/predict_folder.py IMAGES_DIR OUT_DIR [--conf 0.25] [--vis 24]
Writes: predictions.csv (one row per box), per_image.csv, per_tree.csv, summary.json,
        prelabels/*.txt (YOLO format, usable as pre-annotation), vis/*.jpg (sample drawings).
"""
import argparse, csv, json, re, statistics, sys, time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.detector import Detector  # noqa: E402

COLORS = {"Bunch": "#e8590c", "Ripe": "#d9480f", "Underripe": "#e8a33d", "Unripe": "#2f9e44", "Flower": "#ae3ec9", "Abnormal": "#868e96"}

ap = argparse.ArgumentParser()
ap.add_argument("images"); ap.add_argument("out")
ap.add_argument("--conf", type=float, default=0.25); ap.add_argument("--vis", type=int, default=24)
ap.add_argument("--agnostic", action="store_true", help="detection only, one class 'Bunch'")
a = ap.parse_args()
src, out = Path(a.images), Path(a.out)
(out / "prelabels").mkdir(parents=True, exist_ok=True); (out / "vis").mkdir(exist_ok=True)

det = Detector(Path(__file__).resolve().parent.parent / "models" / "best.onnx")
files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
vis_every = max(1, len(files) // a.vis) if a.vis else 0
boxes_rows, img_rows, lat, load_t = [], [], [], []
tree_counts = defaultdict(Counter)
t_all = time.perf_counter()
for i, f in enumerate(files):
    t0 = time.perf_counter()
    img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    load_t.append(time.perf_counter() - t0)
    dets, ms = det.predict(img, conf=a.conf, agnostic=a.agnostic)
    lat.append(ms)
    W, H = img.size
    c = Counter(d.label for d in dets)
    tree = re.sub(r"_\d+$", "", f.stem)
    tree_counts[tree].update(c)
    img_rows.append({"image": f.name, "tree": tree, "side": f.stem.rsplit("_", 1)[-1], "n": len(dets),
                     **({} if a.agnostic else {k: c.get(k, 0) for k in det.names}), "max_conf": round(max((d.confidence for d in dets), default=0), 3)})
    lines = []
    for d in dets:
        x1, y1, x2, y2 = d.box
        boxes_rows.append({"image": f.name, "label": d.label, "conf": round(d.confidence, 4),
                           "x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2),
                           "area_frac": round((x2 - x1) * (y2 - y1) / (W * H), 5)})
        lines.append(f"{0 if a.agnostic else det.names.index(d.label)} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
    (out / "prelabels" / f"{f.stem}.txt").write_text("\n".join(lines))
    if vis_every and i % vis_every == 0:
        v = img.copy(); v.thumbnail((900, 900)); s = v.width / W; g = ImageDraw.Draw(v)
        for d in dets:
            x1, y1, x2, y2 = (t * s for t in d.box); col = COLORS.get(d.label, "#888")
            g.rectangle([x1, y1, x2, y2], outline=col, width=3)
            g.rectangle([x1, max(0, y1 - 14), x1 + 7 * len(d.label) + 34, y1], fill=col)
            g.text((x1 + 2, max(0, y1 - 13)), f"{d.label} {d.confidence:.2f}", fill="white")
        v.save(out / "vis" / f"{f.stem}.jpg", quality=85)
    if (i + 1) % 100 == 0:
        print(f"{i+1}/{len(files)}", flush=True)
total_s = time.perf_counter() - t_all

def write(name, rows):
    if not rows: return
    with open(out / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

write("predictions.csv", boxes_rows); write("per_image.csv", img_rows)
write("per_tree.csv", [{"tree": t, "sides": sum(r["tree"] == t for r in img_rows), "boxes_all_sides": sum(c.values()),
                        **({} if a.agnostic else {k: c.get(k, 0) for k in det.names})} for t, c in sorted(tree_counts.items())])
cls = Counter(r["label"] for r in boxes_rows)
confs = sorted(r["conf"] for r in boxes_rows)
summary = {
    "mode": "detection-only" if a.agnostic else "detection+class", "images": len(files), "trees": len(tree_counts), "conf_threshold": a.conf, "provider": det.provider,
    "boxes_total": len(boxes_rows), "boxes_by_class": dict(cls.most_common()),
    "images_with_zero_boxes": sum(r["n"] == 0 for r in img_rows),
    "boxes_per_image_mean": round(len(boxes_rows) / max(1, len(files)), 2),
    "conf_median": confs[len(confs) // 2] if confs else None,
    "conf_share_ge_0.5": round(sum(c >= 0.5 for c in confs) / max(1, len(confs)), 3),
    "box_area_frac_median": statistics.median(r["area_frac"] for r in boxes_rows) if boxes_rows else None,
    "model_latency_ms_median": round(statistics.median(lat), 1),
    "image_load_s_median": round(statistics.median(load_t), 3),
    "total_seconds": round(total_s, 1),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
