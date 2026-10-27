"""Core logic: frame loading, YOLO detection, slot status, drawing."""
import json
import os

import cv2
import numpy as np

PROC_WIDTH = 1280                 # every frame is resized to this width
TWO_WHEELER = {3}                 # COCO: motorcycle
FOUR_WHEELER = {2, 5, 7}          # COCO: car, bus, truck
SLOTS_FILE = "data/slots.json"
OVERLAP_THRESHOLD = 0.30          # slot counts as occupied above this overlap
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")

GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 220), (0, 255, 255)


def is_image(path):
    return path.lower().endswith(IMAGE_EXT)


def resize(frame):
    h, w = frame.shape[:2]
    if w == PROC_WIDTH:
        return frame
    return cv2.resize(frame, (PROC_WIDTH, int(round(h * PROC_WIDTH / w))))


def read_frame(source, index=0):
    """Return one resized frame from an image or a video file."""
    if is_image(source):
        img = cv2.imread(source)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {source}")
        return resize(img)
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {index} from: {source}")
    return resize(frame)


# ---------------------------------------------------------------- detection
class Detector:
    def __init__(self, weights="yolo11s.pt", conf=0.25, imgsz=1280):
        from ultralytics import YOLO
        import torch
        self.model = YOLO(weights)
        self.conf, self.imgsz = conf, imgsz
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def _run(self, frame):
        return self.model(frame, conf=self.conf, imgsz=self.imgsz,
                          device=self.device, verbose=False)[0]

    def detect(self, frame):
        try:
            result = self._run(frame)
        except Exception:
            if self.device == "cpu":
                raise
            self.device = "cpu"          # Apple GPU failed: fall back to CPU
            result = self._run(frame)
        vehicles = []
        for b in result.boxes:
            cls = int(b.cls[0])
            if cls in TWO_WHEELER:
                vtype = "2W"
            elif cls in FOUR_WHEELER:
                vtype = "4W"
            else:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            vehicles.append({"type": vtype, "box": [x1, y1, x2, y2],
                             "conf": round(float(b.conf[0]), 2)})
        return vehicles


# -------------------------------------------------------------------- slots
def load_slots(shape, path=SLOTS_FILE):
    """Load marked slots; build a mask per slot for fast overlap maths."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    h, w = shape[:2]
    sx, sy = w / data["width"], h / data["height"]
    slots = []
    for s in data["slots"]:
        pts = np.array([[p[0] * sx, p[1] * sy] for p in s["points"]], np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        slots.append({"id": s["id"], "type": s["type"], "points": pts,
                      "mask": mask, "area": max(int(mask.sum()), 1),
                      "occupied": False})
    return slots


def update_status(slots, vehicles, threshold=OVERLAP_THRESHOLD):
    """overlap = area(slot ∩ vehicle box) / area(slot).
    Each vehicle can occupy only its best-matching slot of the same type."""
    h, w = slots[0]["mask"].shape if slots else (0, 0)
    for s in slots:
        s["occupied"] = False
    for v in vehicles:
        x1, y1, x2, y2 = v["box"]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        best, best_ov = None, 0.0
        for s in slots:
            if s["type"] != v["type"]:
                continue
            ov = int(s["mask"][y1:y2, x1:x2].sum()) / s["area"]
            if ov > best_ov:
                best, best_ov = s, ov
        if best is not None and best_ov >= threshold:
            best["occupied"] = True
            v["slot"] = best["id"]
    return slots


def summarize(slots):
    out = {}
    for t in ("2W", "4W"):
        group = [s for s in slots if s["type"] == t]
        free = sum(1 for s in group if not s["occupied"])
        out[t] = {"free": free, "occupied": len(group) - free,
                  "total": len(group)}
    return out


# ------------------------------------------------------------------ drawing
def draw(frame, slots, vehicles):
    out = frame.copy()
    overlay = out.copy()
    for s in slots:
        cv2.fillPoly(overlay, [s["points"]], RED if s["occupied"] else GREEN)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    for s in slots:
        color = RED if s["occupied"] else GREEN
        cv2.polylines(out, [s["points"]], True, color, 2)
        cx, cy = s["points"].mean(axis=0).astype(int)
        cv2.putText(out, f'{s["type"]}-{s["id"]}', (cx - 22, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    for v in vehicles:
        x1, y1, x2, y2 = v["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), YELLOW, 2)
        cv2.putText(out, v["type"], (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
    return out


def analyse(frame, detector, slots):
    """Full pipeline for one frame -> (annotated frame, results dict)."""
    vehicles = detector.detect(frame)
    update_status(slots, vehicles)
    annotated = draw(frame, slots, vehicles)
    results = {
        "counts": summarize(slots),
        "vehicles": {"2W": sum(v["type"] == "2W" for v in vehicles),
                     "4W": sum(v["type"] == "4W" for v in vehicles)},
        "slots": [{"id": s["id"], "type": s["type"],
                   "occupied": s["occupied"]} for s in slots],
        "slots_marked": len(slots) > 0,
    }
    return annotated, results
