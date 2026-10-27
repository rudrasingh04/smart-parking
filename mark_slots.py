"""Mark parking slots once on a reference frame.

Usage:  python mark_slots.py data/parking.mp4 [frame_index]
  - Click the 4 corners of a slot (any order).
  - Press 4 -> save it as a 4-wheeler slot, 2 -> save it as a 2-wheeler slot.
  - u = undo, q = quit. Every slot is saved to data/slots.json immediately.
"""
import json
import sys

import cv2
import numpy as np

from core import SLOTS_FILE, read_frame

source = sys.argv[1] if len(sys.argv) > 1 else "data/parking.mp4"
index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
frame = read_frame(source, index)
h, w = frame.shape[:2]
slots, pts = [], []
COLORS = {"4W": (0, 200, 0), "2W": (255, 128, 0)}


def order(points):
    p = np.array(points, float)
    c = p.mean(axis=0)
    return p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))] \
        .astype(int).tolist()


def save():
    with open(SLOTS_FILE, "w") as f:
        json.dump({"source": source, "frame_index": index,
                   "width": w, "height": h, "slots": slots}, f, indent=1)


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append([x, y])


def render(msg=""):
    img = frame.copy()
    for s in slots:
        p = np.array(s["points"], np.int32)
        cv2.polylines(img, [p], True, COLORS[s["type"]], 2)
        cx, cy = p.mean(axis=0).astype(int)
        cv2.putText(img, f'{s["type"]}-{s["id"]}', (cx - 22, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    for p in pts:
        cv2.circle(img, tuple(p), 5, (0, 255, 255), -1)
    cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
    text = msg or (f"{len(slots)} slots | click 4 corners, then press 4 (car) "
                   f"or 2 (bike) | u=undo q=quit")
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    return img


cv2.namedWindow("Mark slots")
cv2.setMouseCallback("Mark slots", on_mouse)
msg = ""
while True:
    cv2.imshow("Mark slots", render(msg))
    key = cv2.waitKey(30) & 0xFF
    msg = ""
    if key in (ord("q"), 27):
        break
    elif key in (ord("4"), ord("2")):
        if len(pts) == 4:
            slots.append({"id": len(slots) + 1,
                          "type": "4W" if key == ord("4") else "2W",
                          "points": order(pts)})
            pts = []
            save()
        else:
            msg = "Click all 4 corners first"
    elif key == ord("u"):
        if pts:
            pts.pop()
        elif slots:
            slots.pop()
            save()
cv2.destroyAllWindows()
print(f"Saved {len(slots)} slots to {SLOTS_FILE}")
