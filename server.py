"""FastAPI backend.  Run:  uvicorn server:app --port 8000
Environment: PARKING_SOURCE=data/parking.mp4 (video or image)"""
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response

from core import Detector, analyse, is_image, load_slots, resize

SOURCE = os.environ.get("PARKING_SOURCE", "data/parking.mp4")
BASE = Path(__file__).parent
state = {"jpeg": None, "results": {"status": "starting"}}


def worker():
    try:
        detector = Detector()
        image = None
        cap = None
        if is_image(SOURCE):
            image = resize(cv2.imread(SOURCE))
        else:
            cap = cv2.VideoCapture(SOURCE)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
        slots = None
        start = time.time()
        pos = 0
        while True:
            if image is not None:
                frame = image
            else:
                target = int((time.time() - start) * fps)   # real-time pace
                ok = True
                while pos < target and ok:
                    ok = cap.grab()
                    pos += 1
                ok, frame = cap.read() if ok else (False, None)
                pos += 1
                if not ok:                                   # loop the video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    start, pos = time.time(), 0
                    continue
                frame = resize(frame)
            if slots is None:
                slots = load_slots(frame.shape)
            t0 = time.time()
            annotated, results = analyse(frame, detector, slots)
            results["status"] = "ok"
            results["source"] = os.path.basename(SOURCE)
            results["inference_ms"] = int((time.time() - t0) * 1000)
            results["updated"] = time.time()
            ok, buf = cv2.imencode(".jpg", annotated,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            state["jpeg"], state["results"] = buf.tobytes(), results
            if image is not None:
                time.sleep(0.5)
    except Exception as e:                                   # show errors in UI
        state["results"] = {"status": "error", "message": repr(e)}
        raise


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=worker, daemon=True).start()
    yield


app = FastAPI(title="Smart Parking", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/results")
def results():
    return state["results"]


@app.get("/frame")
def frame():
    if state["jpeg"] is None:
        return Response(status_code=204)
    return Response(state["jpeg"], media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})
