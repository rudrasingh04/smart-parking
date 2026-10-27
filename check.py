"""Run the pipeline on ONE frame and save output.jpg (no server needed).
Usage: python check.py data/parking.mp4 [frame_index]"""
import sys

import cv2

from core import Detector, analyse, load_slots, read_frame

source = sys.argv[1] if len(sys.argv) > 1 else "data/parking.mp4"
index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
frame = read_frame(source, index)
slots = load_slots(frame.shape)
annotated, results = analyse(frame, Detector(), slots)
cv2.imwrite("output.jpg", annotated)
print("Vehicles detected:", results["vehicles"])
print("Slot counts      :", results["counts"])
print("Saved output.jpg")
