#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Detect hot dogs in video frames using YOLOv8 and output a JSON boolean array.

Usage:
    python detect.py input.mp4 -o detections.json [--width 1920 --height 1080] [--confidence 0.3]

Output JSON is a flat array of booleans, one per frame:
    [false, false, true, true, ..., false]

Requires: pip install ultralytics opencv-python
"""

import argparse
import json
import sys

import cv2
from ultralytics import YOLO

# COCO class ID for "hot dog"
HOT_DOG_CLASS_ID = 52


def main():
    parser = argparse.ArgumentParser(description="Detect hot dogs per frame")
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument("-o", "--output", required=True, help="Path to output JSON file")
    parser.add_argument("--width", type=int, default=None, help="Resize width (match video-prep)")
    parser.add_argument("--height", type=int, default=None, help="Resize height (match video-prep)")
    parser.add_argument(
        "--confidence", type=float, default=0.3, help="Detection confidence threshold"
    )
    parser.add_argument(
        "--model", default="yolov8n.pt", help="YOLO model to use (default: yolov8n.pt)"
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all)")
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: cannot open {args.input}", file=sys.stderr)
        sys.exit(1)

    detections = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.width and args.height:
            frame = cv2.resize(frame, (args.width, args.height))

        results = model(frame, verbose=False, conf=args.confidence)
        classes = results[0].boxes.cls.cpu().tolist() if len(results[0].boxes) > 0 else []
        has_hot_dog = HOT_DOG_CLASS_ID in [int(c) for c in classes]
        detections.append(has_hot_dog)

        frame_idx += 1
        if frame_idx % 100 == 0:
            positives = sum(detections)
            print(f"Processed {frame_idx} frames ({positives} with hot dog)")

        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

    cap.release()

    positives = sum(detections)
    print(f"Done: {frame_idx} frames, {positives} with hot dog ({positives * 100 // max(frame_idx, 1)}%)")

    with open(args.output, "w") as f:
        json.dump(detections, f)


if __name__ == "__main__":
    main()
