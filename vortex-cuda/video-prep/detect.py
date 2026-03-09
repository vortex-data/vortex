#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ultralytics",
#     "opencv-python",
# ]
# ///
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Detect hot dogs in video frames using YOLOv8 and output a JSON boolean array.

Usage:
    uv run detect.py input.mp4 -o detections.json [--width 1920 --height 1080] [--confidence 0.3]

Output JSON is a flat array of booleans, one per frame:
    [false, false, true, true, ..., false]
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
    parser.add_argument("--verbose", action="store_true", help="Print all detected classes per frame")
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
        boxes = results[0].boxes
        classes = boxes.cls.cpu().tolist() if len(boxes) > 0 else []
        confs = boxes.conf.cpu().tolist() if len(boxes) > 0 else []
        int_classes = [int(c) for c in classes]

        if args.verbose and int_classes:
            names = results[0].names
            labels = [f"{names[c]}({confs[i]:.2f})" for i, c in enumerate(int_classes)]
            print(f"  frame {frame_idx}: {', '.join(labels)}")

        has_hot_dog = HOT_DOG_CLASS_ID in int_classes
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
