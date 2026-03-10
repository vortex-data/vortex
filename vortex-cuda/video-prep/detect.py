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

"""Detect objects in video frames using YOLOv8 and output a JSON boolean array.

Usage:
    uv run detect.py input.mp4 -o detections.json --object bicycle
    uv run detect.py input.mp4 -o detections.json --object "hot dog" --confidence 0.15
    uv run detect.py input.mp4 -o detections.json --list-objects  # show all COCO classes

Output JSON is a flat array of booleans, one per frame:
    [false, false, true, true, ..., false]
"""

import argparse
import json
import sys

import cv2
from ultralytics import YOLO


def get_class_id_by_name(model, name):
    """Look up a COCO class ID by name (case-insensitive)."""
    name_lower = name.lower()
    for class_id, class_name in model.names.items():
        if class_name.lower() == name_lower:
            return class_id
    return None


def main():
    parser = argparse.ArgumentParser(description="Detect objects per frame using YOLOv8")
    parser.add_argument("input", nargs="?", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Path to output JSON file")
    parser.add_argument(
        "--object", required=False, help="Object class name to detect (e.g. 'bicycle', 'hot dog', 'person')"
    )
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
    parser.add_argument("--list-objects", action="store_true", help="List all detectable object classes and exit")
    args = parser.parse_args()

    model = YOLO(args.model)

    if args.list_objects:
        print("Detectable COCO classes:")
        for class_id, class_name in sorted(model.names.items()):
            print(f"  {class_id:3d}: {class_name}")
        return

    if not args.input:
        parser.error("input video file is required")
    if not args.output:
        parser.error("-o/--output is required")
    if not args.object:
        parser.error("--object is required (use --list-objects to see available classes)")

    target_class_id = get_class_id_by_name(model, args.object)
    if target_class_id is None:
        print(f"Error: unknown object class '{args.object}'", file=sys.stderr)
        print("Use --list-objects to see available classes", file=sys.stderr)
        sys.exit(1)

    print(f"Detecting '{args.object}' (class {target_class_id}) in {args.input}")

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

        detected = target_class_id in int_classes
        detections.append(detected)

        frame_idx += 1
        if frame_idx % 100 == 0:
            positives = sum(detections)
            print(f"Processed {frame_idx} frames ({positives} with {args.object})")

        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

    cap.release()

    positives = sum(detections)
    print(f"Done: {frame_idx} frames, {positives} with {args.object} ({positives * 100 // max(frame_idx, 1)}%)")

    with open(args.output, "w") as f:
        json.dump(detections, f)


if __name__ == "__main__":
    main()
