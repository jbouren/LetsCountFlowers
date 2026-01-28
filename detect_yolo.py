#!/usr/bin/env python3
"""
Run YOLO inference on selected frames and output annotated results
"""

from ultralytics import YOLO
import cv2
import os
import glob
from datetime import datetime

def detect_zinnias(model_path="runs/zinnia/train/weights/best.pt", confidence=0.25):
    """Run detection on all selected frames"""

    # Load trained model
    model = YOLO(model_path)

    # Create timestamped output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"yolo_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Get frames
    frames_dir = "selected_frames"
    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))

    print(f"Running YOLO detection on {len(frame_files)} frames...")
    print(f"Model: {model_path}")
    print(f"Confidence threshold: {confidence}")
    print(f"Output: {output_dir}/\n")

    total_flowers = 0
    frame_counts = []

    for frame_path in frame_files:
        frame_name = os.path.basename(frame_path)

        # Run inference
        results = model(frame_path, conf=confidence, verbose=False)

        # Get detection count
        count = len(results[0].boxes)
        total_flowers += count
        frame_counts.append((frame_name, count))

        # Save annotated image
        annotated = results[0].plot()

        # Add count overlay
        cv2.putText(annotated, f"Detected: {count}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        cv2.putText(annotated, f"Running total: {total_flowers}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)

        output_path = os.path.join(output_dir, f"yolo_{frame_name}")
        cv2.imwrite(output_path, annotated)

        print(f"{frame_name}: {count} flowers")

    # Save summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"YOLO Zinnia Detection Summary\n")
        f.write(f"==============================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Confidence: {confidence}\n")
        f.write(f"Frames processed: {len(frame_files)}\n")
        f.write(f"Total flowers detected: {total_flowers}\n")
        f.write(f"Average per frame: {total_flowers/len(frame_files):.1f}\n\n")
        f.write(f"Per-frame counts:\n")
        for name, count in frame_counts:
            f.write(f"  {name}: {count}\n")

    print(f"\n{'='*50}")
    print(f"TOTAL FLOWERS DETECTED: {total_flowers}")
    print(f"{'='*50}")
    print(f"\nOutput saved to: {output_dir}/")

    # Open output folder
    os.system(f"open {output_dir}")

    return total_flowers

if __name__ == "__main__":
    detect_zinnias()
