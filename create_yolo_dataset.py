#!/usr/bin/env python3
"""
Create YOLO dataset from selected frames using template matching for initial annotations.
You can then review and adjust the annotations before training.
"""

import cv2
import os
import glob
import shutil
from ZinniaDetector import ImprovedZinniaDetector

def create_yolo_annotations(threshold=0.6):
    """Generate YOLO format annotations from template matching detections"""

    # Initialize detector
    detector = ImprovedZinniaDetector()
    detector.load_templates(template_dir="templates", template_pattern="zinnia_template_*.jpg")

    # Paths
    frames_dir = "selected_frames"
    output_dir = "yolo_dataset"

    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))

    # Split into train (80%) and val (20%)
    split_idx = int(len(frame_files) * 0.8)
    train_frames = frame_files[:split_idx]
    val_frames = frame_files[split_idx:]

    print(f"Creating YOLO dataset with threshold={threshold}")
    print(f"Train frames: {len(train_frames)}, Val frames: {len(val_frames)}\n")

    total_annotations = 0

    for split_name, frames in [("train", train_frames), ("val", val_frames)]:
        for frame_path in frames:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            frame_name = os.path.basename(frame_path)
            base_name = os.path.splitext(frame_name)[0]

            # Detect flowers
            detections = detector.match_templates(frame, threshold=threshold)

            # Copy image to dataset
            img_dest = os.path.join(output_dir, "images", split_name, frame_name)
            shutil.copy(frame_path, img_dest)

            # Create YOLO format label file
            label_path = os.path.join(output_dir, "labels", split_name, f"{base_name}.txt")

            with open(label_path, "w") as f:
                for det in detections:
                    x, y, bw, bh = det['bbox']

                    # Convert to YOLO format (normalized x_center, y_center, width, height)
                    x_center = (x + bw / 2) / w
                    y_center = (y + bh / 2) / h
                    norm_w = bw / w
                    norm_h = bh / h

                    # Class 0 = zinnia
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                    total_annotations += 1

            print(f"{split_name}/{frame_name}: {len(detections)} annotations")

    # Create dataset YAML config
    yaml_content = f"""# Zinnia Detection Dataset
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

# Classes
names:
  0: zinnia
"""

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n{'='*50}")
    print(f"Dataset created!")
    print(f"Total annotations: {total_annotations}")
    print(f"Config: {yaml_path}")
    print(f"{'='*50}")
    print("\nNext step: Review annotations, then run train_yolo.py")

    return yaml_path

if __name__ == "__main__":
    create_yolo_annotations(threshold=0.6)
