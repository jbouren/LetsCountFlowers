#!/usr/bin/env python3
"""
Generate YOLO training data from color detection
Uses the tuned color detector to create properly-sized bounding boxes
"""

import cv2
import numpy as np
import glob
import os
import shutil
from color_detector import detect_zinnias_by_color

def generate_yolo_dataset(frames_dir="selected_frames", output_dir="yolo_auto", train_ratio=0.8):
    """Generate YOLO dataset from color detections"""
    
    # Create output structure
    for split in ["train", "val"]:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)
    
    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))
    
    # Split into train/val
    split_idx = int(len(frame_files) * train_ratio)
    train_frames = frame_files[:split_idx]
    val_frames = frame_files[split_idx:]
    
    total_annotations = 0
    
    for frames, split in [(train_frames, "train"), (val_frames, "val")]:
        for frame_path in frames:
            img = cv2.imread(frame_path)
            h, w = img.shape[:2]
            
            frame_name = os.path.basename(frame_path)
            base_name = os.path.splitext(frame_name)[0]
            
            # Get detections from color detector
            detections = detect_zinnias_by_color(frame_path)
            
            # Copy image
            shutil.copy(frame_path, f"{output_dir}/images/{split}/{frame_name}")
            
            # Write YOLO format labels
            label_path = f"{output_dir}/labels/{split}/{base_name}.txt"
            with open(label_path, 'w') as f:
                for det in detections:
                    x, y, bw, bh = det['bbox']
                    # Convert to YOLO format (normalized center + size)
                    x_center = (x + bw / 2) / w
                    y_center = (y + bh / 2) / h
                    norm_w = bw / w
                    norm_h = bh / h
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            
            print(f"{split}: {frame_name} - {len(detections)} flowers")
            total_annotations += len(detections)
    
    # Create dataset.yaml
    yaml_content = f"""# Auto-generated Zinnia Dataset from Color Detection
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: zinnia
"""
    with open(f"{output_dir}/dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    print(f"\n{'='*50}")
    print(f"Dataset generated!")
    print(f"Train: {len(train_frames)} images")
    print(f"Val: {len(val_frames)} images")
    print(f"Total annotations: {total_annotations}")
    print(f"Output: {output_dir}/")
    print(f"{'='*50}")
    
    return output_dir


if __name__ == "__main__":
    generate_yolo_dataset()
