#!/usr/bin/env python3
"""
Preprocess images to emphasize flowers by desaturating non-flower colors.
Flowers stay vibrant, background (greens/sand) becomes grayscale.
"""

import cv2
import numpy as np
import glob
import os

def desaturate_non_flowers(image):
    """Keep flower colors vibrant, desaturate everything else"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Flower color ranges (what to KEEP saturated)
    flower_ranges = [
        ([0, 80, 50], [10, 255, 255]),       # Red
        ([170, 80, 50], [180, 255, 255]),    # Red wrap
        ([10, 100, 50], [25, 255, 255]),     # Orange
        ([22, 130, 100], [35, 255, 255]),    # Yellow (high sat only)
        ([140, 50, 50], [170, 255, 255]),    # Pink/Magenta
    ]
    
    # Create mask of flower pixels
    flower_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in flower_ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        flower_mask = cv2.bitwise_or(flower_mask, mask)
    
    # Dilate slightly to include flower edges
    kernel = np.ones((5, 5), np.uint8)
    flower_mask = cv2.dilate(flower_mask, kernel, iterations=1)
    
    # Create grayscale version
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Blend: flowers stay colored, rest becomes gray
    result = image.copy()
    result[flower_mask == 0] = gray_bgr[flower_mask == 0]
    
    return result


def preview_preprocessing(image_path, output_path=None):
    """Preview the preprocessing on a single image"""
    img = cv2.imread(image_path)
    processed = desaturate_non_flowers(img)
    
    if output_path:
        cv2.imwrite(output_path, processed)
        print(f"Saved: {output_path}")
    
    return processed


def preprocess_dataset(input_dir="yolo_auto", output_dir="yolo_preprocessed"):
    """Preprocess entire dataset"""
    for split in ["train", "val"]:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)
        
        # Copy labels as-is
        for label_file in glob.glob(f"{input_dir}/labels/{split}/*.txt"):
            out_path = f"{output_dir}/labels/{split}/{os.path.basename(label_file)}"
            with open(label_file) as f:
                content = f.read()
            with open(out_path, 'w') as f:
                f.write(content)
        
        # Preprocess images
        for img_path in glob.glob(f"{input_dir}/images/{split}/*.jpg"):
            img_name = os.path.basename(img_path)
            out_path = f"{output_dir}/images/{split}/{img_name}"
            
            img = cv2.imread(img_path)
            processed = desaturate_non_flowers(img)
            cv2.imwrite(out_path, processed)
            print(f"Processed: {img_name}")
    
    # Create dataset.yaml
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: zinnia
"""
    with open(f"{output_dir}/dataset.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"\nPreprocessed dataset saved to: {output_dir}/")


if __name__ == "__main__":
    # Preview on one image
    preview_preprocessing(
        "selected_frames/frame_000000.jpg",
        "outputs/preprocessed_preview.jpg"
    )
    print("\nPreview saved. Run preprocess_dataset() to process all images.")
