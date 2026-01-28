#!/usr/bin/env python3
"""
Color-based Zinnia detector - no ML needed
Detects bright colored flowers against green/tan background
"""

import cv2
import numpy as np
import glob
import os

def has_dark_center(img, x, y, bw, bh, threshold=30):
    """Check if detection has a darker center (like a flower stigma)"""
    # Get center region (inner 40%)
    cx, cy = x + bw // 2, y + bh // 2
    inner_w, inner_h = max(5, bw // 3), max(5, bh // 3)

    x1_inner = max(0, cx - inner_w // 2)
    y1_inner = max(0, cy - inner_h // 2)
    x2_inner = min(img.shape[1], cx + inner_w // 2)
    y2_inner = min(img.shape[0], cy + inner_h // 2)

    # Get outer ring (the petals)
    x1_outer, y1_outer = x, y
    x2_outer, y2_outer = x + bw, y + bh

    # Sample brightness (use grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    center_region = gray[y1_inner:y2_inner, x1_inner:x2_inner]
    outer_region = gray[y1_outer:y2_outer, x1_outer:x2_outer]

    if center_region.size == 0 or outer_region.size == 0:
        return False

    center_brightness = np.mean(center_region)
    outer_brightness = np.mean(outer_region)

    # Flower has darker center than edges
    return (outer_brightness - center_brightness) > threshold


def detect_zinnias_by_color(image_path, output_path=None, min_area=200, max_area=15000):
    """Detect zinnias using HSV color segmentation"""

    img = cv2.imread(image_path)
    if img is None:
        return []

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Flower colors - tuned to avoid sand/dried grass
    flower_ranges = [
        ([0, 100, 100], [10, 255, 255]),     # Red - higher saturation
        ([170, 100, 100], [180, 255, 255]),  # Red wrap
        ([10, 120, 100], [22, 255, 255]),    # Orange - tighter hue, higher sat
        ([22, 150, 120], [32, 255, 255]),    # Yellow - VERY high sat only
        ([140, 60, 100], [170, 255, 255]),   # Pink/Magenta
    ]

    # White/cream flowers - will validate with dark center check
    white_range = ([0, 0, 200], [180, 50, 255])

    # Create combined mask for colored flowers
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in flower_ranges:
        m = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.bitwise_or(mask, m)

    # Separate mask for white candidates
    white_mask = cv2.inRange(hsv, np.array(white_range[0]), np.array(white_range[1]))
    
    # Clean up masks
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

    detections = []

    # Process colored flowers (no extra validation needed)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = max(bw, bh) / (min(bw, bh) + 1)
            if aspect < 3:
                detections.append({
                    'bbox': (x, y, bw, bh),
                    'center': (x + bw//2, y + bh//2),
                    'area': area,
                    'type': 'colored'
                })

    # Process white candidates - require dark center
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_accepted = 0
    white_rejected = 0
    for contour in white_contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = max(bw, bh) / (min(bw, bh) + 1)
            if aspect < 3:
                # Check for dark center (flower stigma)
                if has_dark_center(img, x, y, bw, bh, threshold=20):
                    detections.append({
                        'bbox': (x, y, bw, bh),
                        'center': (x + bw//2, y + bh//2),
                        'area': area,
                        'type': 'white'
                    })
                    white_accepted += 1
                else:
                    white_rejected += 1
    
    # Draw results if output path provided
    if output_path:
        result = img.copy()
        for det in detections:
            x, y, bw, bh = det['bbox']
            cv2.rectangle(result, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
        cv2.putText(result, f"Detected: {len(detections)} flowers", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(output_path, result)
    
    return detections


def process_frames(frames_dir="selected_frames", output_dir="outputs/color_detection"):
    """Process all frames with color detection"""
    os.makedirs(output_dir, exist_ok=True)
    
    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))
    
    total = 0
    for frame_path in frame_files:
        frame_name = os.path.basename(frame_path)
        output_path = os.path.join(output_dir, f"detected_{frame_name}")
        
        detections = detect_zinnias_by_color(frame_path, output_path)
        print(f"{frame_name}: {len(detections)} flowers")
        total += len(detections)
    
    print(f"\nTotal: {total} flowers across {len(frame_files)} frames")
    print(f"Output saved to: {output_dir}/")
    return total


if __name__ == "__main__":
    process_frames()
