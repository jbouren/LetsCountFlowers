#!/usr/bin/env python3
"""Debug color detection - show which color range triggered each detection"""

import cv2
import numpy as np

def debug_color_detection(image_path, output_dir="outputs/color_debug"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    
    # Define color ranges with names and debug colors
    color_ranges = [
        ("red1", [0, 80, 80], [10, 255, 255], (0, 0, 255)),       # Red - show as red
        ("red2", [170, 80, 80], [180, 255, 255], (0, 0, 200)),    # Red wrap - dark red
        ("orange", [10, 80, 80], [25, 255, 255], (0, 128, 255)),  # Orange - show as orange
        ("yellow", [25, 60, 80], [35, 255, 255], (0, 255, 255)),  # Yellow - show as yellow
        ("pink", [140, 50, 80], [170, 255, 255], (255, 0, 255)),  # Pink - show as magenta
        ("white", [0, 0, 200], [180, 40, 255], (255, 255, 255)),  # White - show as white
    ]
    
    # Create individual masks and combined visualization
    combined_viz = img.copy()
    all_detections = []
    
    for name, lower, upper, color in color_ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 15000:
                x, y, bw, bh = cv2.boundingRect(contour)
                aspect = max(bw, bh) / (min(bw, bh) + 1)
                if aspect < 3:
                    cv2.rectangle(combined_viz, (x, y), (x+bw, y+bh), color, 2)
                    count += 1
                    all_detections.append((name, x, y, bw, bh, area))
        
        print(f"{name}: {count} detections")
        
        # Save individual mask
        mask_viz = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f"{output_dir}/mask_{name}.jpg", mask_viz)
    
    # Add legend
    y_offset = 30
    for name, lower, upper, color in color_ranges:
        cv2.putText(combined_viz, name, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 25
    
    cv2.imwrite(f"{output_dir}/combined_debug.jpg", combined_viz)
    
    # Sample some pixels from sand/green areas to see their HSV values
    print("\n--- Sampling suspicious areas ---")
    # Sample from corners (likely sand/background)
    sample_points = [
        ("top-left", 50, 50),
        ("top-right", w-50, 50),
        ("bottom-left", 50, h-50),
        ("bottom-right", w-50, h-50),
        ("center", w//2, h//2),
    ]
    
    for name, x, y in sample_points:
        if 0 <= x < w and 0 <= y < h:
            h_val, s_val, v_val = hsv[y, x]
            b, g, r = img[y, x]
            print(f"{name} ({x},{y}): HSV=({h_val}, {s_val}, {v_val}) BGR=({b},{g},{r})")
    
    print(f"\nDebug images saved to {output_dir}/")
    return all_detections

if __name__ == "__main__":
    debug_color_detection("selected_frames/frame_000000.jpg")
