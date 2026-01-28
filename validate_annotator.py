#!/usr/bin/env python3
"""
Fast Y/N Validation Annotator
- Color detection proposes flower candidates
- You press Y to accept, N to reject
- Much faster than clicking each flower!

Controls:
  y = Accept this detection
  n = Reject this detection
  s = Skip to next frame
  q = Quit and save
  b = Go back one detection
"""

import cv2
import numpy as np
import os
import glob

class ValidateAnnotator:
    def __init__(self, frames_dir="selected_frames", output_dir="yolo_validated"):
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        self.frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))
        self.current_frame_idx = 0
        self.current_detection_idx = 0

        self.detections = []  # Current frame's proposed detections
        self.accepted = []    # Accepted detections for current frame

        # Create output dirs
        os.makedirs(f"{output_dir}/images/train", exist_ok=True)
        os.makedirs(f"{output_dir}/images/val", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/val", exist_ok=True)

        print(f"Found {len(self.frame_files)} frames")
        print("\nControls:")
        print("  y = Accept detection")
        print("  n = Reject detection")
        print("  s = Skip to next frame")
        print("  q = Quit and save")
        print("  b = Go back one detection")

    def detect_candidates(self, frame):
        """Use color detection to find flower candidates"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Color ranges for zinnias - wider tolerance to catch more candidates
        flower_colors = [
            ([0, 30, 40], [10, 255, 255]),    # Red
            ([170, 30, 40], [180, 255, 255]), # Red wrap
            ([10, 30, 40], [25, 255, 255]),   # Orange
            ([25, 30, 40], [40, 255, 255]),   # Yellow (expanded)
            ([140, 30, 40], [170, 255, 255]), # Pink/Magenta
            ([0, 0, 180], [180, 30, 255]),    # White/cream flowers
        ]

        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in flower_colors:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 150 < area < 25000:  # Filter by area (lowered min to catch smaller flowers)
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                detections.append({
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': area
                })

        # Sort by position (top-left to bottom-right) for consistent order
        detections.sort(key=lambda d: (d['center'][1] // 100, d['center'][0]))

        return detections

    def save_frame_annotations(self):
        """Save accepted annotations for current frame"""
        if not self.accepted:
            return

        frame_path = self.frame_files[self.current_frame_idx]
        frame = cv2.imread(frame_path)
        h, w = frame.shape[:2]

        frame_name = os.path.basename(frame_path)
        base_name = os.path.splitext(frame_name)[0]

        # 80/20 split
        split = "train" if self.current_frame_idx < len(self.frame_files) * 0.8 else "val"

        # Copy image
        import shutil
        shutil.copy(frame_path, f"{self.output_dir}/images/{split}/{frame_name}")

        # Save labels
        label_path = f"{self.output_dir}/labels/{split}/{base_name}.txt"
        with open(label_path, 'w') as f:
            for det in self.accepted:
                x, y, bw, bh = det['bbox']
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                norm_w = bw / w
                norm_h = bh / h
                f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        print(f"Saved {len(self.accepted)} annotations for {frame_name}")

    def draw_display(self, frame):
        """Draw the current state"""
        display = frame.copy()
        h, w = frame.shape[:2]

        # Draw accepted detections in green
        for det in self.accepted:
            x, y, bw, bh = det['bbox']
            cv2.rectangle(display, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        # Draw current detection in yellow (highlighted)
        if self.current_detection_idx < len(self.detections):
            det = self.detections[self.current_detection_idx]
            x, y, bw, bh = det['bbox']
            cv2.rectangle(display, (x, y), (x + bw, y + bh), (0, 255, 255), 3)
            cv2.circle(display, det['center'], 8, (0, 255, 255), -1)

            # Draw zoomed inset
            pad = 50
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                zoom = cv2.resize(crop, (200, 200))
                display[10:210, w-210:w-10] = zoom
                cv2.rectangle(display, (w-210, 10), (w-10, 210), (0, 255, 255), 2)

        # Draw remaining detections in gray
        for i, det in enumerate(self.detections):
            if i > self.current_detection_idx:
                x, y, bw, bh = det['bbox']
                cv2.rectangle(display, (x, y), (x + bw, y + bh), (128, 128, 128), 1)

        # Info overlay
        frame_name = os.path.basename(self.frame_files[self.current_frame_idx])
        cv2.putText(display, f"Frame {self.current_frame_idx+1}/{len(self.frame_files)}: {frame_name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"Detection {self.current_detection_idx+1}/{len(self.detections)}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"Accepted: {len(self.accepted)}",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "Y=accept  N=reject  S=skip frame  Q=quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        return display

    def run(self):
        """Main loop"""
        cv2.namedWindow("Validate Annotations", cv2.WINDOW_NORMAL)

        while self.current_frame_idx < len(self.frame_files):
            # Load frame and detect candidates
            frame_path = self.frame_files[self.current_frame_idx]
            frame = cv2.imread(frame_path)

            if self.current_detection_idx == 0:
                self.detections = self.detect_candidates(frame)
                self.accepted = []
                print(f"\nFrame {self.current_frame_idx + 1}: Found {len(self.detections)} candidates")

            while self.current_detection_idx < len(self.detections):
                display = self.draw_display(frame)
                cv2.imshow("Validate Annotations", display)

                key = cv2.waitKey(0) & 0xFF

                if key == ord('y'):
                    self.accepted.append(self.detections[self.current_detection_idx])
                    self.current_detection_idx += 1

                elif key == ord('n'):
                    self.current_detection_idx += 1

                elif key == ord('b') and self.current_detection_idx > 0:
                    self.current_detection_idx -= 1
                    # Remove from accepted if it was there
                    if self.accepted and self.detections[self.current_detection_idx] in self.accepted:
                        self.accepted.remove(self.detections[self.current_detection_idx])

                elif key == ord('s'):
                    # Skip to next frame
                    break

                elif key == ord('q'):
                    self.save_frame_annotations()
                    cv2.destroyAllWindows()
                    self.create_dataset_yaml()
                    return

            # Done with this frame
            self.save_frame_annotations()
            self.current_frame_idx += 1
            self.current_detection_idx = 0

        cv2.destroyAllWindows()
        self.create_dataset_yaml()
        print("\nAll frames processed!")

    def create_dataset_yaml(self):
        """Create dataset config file"""
        yaml_content = f"""# Validated Zinnia Dataset
path: {os.path.abspath(self.output_dir)}
train: images/train
val: images/val

names:
  0: zinnia
"""
        with open(f"{self.output_dir}/dataset.yaml", "w") as f:
            f.write(yaml_content)
        print(f"Dataset config: {self.output_dir}/dataset.yaml")


if __name__ == "__main__":
    annotator = ValidateAnnotator()
    annotator.run()
