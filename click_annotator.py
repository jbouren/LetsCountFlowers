#!/usr/bin/env python3
"""
Click-to-Annotate Tool for Zinnia Labeling
- Left click: Add box at click location (default size)
- Right drag: Pan/move the view
- Scroll wheel: Zoom in/out
- Middle click or Shift+Left click: Remove nearest box
- Arrow keys: Navigate frames
- 's': Save and next frame
- 'q': Quit
- 'r': Reset current frame annotations
- '+'/'-': Adjust box size
- 'z': Reset zoom/pan
"""

import cv2
import os
import glob
import numpy as np

class ClickAnnotator:
    def __init__(self, frames_dir="selected_frames", output_dir="yolo_annotations"):
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        self.frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))
        self.current_idx = 0

        # Default box size (will be centered on click)
        self.box_size = 80  # pixels

        # Annotations for current frame: list of (x_center, y_center, width, height) in pixels
        self.annotations = []

        # Zoom and pan state
        self.zoom = 1.0
        self.pan_x = 0  # Pan offset in original image coordinates
        self.pan_y = 0
        self.dragging = False
        self.drag_start = None
        self.shift_held = False

        # Cache current frame and mask for auto-sizing
        self.current_frame = None
        self.flower_mask = None

        # Create output directories
        os.makedirs(f"{output_dir}/images/train", exist_ok=True)
        os.makedirs(f"{output_dir}/images/val", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/val", exist_ok=True)

        # Skip to first unannotated frame
        self.skip_annotated_frames()

        # Load existing annotations if any
        self.load_annotations()

        print(f"Found {len(self.frame_files)} frames")
        print("\nControls:")
        print("  Left click      : Add box")
        print("  Shift+Click     : Remove nearest box")
        print("  Right drag      : Pan view")
        print("  w/e             : Zoom in/out")
        print("  [/]             : Adjust box size")
        print("  s               : Save & next frame")
        print("  a/d             : Prev/next frame")
        print("  r               : Reset frame annotations")
        print("  z               : Reset zoom/pan")
        print("  q               : Quit (saves current)")
        print(f"\nStarting with box size: {self.box_size}px")

    def detect_flower_at_point(self, x, y):
        """Detect flower size at clicked point using local color region"""
        if self.current_frame is None:
            return self.box_size, self.box_size

        h, w = self.current_frame.shape[:2]
        x, y = int(x), int(y)

        # Extract small region around click (150x150)
        search_size = 75
        x1 = max(0, x - search_size)
        y1 = max(0, y - search_size)
        x2 = min(w, x + search_size)
        y2 = min(h, y + search_size)

        crop = self.current_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return self.box_size, self.box_size

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Color ranges for flowers
        flower_colors = [
            ([0, 30, 40], [10, 255, 255]),     # Red
            ([170, 30, 40], [180, 255, 255]),  # Red wrap
            ([10, 30, 40], [25, 255, 255]),    # Orange
            ([25, 30, 40], [40, 255, 255]),    # Yellow
            ([140, 30, 40], [170, 255, 255]),  # Pink/Magenta
        ]

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in flower_colors:
            m = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.bitwise_or(mask, m)

        # Find contours in the local region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find contour closest to center of crop (where we clicked)
        center_x, center_y = (x2 - x1) // 2, (y2 - y1) // 2
        best_contour = None
        min_dist = float('inf')

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100 or area > 10000:  # Filter by area
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            cx, cy = bx + bw // 2, by + bh // 2
            dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_contour = contour

        if best_contour is not None:
            bx, by, bw, bh = cv2.boundingRect(best_contour)
            # Add padding, enforce min/max
            pad = 10
            bw = min(120, max(50, bw + pad * 2))
            bh = min(120, max(50, bh + pad * 2))
            return bw, bh

        return self.box_size, self.box_size

    def skip_annotated_frames(self):
        """Skip to first frame without annotations"""
        for idx, frame_path in enumerate(self.frame_files):
            frame_name = os.path.basename(frame_path)
            base_name = os.path.splitext(frame_name)[0]

            # Check if annotation exists in either train or val
            has_annotation = False
            for split in ["train", "val"]:
                label_path = f"{self.output_dir}/labels/{split}/{base_name}.txt"
                if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                    has_annotation = True
                    break

            if not has_annotation:
                self.current_idx = idx
                print(f"Skipping {idx} already-annotated frames, starting at frame {idx + 1}")
                return

        # All frames annotated, start at first
        print("All frames have annotations! Starting at frame 1 for review.")
        self.current_idx = 0

    def load_annotations(self):
        """Load existing annotations for current frame"""
        self.annotations = []
        frame_path = self.frame_files[self.current_idx]
        frame_name = os.path.basename(frame_path)
        base_name = os.path.splitext(frame_name)[0]

        # Check both train and val
        for split in ["train", "val"]:
            label_path = f"{self.output_dir}/labels/{split}/{base_name}.txt"
            if os.path.exists(label_path):
                frame = cv2.imread(frame_path)
                h, w = frame.shape[:2]

                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            # Convert YOLO format back to pixels
                            x_center = float(parts[1]) * w
                            y_center = float(parts[2]) * h
                            box_w = float(parts[3]) * w
                            box_h = float(parts[4]) * h
                            self.annotations.append([x_center, y_center, box_w, box_h])
                break

    def save_annotations(self):
        """Save annotations in YOLO format"""
        frame_path = self.frame_files[self.current_idx]
        frame_name = os.path.basename(frame_path)
        base_name = os.path.splitext(frame_name)[0]

        frame = cv2.imread(frame_path)
        h, w = frame.shape[:2]

        # Determine split (80% train, 20% val)
        split = "train" if self.current_idx < len(self.frame_files) * 0.8 else "val"

        # Copy image
        import shutil
        shutil.copy(frame_path, f"{self.output_dir}/images/{split}/{frame_name}")

        # Save labels
        label_path = f"{self.output_dir}/labels/{split}/{base_name}.txt"
        with open(label_path, 'w') as f:
            for ann in self.annotations:
                x_center, y_center, box_w, box_h = ann
                # Convert to YOLO format (normalized)
                f.write(f"0 {x_center/w:.6f} {y_center/h:.6f} {box_w/w:.6f} {box_h/h:.6f}\n")

        print(f"Saved {len(self.annotations)} annotations for {frame_name}")

    def screen_to_image(self, x, y):
        """Convert screen coordinates to original image coordinates"""
        img_x = x / self.zoom + self.pan_x
        img_y = y / self.zoom + self.pan_y
        return img_x, img_y

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        self.shift_held = flags & cv2.EVENT_FLAG_SHIFTKEY

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.shift_held:
                # Shift+Left click: Remove nearest box
                img_x, img_y = self.screen_to_image(x, y)
                if self.annotations:
                    distances = [np.sqrt((ann[0]-img_x)**2 + (ann[1]-img_y)**2) for ann in self.annotations]
                    nearest_idx = np.argmin(distances)
                    if distances[nearest_idx] < self.box_size:
                        self.annotations.pop(nearest_idx)
            else:
                # Add box centered at click (convert screen to image coords)
                img_x, img_y = self.screen_to_image(x, y)
                # Auto-detect flower size at this point
                box_w, box_h = self.detect_flower_at_point(img_x, img_y)
                self.annotations.append([img_x, img_y, box_w, box_h])

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Start dragging for pan
            self.dragging = True
            self.drag_start = (x, y)

        elif event == cv2.EVENT_RBUTTONUP:
            self.dragging = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.drag_start:
                # Pan the view
                dx = (self.drag_start[0] - x) / self.zoom
                dy = (self.drag_start[1] - y) / self.zoom
                self.pan_x += dx
                self.pan_y += dy
                self.drag_start = (x, y)

        elif event == cv2.EVENT_MBUTTONDOWN:
            # Middle click: Remove nearest box
            img_x, img_y = self.screen_to_image(x, y)
            if self.annotations:
                distances = [np.sqrt((ann[0]-img_x)**2 + (ann[1]-img_y)**2) for ann in self.annotations]
                nearest_idx = np.argmin(distances)
                if distances[nearest_idx] < self.box_size:
                    self.annotations.pop(nearest_idx)


    def draw_frame(self):
        """Draw current frame with annotations"""
        frame_path = self.frame_files[self.current_idx]
        frame = cv2.imread(frame_path)
        h, w = frame.shape[:2]

        # Cache frame for auto-sizing
        if self.current_frame is None or self.current_frame.shape != frame.shape:
            self.current_frame = frame.copy()
            self.flower_mask = None  # Reset mask for new frame

        # Draw annotations on original frame first
        for ann in self.annotations:
            x_center, y_center, box_w, box_h = ann
            x1 = int(x_center - box_w/2)
            y1 = int(y_center - box_h/2)
            x2 = int(x_center + box_w/2)
            y2 = int(y_center + box_h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (int(x_center), int(y_center)), 3, (0, 255, 0), -1)

        # Apply zoom and pan
        # Calculate the visible region in original image coordinates
        view_w = int(w / self.zoom)
        view_h = int(h / self.zoom)

        # Clamp pan to valid range
        self.pan_x = max(0, min(self.pan_x, w - view_w))
        self.pan_y = max(0, min(self.pan_y, h - view_h))

        x1 = int(self.pan_x)
        y1 = int(self.pan_y)
        x2 = min(w, int(self.pan_x + view_w))
        y2 = min(h, int(self.pan_y + view_h))

        # Crop and resize
        cropped = frame[y1:y2, x1:x2]
        if cropped.size > 0:
            display = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            display = frame.copy()

        # Draw info overlay (on zoomed display)
        frame_name = os.path.basename(frame_path)
        info = f"Frame {self.current_idx+1}/{len(self.frame_files)}: {frame_name}"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"Annotations: {len(self.annotations)} | Box: {self.box_size}px | Zoom: {self.zoom:.1f}x",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, "Click=add | Shift+Click=remove | RightDrag=pan | w/e=zoom | z=reset",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        return display

    def run(self):
        """Main annotation loop"""
        cv2.namedWindow("Zinnia Annotator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Zinnia Annotator", self.mouse_callback)

        while True:
            display = self.draw_frame()
            cv2.imshow("Zinnia Annotator", display)

            key = cv2.waitKey(1) & 0xFF  # Faster response

            if key == ord('q'):
                self.save_annotations()
                break

            elif key == ord('s'):
                self.save_annotations()
                if self.current_idx < len(self.frame_files) - 1:
                    self.current_idx += 1
                    self.load_annotations()
                    self.zoom = 1.0
                    self.pan_x = 0
                    self.pan_y = 0
                    self.current_frame = None  # Reset cache
                    self.flower_mask = None
                    print(f"Frame {self.current_idx + 1}/{len(self.frame_files)}")
                else:
                    print("Reached last frame!")

            # Arrow keys: left=81/2, right=83/3 (varies by OS)
            elif (key == ord('a') or key == 81 or key == 2) and self.current_idx > 0:
                self.save_annotations()
                self.current_idx -= 1
                self.load_annotations()
                self.zoom = 1.0
                self.pan_x = 0
                self.pan_y = 0
                self.current_frame = None
                self.flower_mask = None
                print(f"Frame {self.current_idx + 1}/{len(self.frame_files)}")

            elif (key == ord('d') or key == 83 or key == 3) and self.current_idx < len(self.frame_files) - 1:
                self.save_annotations()
                self.current_idx += 1
                self.load_annotations()
                self.zoom = 1.0
                self.pan_x = 0
                self.pan_y = 0
                self.current_frame = None
                self.flower_mask = None
                print(f"Frame {self.current_idx + 1}/{len(self.frame_files)}")

            elif key == ord('r'):
                self.annotations = []

            elif key == ord('z'):
                # Reset zoom and pan
                self.zoom = 1.0
                self.pan_x = 0
                self.pan_y = 0
                print("Zoom/pan reset")

            elif key == ord('w'):
                # Zoom in
                self.zoom = min(5.0, self.zoom * 1.3)
                print(f"Zoom: {self.zoom:.1f}x")

            elif key == ord('e'):
                # Zoom out
                self.zoom = max(0.3, self.zoom / 1.3)
                print(f"Zoom: {self.zoom:.1f}x")

            elif key == ord(']') or key == ord('+') or key == ord('='):
                self.box_size = min(200, self.box_size + 10)
                print(f"Box size: {self.box_size}px")

            elif key == ord('[') or key == ord('-'):
                self.box_size = max(20, self.box_size - 10)
                print(f"Box size: {self.box_size}px")

        cv2.destroyAllWindows()

        # Create dataset.yaml
        yaml_content = f"""# Zinnia Dataset
path: {os.path.abspath(self.output_dir)}
train: images/train
val: images/val

names:
  0: zinnia
"""
        with open(f"{self.output_dir}/dataset.yaml", "w") as f:
            f.write(yaml_content)

        print(f"\nDataset saved to: {self.output_dir}/")
        print(f"Config: {self.output_dir}/dataset.yaml")
        print("\nTo train: python3 train_yolo.py")


if __name__ == "__main__":
    annotator = ClickAnnotator()
    annotator.run()
