#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 21:40:20 2025

@author: jacob
"""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from collections import defaultdict

class PanningZinniaCounter:
    def __init__(self, overlap_threshold=0.3, movement_threshold=50):
        """
        Initialize the panning video flower counter
        
        Args:
            overlap_threshold: Minimum overlap ratio to consider flowers as duplicates
            movement_threshold: Minimum pixel movement between frames to detect significant panning
        """
        self.overlap_threshold = overlap_threshold
        self.movement_threshold = movement_threshold
        self.tracked_flowers = []  # Store all unique flowers found
        self.frame_data = []       # Store data for each frame
        
    def detect_camera_movement(self, prev_frame, curr_frame):
        """Detect camera movement between frames using optical flow"""
        if prev_frame is None:
            return 0, 0
            
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame
        
        # Detect keypoints and compute optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2, 
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Detect corner points in previous frame
        corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, 
                                        minDistance=10, blockSize=3)
        
        if corners is None or len(corners) < 10:
            return 0, 0
            
        # Calculate optical flow
        new_corners, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None, **lk_params)
        
        # Filter good points
        good_old = corners[status == 1]
        good_new = new_corners[status == 1]
        
        if len(good_old) < 5:
            return 0, 0
            
        # Calculate median movement (more robust than mean)
        movements = good_new - good_old
        median_dx = np.median(movements[:, 0])
        median_dy = np.median(movements[:, 1])
        
        return median_dx, median_dy
    
    def detect_flowers_in_frame(self, frame):
        """Detect flowers in a single frame - same as before but returns bounding boxes"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Color ranges for zinnias
        flower_colors = [
            ([0, 50, 50], [10, 255, 255]),    # Red
            ([170, 50, 50], [180, 255, 255]), # Red (wrap-around)
            ([10, 50, 50], [25, 255, 255]),   # Orange
            ([25, 50, 50], [35, 255, 255]),   # Yellow
            ([120, 50, 50], [150, 255, 255]), # Purple
        ]
        
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in flower_colors:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Clean up mask
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours and extract flower regions
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        flowers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 15000:  # Filter by area
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  # Filter by circularity
                        # Get bounding box and center
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        flowers.append({
                            'bbox': (x, y, w, h),
                            'center': (center_x, center_y),
                            'area': area,
                            'contour': contour,
                            'circularity': circularity
                        })
        
        return flowers
    
    def calculate_overlap(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def is_duplicate_flower(self, new_flower, camera_dx, camera_dy):
        """Check if a flower is a duplicate of an already tracked flower"""
        new_center = new_flower['center']
        new_bbox = new_flower['bbox']
        
        for tracked_flower in self.tracked_flowers:
            # Predict where the tracked flower should be based on camera movement
            old_center = tracked_flower['last_center']
            predicted_center = (old_center[0] - camera_dx, old_center[1] - camera_dy)
            
            # Calculate distance between predicted and actual position
            distance = np.sqrt((new_center[0] - predicted_center[0])**2 + 
                             (new_center[1] - predicted_center[1])**2)
            
            # If close enough, check overlap with original bounding box
            if distance < 100:  # Adjust this threshold based on your video
                # Adjust tracked flower's bbox based on camera movement
                old_bbox = tracked_flower['bbox']
                predicted_bbox = (old_bbox[0] - camera_dx, old_bbox[1] - camera_dy, 
                                old_bbox[2], old_bbox[3])
                
                overlap = self.calculate_overlap(new_bbox, predicted_bbox)
                if overlap > self.overlap_threshold:
                    # Update the tracked flower's position
                    tracked_flower['last_center'] = new_center
                    tracked_flower['bbox'] = new_bbox
                    tracked_flower['frames_seen'] += 1
                    return True
        
        return False
    
    def process_video_smart(self, video_path, output_path=None, sample_every_n_frames=1):
        """
        Process video with smart duplicate detection
        
        Args:
            sample_every_n_frames: Process every nth frame to speed up analysis
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        prev_frame = None
        frame_number = 0
        processed_frames = 0
        
        print(f"Processing video with {total_frames} frames...")
        print(f"Sampling every {sample_every_n_frames} frame(s)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every nth frame
            if frame_number % sample_every_n_frames != 0:
                frame_number += 1
                continue
            
            # Detect camera movement
            camera_dx, camera_dy = self.detect_camera_movement(prev_frame, frame)
            camera_movement = np.sqrt(camera_dx**2 + camera_dy**2)
            
            # Detect flowers in current frame
            current_flowers = self.detect_flowers_in_frame(frame)
            
            # Check each flower against tracked flowers
            new_flowers_count = 0
            frame_flowers = []
            
            for flower in current_flowers:
                if not self.is_duplicate_flower(flower, camera_dx, camera_dy):
                    # This is a new flower
                    new_flower = {
                        'id': len(self.tracked_flowers),
                        'first_seen_frame': frame_number,
                        'last_center': flower['center'],
                        'bbox': flower['bbox'],
                        'area': flower['area'],
                        'frames_seen': 1
                    }
                    self.tracked_flowers.append(new_flower)
                    new_flowers_count += 1
                
                frame_flowers.append(flower)
            
            # Store frame data
            self.frame_data.append({
                'frame_number': frame_number,
                'flowers_in_frame': len(current_flowers),
                'new_flowers': new_flowers_count,
                'total_unique_flowers': len(self.tracked_flowers),
                'camera_movement': camera_movement
            })
            
            # Create visualization
            result_frame = frame.copy()
            
            # Draw all flowers in frame
            for flower in frame_flowers:
                x, y, w, h = flower['bbox']
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(result_frame, flower['center'], 3, (0, 255, 0), -1)
            
            # Add information overlay
            cv2.putText(result_frame, f'Frame flowers: {len(current_flowers)}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(result_frame, f'Total unique: {len(self.tracked_flowers)}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(result_frame, f'New this frame: {new_flowers_count}', (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(result_frame, f'Movement: {camera_movement:.1f}px', (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(result_frame, f'Frame: {frame_number}', (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if out:
                out.write(result_frame)
            
            if processed_frames % 10 == 0:
                print(f"Processed frame {frame_number}, unique flowers so far: {len(self.tracked_flowers)}")
            
            prev_frame = frame.copy()
            frame_number += 1
            processed_frames += 1
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        return len(self.tracked_flowers)
    
    def manual_frame_selection(self, video_path, output_frames_dir="selected_frames"):
        """
        Alternative approach: Select frames with minimal overlap for manual analysis
        """
        import os
        os.makedirs(output_frames_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Analyze camera movement throughout video
        prev_frame = None
        frame_number = 0
        movements = []
        
        print("Analyzing camera movement...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if prev_frame is not None:
                dx, dy = self.detect_camera_movement(prev_frame, frame)
                movement = np.sqrt(dx**2 + dy**2)
                movements.append((frame_number, movement))
            
            prev_frame = frame.copy()
            frame_number += 1
        
        cap.release()
        
        # Find frames with significant movement (indicating new areas)
        movement_threshold = np.percentile([m[1] for m in movements], 70)  # Top 30% of movements
        selected_frames = []
        
        # Always include first frame
        selected_frames.append(0)
        
        cumulative_movement = 0
        min_distance_between_frames = total_frames // 20  # At least 20 selected frames
        
        for frame_num, movement in movements:
            cumulative_movement += movement
            
            # Select frame if significant movement accumulated and enough distance from last selection
            if (cumulative_movement > movement_threshold * 5 and 
                frame_num - selected_frames[-1] > min_distance_between_frames):
                selected_frames.append(frame_num)
                cumulative_movement = 0
        
        # Save selected frames
        cap = cv2.VideoCapture(video_path)
        saved_count = 0
        
        for frame_num in selected_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                filename = os.path.join(output_frames_dir, f"frame_{frame_num:06d}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1
        
        cap.release()
        
        print(f"Saved {saved_count} frames to {output_frames_dir}")
        print(f"Selected frames: {selected_frames}")
        print("You can now manually count flowers in these frames and sum the results.")
        
        return selected_frames
    
    def generate_report(self, video_path):
        """Generate comprehensive analysis report"""
        if not self.frame_data:
            print("No data to analyze. Run process_video_smart first.")
            return
        
        print(f"\n=== Panning Video Flower Count Report ===")
        print(f"Video: {video_path}")
        print(f"Total unique flowers detected: {len(self.tracked_flowers)}")
        print(f"Frames processed: {len(self.frame_data)}")
        
        # Analyze tracking confidence
        well_tracked = sum(1 for f in self.tracked_flowers if f['frames_seen'] >= 3)
        print(f"Flowers seen in 3+ frames (high confidence): {well_tracked}")
        
        avg_flowers_per_frame = np.mean([fd['flowers_in_frame'] for fd in self.frame_data])
        print(f"Average flowers visible per frame: {avg_flowers_per_frame:.1f}")
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        frames = [fd['frame_number'] for fd in self.frame_data]
        flowers_per_frame = [fd['flowers_in_frame'] for fd in self.frame_data]
        plt.plot(frames, flowers_per_frame)
        plt.title('Flowers Detected Per Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('Flower Count')
        
        plt.subplot(2, 3, 2)
        cumulative_unique = [fd['total_unique_flowers'] for fd in self.frame_data]
        plt.plot(frames, cumulative_unique)
        plt.title('Cumulative Unique Flowers')
        plt.xlabel('Frame Number')
        plt.ylabel('Total Unique Flowers')
        
        plt.subplot(2, 3, 3)
        movements = [fd['camera_movement'] for fd in self.frame_data]
        plt.plot(frames, movements)
        plt.title('Camera Movement')
        plt.xlabel('Frame Number')
        plt.ylabel('Movement (pixels)')
        
        plt.subplot(2, 3, 4)
        new_per_frame = [fd['new_flowers'] for fd in self.frame_data]
        plt.plot(frames, new_per_frame)
        plt.title('New Flowers Discovered Per Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('New Flowers')
        
        plt.subplot(2, 3, 5)
        frames_seen = [f['frames_seen'] for f in self.tracked_flowers]
        plt.hist(frames_seen, bins=range(1, max(frames_seen) + 2), alpha=0.7)
        plt.title('Distribution of Tracking Duration')
        plt.xlabel('Frames Flower Was Visible')
        plt.ylabel('Number of Flowers')
        
        plt.subplot(2, 3, 6)
        areas = [f['area'] for f in self.tracked_flowers]
        plt.hist(areas, bins=30, alpha=0.7)
        plt.title('Distribution of Flower Sizes')
        plt.xlabel('Area (pixels)')
        plt.ylabel('Number of Flowers')
        
        plt.tight_layout()
        plt.savefig('panning_video_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
if __name__ == "__main__":
    counter = PanningZinniaCounter(overlap_threshold=0.3, movement_threshold=30)
    
    video_path = "/Users/jacob/code/LetsCountFlowers/drone.mov"
    
    # Option 1: Smart tracking (recommended)
    print("Running smart tracking analysis...")
    total_flowers = counter.process_video_smart(video_path, "tracked_output.mp4", sample_every_n_frames=2)
    counter.generate_report(video_path)
    
    # Option 2: Manual frame selection (if smart tracking isn't accurate enough)
    print("\nGenerating frames for manual counting...")
    selected_frames = counter.manual_frame_selection(video_path)
    print(f"For manual verification, count flowers in the selected frames and sum the results.")