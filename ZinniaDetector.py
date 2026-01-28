import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from collections import defaultdict
import os

class ImprovedZinniaDetector:
    def __init__(self):
        self.templates = []  # Store zinnia templates
        self.verified_flowers = []  # Store manually verified flowers
        self.false_positives = []   # Store rejected detections
        
    def load_templates(self, template_dir=".", template_pattern="zinnia_template_*.jpg"):
        """
        Load saved template images from disk
        
        Args:
            template_dir: Directory containing template images
            template_pattern: Filename pattern for templates (supports wildcards)
        """
        import glob
        
        # Find all template files matching the pattern
        pattern_path = os.path.join(template_dir, template_pattern)
        template_files = glob.glob(pattern_path)
        template_files.sort()  # Sort for consistent ordering
        
        self.templates = []  # Clear existing templates
        
        for template_file in template_files:
            template = cv2.imread(template_file)
            if template is not None:
                self.templates.append(template)
                print(f"Loaded template: {template_file} (size: {template.shape[1]}x{template.shape[0]})")
            else:
                print(f"Warning: Could not load template: {template_file}")
        
        print(f"Total templates loaded: {len(self.templates)}")
        return len(self.templates)
    
    def load_templates_from_list(self, template_paths):
        """
        Load templates from a specific list of file paths
        
        Args:
            template_paths: List of file paths to template images
        """
        self.templates = []
        
        for template_path in template_paths:
            template = cv2.imread(template_path)
            if template is not None:
                self.templates.append(template)
                print(f"Loaded template: {template_path}")
            else:
                print(f"Warning: Could not load template: {template_path}")
        
        print(f"Total templates loaded: {len(self.templates)}")
        return len(self.templates)
        
    def create_template_from_selection(self, image_path, save_template=True):
        """
        Interactive template creation - click and drag to select a zinnia
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
            
        # Create a copy for display
        display_img = image.copy()
        
        print("Instructions:")
        print("1. Click and drag to select a zinnia flower")
        print("2. Press 's' to save the selection as a template")
        print("3. Press 'r' to reset selection")
        print("4. Press 'q' to quit")
        
        # Global variables for mouse callback
        global selecting, start_point, end_point, current_rect
        selecting = False
        start_point = None
        end_point = None
        current_rect = None
        
        def mouse_callback(event, x, y, flags, param):
            global selecting, start_point, end_point, current_rect
            nonlocal display_img
            
            if event == cv2.EVENT_LBUTTONDOWN:
                selecting = True
                start_point = (x, y)
                end_point = (x, y)
                
            elif event == cv2.EVENT_MOUSEMOVE and selecting:
                end_point = (x, y)
                display_img = image.copy()
                cv2.rectangle(display_img, start_point, end_point, (0, 255, 0), 2)
                
            elif event == cv2.EVENT_LBUTTONUP:
                selecting = False
                end_point = (x, y)
                if start_point and end_point:
                    # Ensure rectangle has positive dimensions
                    x1, y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
                    x2, y2 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
                    current_rect = (x1, y1, x2-x1, y2-y1)
        
        cv2.namedWindow('Select Zinnia Template', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Select Zinnia Template', mouse_callback)
        
        templates_created = []
        
        while True:
            cv2.imshow('Select Zinnia Template', display_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and current_rect:
                # Save the selected region as template
                x, y, w, h = current_rect
                if w > 20 and h > 20:  # Minimum size check
                    template = image[y:y+h, x:x+w]
                    templates_created.append(template.copy())
                    
                    if save_template:
                        template_filename = f"zinnia_template_{len(self.templates)}.jpg"
                        cv2.imwrite(template_filename, template)
                        print(f"Template saved as {template_filename}")
                    
                    self.templates.append(template)
                    print(f"Template {len(templates_created)} created (size: {w}x{h})")
                    
                    # Reset for next selection
                    display_img = image.copy()
                    current_rect = None
                else:
                    print("Selection too small, please select a larger area")
                    
            elif key == ord('r'):
                # Reset
                display_img = image.copy()
                current_rect = None
                
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return templates_created
    
    def match_templates(self, image, threshold=0.7, method=cv2.TM_CCOEFF_NORMED):
        """
        Find zinnia flowers using template matching
        """
        if not self.templates:
            print("No templates available. Create templates first using create_template_from_selection()")
            return []
        
        matches = []
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for i, template in enumerate(self.templates):
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            h, w = gray_template.shape
            
            # Perform template matching at multiple scales
            for scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
                # Resize template
                new_w, new_h = int(w * scale), int(h * scale)
                if new_w < 10 or new_h < 10 or new_w > image.shape[1] or new_h > image.shape[0]:
                    continue
                    
                scaled_template = cv2.resize(gray_template, (new_w, new_h))
                
                # Template matching
                result = cv2.matchTemplate(gray_image, scaled_template, method)
                locations = np.where(result >= threshold)
                
                for pt in zip(*locations[::-1]):  # Switch x and y coordinates
                    matches.append({
                        'bbox': (pt[0], pt[1], new_w, new_h),
                        'center': (pt[0] + new_w//2, pt[1] + new_h//2),
                        'confidence': result[pt[1], pt[0]],
                        'template_id': i,
                        'scale': scale
                    })
        
        # Remove overlapping detections (Non-Maximum Suppression)
        if matches:
            matches = self.non_max_suppression(matches, overlap_threshold=0.3)
        
        return matches
    
    def non_max_suppression(self, detections, overlap_threshold=0.3):
        """Remove overlapping detections, keeping the one with highest confidence"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        for detection in detections:
            overlap_found = False
            for kept_detection in keep:
                overlap = self.calculate_overlap(detection['bbox'], kept_detection['bbox'])
                if overlap > overlap_threshold:
                    overlap_found = True
                    break
            
            if not overlap_found:
                keep.append(detection)
        
        return keep
    
    def calculate_overlap(self, bbox1, bbox2):
        """Calculate IoU (Intersection over Union) between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def manual_verification_interface(self, image, detections):
        """
        Interactive interface to verify/reject detections
        """
        if not detections:
            print("No detections to verify")
            return [], []
        
        verified = []
        rejected = []
        current_idx = 0
        
        print("\nManual Verification Interface:")
        print("'y' - Accept detection as zinnia")
        print("'n' - Reject detection") 
        print("'s' - Skip to next")
        print("'b' - Go back to previous")
        print("'q' - Quit verification")
        print(f"\nTotal detections to verify: {len(detections)}")
        
        while current_idx < len(detections):
            detection = detections[current_idx]
            
            # Create display image
            display_img = image.copy()
            
            # Highlight current detection
            x, y, w, h = detection['bbox']
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(display_img, detection['center'], 5, (0, 255, 0), -1)
            
            # Show other detections in different colors
            for i, other_det in enumerate(detections):
                if i != current_idx:
                    ox, oy, ow, oh = other_det['bbox']
                    color = (128, 128, 128)  # Gray for others
                    cv2.rectangle(display_img, (ox, oy), (ox + ow, oy + oh), color, 1)
            
            # Add information text
            info_text = [
                f"Detection {current_idx + 1}/{len(detections)}",
                f"Confidence: {detection['confidence']:.3f}",
                f"Template ID: {detection['template_id']}",
                f"Scale: {detection['scale']:.2f}",
                f"Size: {w}x{h}",
                "",
                "y=Accept, n=Reject, s=Skip, b=Back, q=Quit"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(display_img, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show zoomed view of the detection
            zoom_factor = 3
            crop = image[max(0, y-20):min(image.shape[0], y+h+20), 
                        max(0, x-20):min(image.shape[1], x+w+20)]
            if crop.size > 0:
                zoomed = cv2.resize(crop, None, fx=zoom_factor, fy=zoom_factor, 
                                  interpolation=cv2.INTER_CUBIC)
                # Place zoomed view in corner
                zoom_h, zoom_w = zoomed.shape[:2]
                if zoom_h < display_img.shape[0]//2 and zoom_w < display_img.shape[1]//2:
                    display_img[-zoom_h:, -zoom_w:] = zoomed
            
            cv2.imshow('Verify Detections', display_img)
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('y'):
                verified.append(detection)
                print(f"✓ Accepted detection {current_idx + 1}")
                current_idx += 1
                
            elif key == ord('n'):
                rejected.append(detection)
                print(f"✗ Rejected detection {current_idx + 1}")
                current_idx += 1
                
            elif key == ord('s'):
                print(f"⊘ Skipped detection {current_idx + 1}")
                current_idx += 1
                
            elif key == ord('b') and current_idx > 0:
                current_idx -= 1
                # Remove from verified/rejected if going back
                if verified and verified[-1] == detections[current_idx]:
                    verified.pop()
                elif rejected and rejected[-1] == detections[current_idx]:
                    rejected.pop()
                print(f"← Back to detection {current_idx + 1}")
                
            elif key == ord('q'):
                print("Verification stopped by user")
                break
        
        cv2.destroyAllWindows()
        
        print(f"\nVerification complete:")
        print(f"Accepted: {len(verified)} detections")
        print(f"Rejected: {len(rejected)} detections")
        
        return verified, rejected
    
    def analyze_frame_with_verification(self, image_path, template_threshold=0.6):
        """
        Complete workflow: detect flowers and verify results
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
        
        print(f"Analyzing frame: {image_path}")
        
        # Step 1: Template matching
        if self.templates:
            print("Using existing templates for detection...")
            detections = self.match_templates(image, threshold=template_threshold)
        else:
            print("No templates found. Please create templates first.")
            return None
        
        print(f"Found {len(detections)} potential matches")
        
        if not detections:
            print("No flowers detected")
            return {'verified': [], 'rejected': [], 'total_detected': 0}
        
        # Step 2: Manual verification
        verified, rejected = self.manual_verification_interface(image, detections)
        
        # Step 3: Store results for learning
        self.verified_flowers.extend(verified)
        self.false_positives.extend(rejected)
        
        return {
            'verified': verified,
            'rejected': rejected, 
            'total_detected': len(detections),
            'accuracy': len(verified) / len(detections) if detections else 0
        }
    
    def _get_next_index(self, directory, prefix):
        """Get the next available index for saving files"""
        import glob
        existing = glob.glob(os.path.join(directory, f"{prefix}_*.jpg"))
        if not existing:
            return 0
        # Extract numbers from filenames and find max
        indices = []
        for f in existing:
            basename = os.path.basename(f)
            # Extract number from filename like "flower_0001.jpg"
            try:
                num = int(basename.replace(prefix + "_", "").replace(".jpg", ""))
                indices.append(num)
            except ValueError:
                continue
        return max(indices) + 1 if indices else 0

    def save_training_data(self, image, output_dir="training_data"):
        """Save verified flowers and false positives for future training (accumulates across sessions)"""
        os.makedirs(output_dir, exist_ok=True)
        flowers_dir = os.path.join(output_dir, "flowers")
        not_flowers_dir = os.path.join(output_dir, "not_flowers")
        os.makedirs(flowers_dir, exist_ok=True)
        os.makedirs(not_flowers_dir, exist_ok=True)

        # Get starting indices to avoid overwriting
        flower_idx = self._get_next_index(flowers_dir, "flower")
        not_flower_idx = self._get_next_index(not_flowers_dir, "not_flower")

        # Save verified flowers
        for i, flower in enumerate(self.verified_flowers):
            x, y, w, h = flower['bbox']
            crop = image[y:y+h, x:x+w]
            filename = os.path.join(flowers_dir, f"flower_{flower_idx + i:04d}.jpg")
            cv2.imwrite(filename, crop)

        # Save false positives
        for i, fp in enumerate(self.false_positives):
            x, y, w, h = fp['bbox']
            crop = image[y:y+h, x:x+w]
            filename = os.path.join(not_flowers_dir, f"not_flower_{not_flower_idx + i:04d}.jpg")
            cv2.imwrite(filename, crop)

        total_flowers = flower_idx + len(self.verified_flowers)
        total_not_flowers = not_flower_idx + len(self.false_positives)
        print(f"Saved {len(self.verified_flowers)} flower samples and {len(self.false_positives)} non-flower samples")
        print(f"Total training data: {total_flowers} flowers, {total_not_flowers} not_flowers")

# Utility functions for integration with panning video counter
def integrate_with_panning_counter():
    """
    Example of how to integrate template matching with the panning video counter
    """
    
    class TemplateBasedPanningCounter:
        def __init__(self, zinnia_detector, overlap_threshold=0.3):
            self.detector = zinnia_detector
            self.overlap_threshold = overlap_threshold
            self.tracked_flowers = []
            
        def detect_flowers_in_frame_template(self, frame, threshold=0.6):
            """Replace the color-based detection with template matching"""
            detections = self.detector.match_templates(frame, threshold=threshold)
            
            flowers = []
            for detection in detections:
                x, y, w, h = detection['bbox']
                flowers.append({
                    'bbox': (x, y, w, h),
                    'center': detection['center'],
                    'area': w * h,
                    'confidence': detection['confidence'],
                    'template_id': detection['template_id']
                })
            
            return flowers
    
    return TemplateBasedPanningCounter

# Example usage workflow
if __name__ == "__main__":
    detector = ImprovedZinniaDetector()
    
    # Step 1: Load existing templates (CHOOSE ONE METHOD)
    print("=== Step 1: Load Templates ===")
    
    # Method A: Load all templates matching a pattern
    num_loaded = detector.load_templates(template_dir="templates", template_pattern="zinnia_template_*.jpg")
    
    # Method B: Load specific template files
    # template_files = ["zinnia_template_0.jpg", "zinnia_template_1.jpg", "zinnia_template_2.jpg"]
    # num_loaded = detector.load_templates_from_list(template_files)
    
    # Method C: Load from a different directory
    # num_loaded = detector.load_templates(template_dir="templates", template_pattern="*.jpg")
    
    if num_loaded == 0:
        print("No templates found! Creating new templates...")
        # Create templates from a sample frame
        sample_frame_path = "path/to/sample_frame.jpg"
        templates = detector.create_template_from_selection(sample_frame_path)
        if not templates:
            print("No templates created. Exiting.")
            exit()
    
    # Step 2: Test detection on a frame
    print("\n=== Step 2: Test Detection ===")
    test_frame_path = "selected_frames/frame_000234.jpg"
    
    # Run detection and verification with loaded templates
    results = detector.analyze_frame_with_verification(test_frame_path, template_threshold=0.6)
    
    if results:
        print(f"\nResults:")
        print(f"Total detections: {results['total_detected']}")
        print(f"Verified flowers: {len(results['verified'])}")
        print(f"False positives: {len(results['rejected'])}")
        print(f"Accuracy: {results['accuracy']:.2%}")
        
        # Step 3: Save training data for future improvements
        test_image = cv2.imread(test_frame_path)
        if test_image is not None:
            detector.save_training_data(test_image)
    
    print("\nWorkflow complete!")