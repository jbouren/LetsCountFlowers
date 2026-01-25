# LetsCountFlowers

A computer vision project for detecting and counting Zinnia flowers in drone footage using machine learning and image recognition techniques.

## Overview

This project analyzes panning drone video footage of a Zinnia garden to accurately count individual flowers. It addresses the challenge of counting flowers in moving video where the same flower may appear in multiple frames.

## Features

### Panning Video Analysis (`zinniacounter.py`)

The `PanningZinniaCounter` class provides:

- **Optical Flow-Based Camera Tracking**: Detects camera movement between frames using Lucas-Kanade optical flow to track how the scene shifts
- **Color-Based Flower Detection**: Identifies Zinnias using HSV color ranges for red, orange, yellow, and purple flowers
- **Duplicate Detection**: Tracks flowers across frames using bounding box overlap and predicted positions based on camera movement, preventing double-counting
- **Smart Sampling**: Option to process every Nth frame for faster analysis
- **Comprehensive Reporting**: Generates visualizations showing flowers per frame, cumulative counts, camera movement, and tracking confidence

### Template-Based Detection (`ZinniaDetector.py`)

The `ImprovedZinniaDetector` class provides:

- **Template Matching**: Uses pre-captured Zinnia templates to find flowers via multi-scale template matching
- **Interactive Template Creation**: Click-and-drag interface to create new templates from sample images
- **Manual Verification Interface**: Review and verify/reject detections to improve accuracy
- **Non-Maximum Suppression**: Removes overlapping detections, keeping the highest confidence match
- **Training Data Export**: Saves verified flowers and false positives for future model training

## Project Structure

```
LetsCountFlowers/
├── zinniacounter.py          # Main panning video analysis with optical flow
├── ZinniaDetector.py         # Template-based detection with manual verification
├── drone.mov                 # Source drone footage
├── zinnia_template_*.jpg     # Template images of Zinnia flowers (13 templates)
├── selected_frames/          # Key frames extracted for analysis
│   └── frame_*.jpg           # 20 frames selected based on camera movement
├── training_data/            # Training data for model improvement
│   ├── flowers/              # Verified flower samples
│   └── not_flowers/          # Rejected detections (false positives)
├── panning_video_analysis.png # Generated analysis visualization
└── sampleout.jpg             # Sample output image
```

## Dependencies

- OpenCV (`cv2`)
- NumPy
- scikit-learn (DBSCAN clustering)
- Matplotlib

## Usage

### Analyze Drone Video

```python
from zinniacounter import PanningZinniaCounter

counter = PanningZinniaCounter(overlap_threshold=0.3, movement_threshold=30)
total_flowers = counter.process_video_smart("drone.mov", "output.mp4", sample_every_n_frames=2)
counter.generate_report("drone.mov")
```

### Template-Based Detection

```python
from ZinniaDetector import ImprovedZinniaDetector

detector = ImprovedZinniaDetector()
detector.load_templates(template_dir=".", template_pattern="zinnia_template_*.jpg")
results = detector.analyze_frame_with_verification("selected_frames/frame_000234.jpg")
```

## How It Works

1. **Frame Extraction**: Key frames are selected based on camera movement to ensure coverage of the entire garden
2. **Flower Detection**: Flowers are identified using color segmentation (HSV thresholds) or template matching
3. **Tracking**: Optical flow estimates camera movement to predict where previously-seen flowers should appear
4. **Deduplication**: Flowers are matched across frames using position prediction and bounding box overlap
5. **Counting**: Only unique flowers are counted, with confidence scores based on how many frames each flower was tracked

## Output

The analysis generates:
- Annotated video showing detected flowers and running counts
- Analysis plots showing detection statistics over time
- Final count of unique Zinnia flowers in the footage
