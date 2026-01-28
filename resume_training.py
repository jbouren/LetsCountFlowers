#!/usr/bin/env python3
"""Resume YOLO training from last checkpoint"""

from ultralytics import YOLO
import os

# Find the most recent last.pt
checkpoint = "runs/detect/runs/zinnia/train/weights/last.pt"

if os.path.exists(checkpoint):
    print(f"Resuming from: {checkpoint}")
    model = YOLO(checkpoint)
    model.train(resume=True)
else:
    print(f"No checkpoint found at {checkpoint}")
    print("Run train_yolo.py first to start training")
