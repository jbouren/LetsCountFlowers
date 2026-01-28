#!/usr/bin/env python3
"""
Train YOLOv8 model on Zinnia dataset using M2 GPU (MPS)
"""

from ultralytics import YOLO
import torch

def train_zinnia_detector():
    # Check for MPS (Apple Silicon GPU)
    if torch.backends.mps.is_available():
        print("✓ MPS (Apple Silicon GPU) available - will use for training")
        device = "mps"
    else:
        print("MPS not available, using CPU")
        device = "cpu"

    # Load YOLOv8 nano model (smallest, fastest to train)
    # Options: yolov8n (nano), yolov8s (small), yolov8m (medium), yolov8l (large)
    model = YOLO("yolov8n.pt")

    # Train the model
    results = model.train(
        data="yolo_augmented/dataset.yaml",
        epochs=50,              # Increase for better accuracy
        imgsz=640,              # Image size
        batch=8,                # Batch size (adjust based on memory)
        device=device,          # Use MPS for M2 GPU
        patience=10,            # Early stopping patience
        save=True,              # Save checkpoints
        project="runs/zinnia",  # Output directory
        name="train",           # Run name
        exist_ok=True,          # Overwrite existing
        pretrained=True,        # Use pretrained weights
        verbose=True
    )

    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best model saved to: runs/zinnia/train/weights/best.pt")
    print("="*50)

    return results

if __name__ == "__main__":
    train_zinnia_detector()
