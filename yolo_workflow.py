#!/usr/bin/env python3
"""
YOLO Active Learning Workflow for Zinnia Detection
Step 1: Run pre-trained model
Step 2: Review outputs
Step 3: Fine-tune on corrections
"""

from ultralytics import YOLO
import cv2
import os
import glob
import shutil
from datetime import datetime

def step1_pretrained_detection():
    """Run pre-trained YOLOv8 to see what it detects"""

    print("="*60)
    print("STEP 1: Running pre-trained YOLOv8")
    print("="*60)

    # Load pre-trained model (trained on COCO - has 'potted plant' class)
    model = YOLO("yolov8n.pt")

    # Create output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"yolo_pretrained_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    frames_dir = "selected_frames"
    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))

    print(f"Processing {len(frame_files)} frames...")
    print(f"Output: {output_dir}/\n")

    total_detections = 0

    for frame_path in frame_files:
        frame_name = os.path.basename(frame_path)

        # Run inference (low confidence to catch more)
        results = model(frame_path, conf=0.1, verbose=False)

        # Count all detections
        count = len(results[0].boxes)
        total_detections += count

        # Save annotated image
        annotated = results[0].plot()
        output_path = os.path.join(output_dir, f"pretrained_{frame_name}")
        cv2.imwrite(output_path, annotated)

        # Print what classes were detected
        if count > 0:
            classes = results[0].boxes.cls.tolist()
            class_names = [model.names[int(c)] for c in classes]
            unique_classes = list(set(class_names))
            print(f"{frame_name}: {count} detections - classes: {unique_classes}")
        else:
            print(f"{frame_name}: 0 detections")

    print(f"\nTotal detections: {total_detections}")
    print(f"Output saved to: {output_dir}/")

    # Open folder
    os.system(f"open {output_dir}")

    return output_dir


def step2_create_dataset_for_finetuning(source_output_dir):
    """
    Create dataset structure for fine-tuning.
    User should review images in source_output_dir and we'll use those labels.
    """
    print("\n" + "="*60)
    print("STEP 2: Preparing dataset for fine-tuning")
    print("="*60)

    dataset_dir = "yolo_finetune_dataset"
    os.makedirs(f"{dataset_dir}/images/train", exist_ok=True)
    os.makedirs(f"{dataset_dir}/images/val", exist_ok=True)
    os.makedirs(f"{dataset_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{dataset_dir}/labels/val", exist_ok=True)

    # Copy frames
    frames_dir = "selected_frames"
    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))

    # 80/20 split
    split_idx = int(len(frame_files) * 0.8)
    train_frames = frame_files[:split_idx]
    val_frames = frame_files[split_idx:]

    for frames, split in [(train_frames, "train"), (val_frames, "val")]:
        for frame_path in frames:
            frame_name = os.path.basename(frame_path)
            shutil.copy(frame_path, f"{dataset_dir}/images/{split}/{frame_name}")

            # Create empty label file (user will annotate)
            base_name = os.path.splitext(frame_name)[0]
            label_path = f"{dataset_dir}/labels/{split}/{base_name}.txt"
            open(label_path, 'a').close()

    # Create dataset config
    yaml_content = f"""# Zinnia Fine-tuning Dataset
path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/val

names:
  0: zinnia
"""

    with open(f"{dataset_dir}/dataset.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Dataset structure created at: {dataset_dir}/")
    print(f"Train images: {len(train_frames)}")
    print(f"Val images: {len(val_frames)}")
    print("\nNext: Annotate images using LabelImg or Roboflow, then run step3")
    print("Or run: step3_train() to train on current labels")

    return dataset_dir


def step3_train(epochs=100):
    """Fine-tune YOLO on annotated dataset"""

    print("\n" + "="*60)
    print("STEP 3: Fine-tuning YOLOv8 on Zinnia dataset")
    print("="*60)

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load pre-trained model
    model = YOLO("yolov8n.pt")

    # Train
    results = model.train(
        data="yolo_finetune_dataset/dataset.yaml",
        epochs=epochs,
        imgsz=640,
        batch=8,
        device=device,
        patience=20,
        project="runs/zinnia_finetune",
        name="train",
        exist_ok=True,
        verbose=True
    )

    print("\n" + "="*60)
    print("Training complete!")
    print("Best model: runs/zinnia_finetune/train/weights/best.pt")
    print("="*60)

    return results


def step4_inference(model_path="runs/zinnia_finetune/train/weights/best.pt", conf=0.25):
    """Run inference with fine-tuned model"""

    print("\n" + "="*60)
    print("STEP 4: Running inference with fine-tuned model")
    print("="*60)

    model = YOLO(model_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"yolo_finetuned_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    frames_dir = "selected_frames"
    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))

    total = 0
    for frame_path in frame_files:
        frame_name = os.path.basename(frame_path)
        results = model(frame_path, conf=conf, verbose=False)
        count = len(results[0].boxes)
        total += count

        annotated = results[0].plot()
        cv2.imwrite(os.path.join(output_dir, f"detected_{frame_name}"), annotated)
        print(f"{frame_name}: {count} zinnias")

    print(f"\nTotal: {total} zinnias")
    print(f"Output: {output_dir}/")
    os.system(f"open {output_dir}")


if __name__ == "__main__":
    print("YOLO Active Learning Workflow")
    print("="*60)
    print("1. step1_pretrained_detection() - See what pre-trained YOLO finds")
    print("2. step2_create_dataset_for_finetuning() - Set up dataset structure")
    print("3. step3_train() - Fine-tune on your annotations")
    print("4. step4_inference() - Run detection with fine-tuned model")
    print("="*60)

    # Run step 1 by default
    step1_pretrained_detection()
