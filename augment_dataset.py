#!/usr/bin/env python3
"""
Augment the annotated dataset to create more training images
"""

import cv2
import numpy as np
import os
import glob

def augment_image_and_labels(image, labels, aug_type):
    """
    Augment image and adjust bounding boxes accordingly
    Labels format: list of [class, x_center, y_center, width, height] (normalized)
    """
    h, w = image.shape[:2]
    new_image = image.copy()
    new_labels = [l.copy() for l in labels]

    if aug_type == "hflip":
        new_image = cv2.flip(image, 1)
        for label in new_labels:
            label[1] = 1.0 - label[1]  # flip x_center

    elif aug_type == "brightness_up":
        new_image = cv2.convertScaleAbs(image, alpha=1.3, beta=20)

    elif aug_type == "brightness_down":
        new_image = cv2.convertScaleAbs(image, alpha=0.7, beta=-20)

    elif aug_type == "contrast":
        new_image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

    elif aug_type == "saturation":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.4
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        new_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    elif aug_type == "hue_shift":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] += 15
        hsv[:, :, 0] = hsv[:, :, 0] % 180
        new_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    elif aug_type == "blur":
        new_image = cv2.GaussianBlur(image, (5, 5), 0)

    elif aug_type == "noise":
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        new_image = cv2.add(image, noise)

    elif aug_type == "rotate_5":
        # Small rotation
        M = cv2.getRotationMatrix2D((w/2, h/2), 5, 1)
        new_image = cv2.warpAffine(image, M, (w, h))
        # Note: for small rotations, bbox adjustment is minimal

    elif aug_type == "rotate_neg5":
        M = cv2.getRotationMatrix2D((w/2, h/2), -5, 1)
        new_image = cv2.warpAffine(image, M, (w, h))

    return new_image, new_labels


def augment_dataset(input_dir="yolo_auto", output_dir="yolo_augmented"):
    """Augment all images in the dataset"""

    augmentations = [
        "hflip",
        "brightness_up",
        "brightness_down",
        "contrast",
        "saturation",
        "hue_shift",
        "blur",
        "noise",
        # "rotate_5",      # Disabled - bbox transformation not implemented
        # "rotate_neg5"    # Disabled - bbox transformation not implemented
    ]

    # Create output structure
    for split in ["train", "val"]:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    total_images = 0

    for split in ["train", "val"]:
        image_dir = f"{input_dir}/images/{split}"
        label_dir = f"{input_dir}/labels/{split}"

        image_files = glob.glob(f"{image_dir}/*.jpg")

        for img_path in image_files:
            image = cv2.imread(img_path)
            if image is None:
                continue

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = f"{label_dir}/{base_name}.txt"

            # Load labels
            labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            labels.append([float(p) for p in parts])

            # Copy original
            cv2.imwrite(f"{output_dir}/images/{split}/{base_name}.jpg", image)
            with open(f"{output_dir}/labels/{split}/{base_name}.txt", 'w') as f:
                for label in labels:
                    f.write(f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
            total_images += 1

            # Create augmented versions
            for aug in augmentations:
                aug_image, aug_labels = augment_image_and_labels(image, labels, aug)

                aug_name = f"{base_name}_{aug}"
                cv2.imwrite(f"{output_dir}/images/{split}/{aug_name}.jpg", aug_image)

                with open(f"{output_dir}/labels/{split}/{aug_name}.txt", 'w') as f:
                    for label in aug_labels:
                        f.write(f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

                total_images += 1

            print(f"Augmented {base_name}: 1 original + {len(augmentations)} augmented")

    # Create dataset.yaml
    yaml_content = f"""# Augmented Zinnia Dataset
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: zinnia
"""
    with open(f"{output_dir}/dataset.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\n{'='*50}")
    print(f"Augmentation complete!")
    print(f"Total images: {total_images}")
    print(f"Output: {output_dir}/")
    print(f"Config: {output_dir}/dataset.yaml")
    print(f"{'='*50}")

    return output_dir


if __name__ == "__main__":
    augment_dataset()
