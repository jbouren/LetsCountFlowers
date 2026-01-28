#!/usr/bin/env python3
"""Train YOLO (resume if exists) and send iMessage when done"""

import subprocess
import os

def send_imessage(phone_number, message):
    """Send iMessage via AppleScript"""
    script = f'''
    tell application "Messages"
        set targetService to 1st account whose service type = iMessage
        set targetBuddy to participant "{phone_number}" of targetService
        send "{message}" to targetBuddy
    end tell
    '''
    subprocess.run(["osascript", "-e", script])

def main():
    # Your phone number or email for iMessage
    MY_NUMBER = os.environ.get("MY_PHONE", "YOUR_PHONE_OR_EMAIL")
    
    from ultralytics import YOLO
    
    checkpoint = "runs/detect/runs/zinnia/train/weights/last.pt"
    
    try:
        if os.path.exists(checkpoint):
            print(f"Resuming from checkpoint: {checkpoint}")
            model = YOLO(checkpoint)
            results = model.train(resume=True)
        else:
            print("Starting fresh training...")
            model = YOLO("yolov8n.pt")
            results = model.train(
                data="yolo_augmented/dataset.yaml",
                epochs=50,
                imgsz=640,
                batch=8,
                device="mps",
                project="runs/zinnia",
                name="train",
                exist_ok=True,
                patience=10,
            )
        
        # Get final metrics
        metrics_file = "runs/detect/runs/zinnia/train/results.csv"
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                lines = f.readlines()
                if lines:
                    last = lines[-1].split(",")
                    mAP50 = float(last[6]) if len(last) > 6 else 0
                    msg = f"🌸 YOLO done! mAP50: {mAP50:.1%}"
        else:
            msg = "🌸 YOLO training complete!"
            
    except Exception as e:
        msg = f"❌ Training failed: {str(e)[:50]}"
    
    print(msg)
    
    if MY_NUMBER != "YOUR_PHONE_OR_EMAIL":
        send_imessage(MY_NUMBER, msg)
        print(f"Notification sent to {MY_NUMBER}")
    else:
        print("\nTo get iMessage notifications, run with:")
        print("  MY_PHONE='+1234567890' python3 train_with_notify.py")

if __name__ == "__main__":
    main()
