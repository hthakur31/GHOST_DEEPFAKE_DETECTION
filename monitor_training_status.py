#!/usr/bin/env python3
"""
Monitor XceptionNet training status and check for errors.
"""

import os
import time
import glob
from pathlib import Path

def check_training_status():
    """Check if training is running and monitor progress."""
    
    # Check for recent log files
    log_pattern = "training_*.log"
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        print("No training log files found.")
        return
    
    # Get the most recent log file
    latest_log = max(log_files, key=os.path.getctime)
    print(f"Latest log file: {latest_log}")
    
    # Check if log file has content
    try:
        with open(latest_log, 'r') as f:
            content = f.read()
            if content:
                lines = content.strip().split('\n')
                print(f"Log file has {len(lines)} lines")
                print("Recent log entries:")
                # Show last 5 lines
                for line in lines[-5:]:
                    print(f"  {line}")
            else:
                print("Log file is empty")
    except Exception as e:
        print(f"Error reading log file: {e}")
    
    # Check for model files
    model_files = glob.glob("xception_model_*.pth")
    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        model_time = os.path.getctime(latest_model)
        print(f"Latest model file: {latest_model}")
        print(f"Model created: {time.ctime(model_time)}")
    else:
        print("No XceptionNet model files found yet.")
    
    # Check dataset folder
    dataset_path = Path("FaceForensics++")
    if dataset_path.exists():
        real_videos = list(dataset_path.glob("original_sequences/youtube/c23/videos/*.mp4"))
        fake_videos = list(dataset_path.glob("manipulated_sequences/Deepfakes/c23/videos/*.mp4"))
        print(f"Dataset: {len(real_videos)} real videos, {len(fake_videos)} fake videos")
    else:
        print("FaceForensics++ dataset folder not found")
    
    # Check for training process (basic check)
    print("\nTraining status check complete.")

if __name__ == "__main__":
    print("=== XceptionNet Training Status Monitor ===")
    check_training_status()
