#!/usr/bin/env python3
"""
Create a clean version of the training script that skips known problematic videos
"""

import cv2
import os
import logging
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_video_check(video_path, max_timeout=3):
    """
    Quick check if video is readable (for filtering before training)
    Returns: True if video seems OK, False if corrupted
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return False
        
        # Check basic properties
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if frame_count <= 0 or fps <= 0:
            cap.release()
            return False
        
        # Quick duration check
        duration = frame_count / fps if fps > 0 else 0
        if duration > 1800:  # Skip videos longer than 30 minutes
            cap.release()
            return False
        
        # Try to read first frame quickly
        start_time = time.time()
        ret, frame = cap.read()
        elapsed = time.time() - start_time
        
        cap.release()
        
        if not ret or elapsed > max_timeout:
            return False
        
        return True
        
    except Exception:
        return False

def filter_valid_videos(video_list, video_type="unknown"):
    """Filter out corrupted videos from a list"""
    valid_videos = []
    corrupted_count = 0
    
    logger.info(f"Filtering {len(video_list)} {video_type} videos...")
    
    for i, video_path in enumerate(video_list):
        if i % 10 == 0:
            logger.info(f"Checking {video_type} video {i+1}/{len(video_list)}")
        
        if quick_video_check(video_path):
            valid_videos.append(video_path)
        else:
            corrupted_count += 1
            logger.debug(f"Skipping corrupted {video_type} video: {video_path.name}")
    
    logger.info(f"{video_type.capitalize()} videos: {len(valid_videos)} valid, {corrupted_count} corrupted/skipped")
    return valid_videos

def create_filtered_dataset_lists():
    """Create lists of valid videos for training"""
    
    dataset_path = Path("G:/Deefake_detection_app/dataset")
    real_path = dataset_path / "original_sequences" / "youtube" / "c23" / "videos"
    fake_path = dataset_path / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    
    logger.info("Creating filtered dataset lists...")
    
    # Get all videos
    real_videos = list(real_path.glob("*.mp4")) if real_path.exists() else []
    fake_videos = list(fake_path.glob("*.mp4")) if fake_path.exists() else []
    
    logger.info(f"Found {len(real_videos)} real videos, {len(fake_videos)} fake videos")
    
    # Filter out corrupted videos
    valid_real_videos = filter_valid_videos(real_videos, "real")
    valid_fake_videos = filter_valid_videos(fake_videos, "fake")
    
    # Save filtered lists
    output_dir = Path("G:/Deefake_detection_app")
    
    # Save valid real videos
    real_list_file = output_dir / "valid_real_videos.txt"
    with open(real_list_file, 'w') as f:
        for video in valid_real_videos:
            f.write(f"{video}\n")
    
    # Save valid fake videos  
    fake_list_file = output_dir / "valid_fake_videos.txt"
    with open(fake_list_file, 'w') as f:
        for video in valid_fake_videos:
            f.write(f"{video}\n")
    
    logger.info(f"Saved {len(valid_real_videos)} valid real videos to: {real_list_file}")
    logger.info(f"Saved {len(valid_fake_videos)} valid fake videos to: {fake_list_file}")
    
    return valid_real_videos, valid_fake_videos

def load_filtered_video_lists():
    """Load previously filtered video lists"""
    
    output_dir = Path("G:/Deefake_detection_app")
    real_list_file = output_dir / "valid_real_videos.txt"
    fake_list_file = output_dir / "valid_fake_videos.txt"
    
    valid_real_videos = []
    valid_fake_videos = []
    
    if real_list_file.exists():
        with open(real_list_file, 'r') as f:
            valid_real_videos = [Path(line.strip()) for line in f if line.strip()]
    
    if fake_list_file.exists():
        with open(fake_list_file, 'r') as f:
            valid_fake_videos = [Path(line.strip()) for line in f if line.strip()]
    
    return valid_real_videos, valid_fake_videos

if __name__ == "__main__":
    print("Creating filtered dataset (removing corrupted videos)...")
    
    # Check if filtered lists already exist
    output_dir = Path("G:/Deefake_detection_app")
    real_list_file = output_dir / "valid_real_videos.txt"
    fake_list_file = output_dir / "valid_fake_videos.txt"
    
    if real_list_file.exists() and fake_list_file.exists():
        logger.info("Loading existing filtered video lists...")
        valid_real_videos, valid_fake_videos = load_filtered_video_lists()
    else:
        logger.info("Creating new filtered video lists...")
        valid_real_videos, valid_fake_videos = create_filtered_dataset_lists()
    
    logger.info(f"Final dataset: {len(valid_real_videos)} real videos, {len(valid_fake_videos)} fake videos")
    logger.info("You can now use these filtered video lists for training to avoid 'moov atom not found' errors.")
