#!/usr/bin/env python3
"""
Quick fix for moov atom errors - identify and move problematic videos
"""

import cv2
import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_quarantine_folder():
    """Create a folder to move corrupted videos"""
    quarantine_path = Path("G:/Deefake_detection_app/quarantine_videos")
    quarantine_path.mkdir(exist_ok=True)
    return quarantine_path

def test_and_quarantine_video(video_path, quarantine_path):
    """Test video and move to quarantine if corrupted"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        # Test 1: Can we open the video?
        if not cap.isOpened():
            logger.warning(f"QUARANTINE: Cannot open {video_path.name}")
            shutil.move(str(video_path), str(quarantine_path / video_path.name))
            return True
        
        # Test 2: Can we read basic properties?
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if frame_count <= 0 or fps <= 0:
            cap.release()
            logger.warning(f"QUARANTINE: Invalid properties {video_path.name}")
            shutil.move(str(video_path), str(quarantine_path / video_path.name))
            return True
        
        # Test 3: Can we read the first frame?
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            logger.warning(f"QUARANTINE: Cannot read frames {video_path.name}")
            shutil.move(str(video_path), str(quarantine_path / video_path.name))
            return True
        
        # Video seems OK
        return False
        
    except Exception as e:
        logger.warning(f"QUARANTINE: Exception with {video_path.name}: {e}")
        try:
            shutil.move(str(video_path), str(quarantine_path / video_path.name))
            return True
        except:
            logger.error(f"Could not move {video_path.name}")
            return False

def clean_dataset():
    """Remove corrupted videos from dataset"""
    
    dataset_path = Path("G:/Deefake_detection_app/dataset")
    real_path = dataset_path / "original_sequences" / "youtube" / "c23" / "videos"
    fake_path = dataset_path / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    
    quarantine_path = create_quarantine_folder()
    
    corrupted_count = 0
    total_checked = 0
    
    # Check real videos
    if real_path.exists():
        real_videos = list(real_path.glob("*.mp4"))
        logger.info(f"Checking {len(real_videos)} real videos...")
        
        for video in real_videos:
            total_checked += 1
            if test_and_quarantine_video(video, quarantine_path / "real"):
                corrupted_count += 1
                logger.info(f"Moved corrupted real video: {video.name}")
    
    # Check fake videos
    if fake_path.exists():
        fake_videos = list(fake_path.glob("*.mp4"))
        logger.info(f"Checking {len(fake_videos)} fake videos...")
        
        for video in fake_videos:
            total_checked += 1
            if test_and_quarantine_video(video, quarantine_path / "fake"):
                corrupted_count += 1
                logger.info(f"Moved corrupted fake video: {video.name}")
    
    logger.info(f"Dataset cleaning complete!")
    logger.info(f"Total videos checked: {total_checked}")
    logger.info(f"Corrupted videos moved: {corrupted_count}")
    logger.info(f"Clean videos remaining: {total_checked - corrupted_count}")
    logger.info(f"Quarantined videos location: {quarantine_path}")

if __name__ == "__main__":
    logger.info("Starting dataset cleanup to remove corrupted videos...")
    
    # Create quarantine folders
    quarantine_base = create_quarantine_folder()
    (quarantine_base / "real").mkdir(exist_ok=True)
    (quarantine_base / "fake").mkdir(exist_ok=True)
    
    clean_dataset()
    
    logger.info("You can now restart training without moov atom errors!")
