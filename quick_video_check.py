#!/usr/bin/env python3
"""
Quick script to identify videos causing 'moov atom not found' errors
"""

import cv2
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_video_openability(video_path):
    """Test if a video can be opened without errors"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "Cannot open video"
        
        # Try to read one frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False, "Cannot read frames"
        
        return True, "OK"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def check_dataset_for_corrupted_videos():
    """Check dataset for corrupted videos"""
    
    dataset_path = Path("G:/Deefake_detection_app/dataset")
    real_path = dataset_path / "original_sequences" / "youtube" / "c23" / "videos"
    fake_path = dataset_path / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    
    corrupted_videos = []
    
    # Check real videos
    if real_path.exists():
        real_videos = list(real_path.glob("*.mp4"))
        logger.info(f"Checking {len(real_videos)} real videos...")
        
        for video in real_videos[:10]:  # Check first 10 for quick test
            is_ok, error = test_video_openability(video)
            if not is_ok:
                corrupted_videos.append(('real', video.name, error))
                logger.warning(f"CORRUPTED REAL: {video.name} - {error}")
    
    # Check fake videos
    if fake_path.exists():
        fake_videos = list(fake_path.glob("*.mp4"))
        logger.info(f"Checking {len(fake_videos)} fake videos...")
        
        for video in fake_videos[:10]:  # Check first 10 for quick test
            is_ok, error = test_video_openability(video)
            if not is_ok:
                corrupted_videos.append(('fake', video.name, error))
                logger.warning(f"CORRUPTED FAKE: {video.name} - {error}")
    
    logger.info(f"Found {len(corrupted_videos)} corrupted videos in sample")
    return corrupted_videos

if __name__ == "__main__":
    logger.info("Quick check for corrupted videos...")
    corrupted = check_dataset_for_corrupted_videos()
    
    if corrupted:
        logger.info("CORRUPTED VIDEOS FOUND:")
        for video_type, name, error in corrupted:
            logger.info(f"  {video_type}: {name} - {error}")
    else:
        logger.info("No corrupted videos found in sample")
