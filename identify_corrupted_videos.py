#!/usr/bin/env python3
"""
Identify corrupted or problematic videos in the dataset
"""

import cv2
import os
import logging
from pathlib import Path
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_video_file(video_path, timeout=5):
    """
    Check if a video file is readable and not corrupted
    Returns: (is_valid, error_message, duration)
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return False, "Cannot open video file", 0
        
        # Check basic properties
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if frame_count <= 0 or fps <= 0:
            cap.release()
            return False, "Invalid frame count or FPS", 0
        
        duration = frame_count / fps if fps > 0 else 0
        
        # Try to read first few frames with timeout
        start_time = time.time()
        frames_read = 0
        max_frames_to_test = 5
        
        while frames_read < max_frames_to_test:
            if time.time() - start_time > timeout:
                cap.release()
                return False, f"Timeout reading frames (>{timeout}s)", duration
            
            ret, frame = cap.read()
            if not ret:
                if frames_read == 0:
                    cap.release()
                    return False, "Cannot read any frames", duration
                else:
                    break  # End of video, but we read some frames
            
            frames_read += 1
        
        cap.release()
        
        # If we got here, video seems OK
        if duration > 3600:  # More than 1 hour
            return False, f"Video too long ({duration:.1f}s)", duration
        
        return True, "OK", duration
        
    except Exception as e:
        return False, f"Exception: {str(e)}", 0

def scan_dataset(dataset_path):
    """Scan the entire dataset for corrupted videos"""
    
    dataset_path = Path(dataset_path)
    
    # Define paths
    real_path = dataset_path / "original_sequences" / "youtube" / "c23" / "videos"
    fake_path = dataset_path / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    
    corrupted_videos = []
    valid_videos = []
    
    # Check real videos
    logger.info("Checking real videos...")
    if real_path.exists():
        real_videos = list(real_path.glob("*.mp4"))
        logger.info(f"Found {len(real_videos)} real videos")
        
        for i, video_path in enumerate(real_videos):
            logger.info(f"Checking real video {i+1}/{len(real_videos)}: {video_path.name}")
            is_valid, error_msg, duration = check_video_file(video_path)
            
            if is_valid:
                valid_videos.append(('real', video_path, duration))
                logger.info(f"  ✓ Valid ({duration:.1f}s)")
            else:
                corrupted_videos.append(('real', video_path, error_msg))
                logger.warning(f"  ✗ Corrupted: {error_msg}")
    else:
        logger.warning(f"Real video path not found: {real_path}")
    
    # Check fake videos
    logger.info("\nChecking fake videos...")
    if fake_path.exists():
        fake_videos = list(fake_path.glob("*.mp4"))
        logger.info(f"Found {len(fake_videos)} fake videos")
        
        for i, video_path in enumerate(fake_videos):
            logger.info(f"Checking fake video {i+1}/{len(fake_videos)}: {video_path.name}")
            is_valid, error_msg, duration = check_video_file(video_path)
            
            if is_valid:
                valid_videos.append(('fake', video_path, duration))
                logger.info(f"  ✓ Valid ({duration:.1f}s)")
            else:
                corrupted_videos.append(('fake', video_path, error_msg))
                logger.warning(f"  ✗ Corrupted: {error_msg}")
    else:
        logger.warning(f"Fake video path not found: {fake_path}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SCAN SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total valid videos: {len(valid_videos)}")
    logger.info(f"Total corrupted videos: {len(corrupted_videos)}")
    
    if corrupted_videos:
        logger.info(f"\nCORRUPTED VIDEOS ({len(corrupted_videos)}):")
        for video_type, video_path, error_msg in corrupted_videos:
            logger.info(f"  [{video_type}] {video_path.name}: {error_msg}")
    
    # Create corrupted videos list file
    corrupted_list_file = dataset_path.parent / "corrupted_videos.txt"
    with open(corrupted_list_file, 'w') as f:
        f.write("# Corrupted videos found in dataset\n")
        f.write(f"# Scan date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for video_type, video_path, error_msg in corrupted_videos:
            f.write(f"{video_type},{video_path},{error_msg}\n")
    
    logger.info(f"\nCorrupted videos list saved to: {corrupted_list_file}")
    
    # Create valid videos list
    valid_list_file = dataset_path.parent / "valid_videos.txt"
    with open(valid_list_file, 'w') as f:
        f.write("# Valid videos in dataset\n")
        f.write(f"# Scan date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for video_type, video_path, duration in valid_videos:
            f.write(f"{video_type},{video_path},{duration:.1f}\n")
    
    logger.info(f"Valid videos list saved to: {valid_list_file}")
    
    return valid_videos, corrupted_videos

if __name__ == "__main__":
    dataset_path = "G:/Deefake_detection_app/dataset"
    logger.info(f"Scanning dataset: {dataset_path}")
    
    valid_videos, corrupted_videos = scan_dataset(dataset_path)
    
    logger.info(f"\nScan complete!")
    logger.info(f"Valid: {len(valid_videos)}, Corrupted: {len(corrupted_videos)}")
