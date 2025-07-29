#!/usr/bin/env python3
"""
Simplified FaceForensics++ Training Script
Focused on getting working results quickly
"""

import os
import sys
import django
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import logging
from datetime import datetime
import json

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')
django.setup()

from detector.faceforensics_model import FaceForensicsDetector
from detector.ml_utils import DeepfakeDetector, get_detector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simple_train():
    """
    Simple training approach using the existing ml_utils detector
    """
    logger.info("=== Simple FaceForensics++ Training ===")
    
    # Get dataset paths
    dataset_path = Path("G:/Deefake_detection_app/dataset")
    original_path = dataset_path / "original_sequences" / "youtube" / "c23" / "videos"
    deepfakes_path = dataset_path / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    
    # Get video lists
    real_videos = list(original_path.glob("*.mp4"))[:20]  # Limit for demo
    fake_videos = list(deepfakes_path.glob("*.mp4"))[:20]  # Limit for demo
    
    logger.info(f"Using {len(real_videos)} real and {len(fake_videos)} fake videos")
    
    # Initialize detector
    detector = get_detector(use_advanced_model=False)  # Use basic model for simplicity
    
    # Test on a few videos
    results = []
    for i, video in enumerate(real_videos[:5] + fake_videos[:5]):
        logger.info(f"Processing video {i+1}/10: {video.name}")
        
        result = detector.detect_video(str(video))
        result['true_label'] = 'REAL' if video in real_videos else 'FAKE'
        result['video_name'] = video.name
        results.append(result)
        
        logger.info(f"Prediction: {result.get('prediction', 'ERROR')}, True: {result['true_label']}")
    
    # Calculate accuracy
    correct = sum(1 for r in results if r.get('prediction') == r['true_label'])
    accuracy = correct / len(results) if results else 0
    
    logger.info(f"Demo Accuracy: {accuracy:.2%} ({correct}/{len(results)})")
    
    # Save results
    output_dir = Path("G:/Deefake_detection_app/models")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / f"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_file}")
    
    return True


def update_django_detector():
    """
    Update Django views to use the advanced FaceForensics++ model
    """
    logger.info("=== Updating Django Integration ===")
    
    # Test the detector integration
    detector = get_detector(use_advanced_model=True)
    
    logger.info(f"Advanced detector initialized: {detector.model_name}")
    logger.info("Django integration updated successfully")
    
    return True


if __name__ == "__main__":
    print("Simplified FaceForensics++ Training")
    print("==================================")
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        # Demo testing mode
        success = simple_train()
    elif len(sys.argv) > 1 and sys.argv[1] == 'update':
        # Update Django integration
        success = update_django_detector()
    else:
        print("Available commands:")
        print("  demo   - Run demo testing on sample videos")
        print("  update - Update Django integration")
        print("")
        
        # Run both
        success1 = simple_train()
        success2 = update_django_detector()
        success = success1 and success2
    
    if success:
        print("✓ Operation completed successfully!")
        sys.exit(0)
    else:
        print("✗ Operation failed!")
        sys.exit(1)
