#!/usr/bin/env python3
"""
Debug script to check dataset output format
"""

import sys
import logging
from pathlib import Path
from detector.faceforensics_model import FaceForensicsDetector, FaceForensicsDataset
import torch
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset():
    """Test the dataset output format"""
    
    # Get some sample videos
    dataset_path = Path("G:/Deefake_detection_app/dataset")
    real_path = dataset_path / "original_sequences" / "youtube" / "c23" / "videos"
    fake_path = dataset_path / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    
    real_videos = list(real_path.glob("*.mp4"))[:2]  # Just 2 videos
    fake_videos = list(fake_path.glob("*.mp4"))[:2]  # Just 2 videos
    
    logger.info(f"Testing with {len(real_videos)} real and {len(fake_videos)} fake videos")
    
    # Create dataset
    detector = FaceForensicsDetector(device='cpu')
    train_transform, val_transform = detector.get_transforms()
    
    dataset = FaceForensicsDataset(
        real_videos=real_videos,
        fake_videos=fake_videos,
        transform=val_transform
    )
    
    logger.info(f"Dataset created with {len(dataset)} samples")
    
    # Test single sample
    try:
        sample = dataset[0]
        logger.info(f"Single sample return type: {type(sample)}")
        logger.info(f"Single sample length: {len(sample)}")
        
        if len(sample) == 4:
            face_tensor, freq_tensor, label, aux_label = sample
            logger.info(f"Face tensor shape: {face_tensor.shape}")
            logger.info(f"Freq tensor shape: {freq_tensor.shape}")
            logger.info(f"Label: {label}")
            logger.info(f"Aux label: {aux_label}")
        else:
            logger.info(f"Unexpected sample format: {sample}")
    except Exception as e:
        logger.error(f"Error getting single sample: {e}")
        import traceback
        traceback.print_exc()
    
    # Test dataloader
    try:
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        logger.info("Testing DataLoader...")
        
        for batch_idx, batch_data in enumerate(dataloader):
            logger.info(f"Batch {batch_idx}:")
            logger.info(f"  Batch type: {type(batch_data)}")
            logger.info(f"  Batch length: {len(batch_data)}")
            
            if len(batch_data) == 4:
                face_batch, freq_batch, label_batch, aux_label_batch = batch_data
                logger.info(f"  Face batch shape: {face_batch.shape}")
                logger.info(f"  Freq batch shape: {freq_batch.shape}")
                logger.info(f"  Label batch shape: {label_batch.shape}")
                logger.info(f"  Aux label batch shape: {aux_label_batch.shape}")
            else:
                logger.info(f"  Unexpected batch format: {batch_data}")
            
            break  # Just test first batch
            
    except Exception as e:
        logger.error(f"Error with DataLoader: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
