#!/usr/bin/env python3
"""
Debug script to simulate the exact training loop that failed
"""

import sys
import logging
import torch
import torch.nn as nn
from pathlib import Path
from detector.faceforensics_model import FaceForensicsDetector, FaceForensicsDataset, FaceForensicsTrainer
from torch.utils.data import DataLoader, random_split

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_training():
    """Simulate the exact training loop to find the error"""
    
    # Get some sample videos
    dataset_path = Path("G:/Deefake_detection_app/dataset")
    real_path = dataset_path / "original_sequences" / "youtube" / "c23" / "videos"
    fake_path = dataset_path / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    
    real_videos = list(real_path.glob("*.mp4"))[:3]  # Just 3 videos
    fake_videos = list(fake_path.glob("*.mp4"))[:3]  # Just 3 videos
    
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
    
    # Split dataset exactly like in the training code
    train_split = 0.7
    val_split = 0.2
    
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    logger.info(f"Dataset splits: train={train_size}, val={val_size}, test={test_size}")
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    batch_size = 2
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Setup training exactly like in the code
    optimizer = torch.optim.Adam(detector.model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    aux_criterion = nn.CrossEntropyLoss()
    
    trainer = FaceForensicsTrainer(detector.model, 'cpu')
    
    logger.info("Starting simulated training...")
    
    try:
        # Try one epoch
        logger.info("Testing train_epoch...")
        train_loss, train_acc, train_aux_loss = trainer.train_epoch(
            train_loader, optimizer, criterion, aux_criterion
        )
        logger.info(f"Train loss: {train_loss}, Train acc: {train_acc}, Aux loss: {train_aux_loss}")
        
        logger.info("Testing validate...")
        val_loss, val_acc = trainer.validate(val_loader, criterion, aux_criterion)
        logger.info(f"Val loss: {val_loss}, Val acc: {val_acc}")
        
        logger.info("✓ Simulation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during simulated training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simulate_training()
