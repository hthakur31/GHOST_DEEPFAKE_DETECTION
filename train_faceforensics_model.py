#!/usr/bin/env python3
"""
Training script for FaceForensics++ deepfake detection model
This script trains the advanced FaceForensics++ model on the downloaded dataset
"""

import os
import sys
import django
from pathlib import Path
import torch
import logging
from datetime import datetime

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')
django.setup()

from detector.faceforensics_model import FaceForensicsDetector, FaceForensicsDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_model():
    """
    Train the FaceForensics++ deepfake detection model
    """
    try:
        # Configuration
        config = {
            'data_path': 'G:/FaceForensics++',  # Path to downloaded FaceForensics++ dataset
            'output_dir': 'G:/Deefake_detection_app/models',  # Where to save trained models
            'batch_size': 16,
            'num_epochs': 50,
            'learning_rate': 0.001,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_workers': 4,
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        }
        
        logger.info("=== FaceForensics++ Model Training ===")
        logger.info(f"Configuration: {config}")
        
        # Create output directory
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset exists
        data_path = Path(config['data_path'])
        if not data_path.exists():
            logger.error(f"Dataset path does not exist: {data_path}")
            logger.info("Please make sure you have downloaded the FaceForensics++ dataset first.")
            logger.info("You can use the download-FaceForensics.py script to download it.")
            return False
        
        # Initialize detector (this will create the model architecture)
        logger.info("Initializing FaceForensics++ detector...")
        detector = FaceForensicsDetector(
            model_path=None,  # Start with untrained model
            device=config['device']
        )
        
        # Check if training data is available
        real_videos_path = data_path / 'original_sequences' / 'youtube'
        fake_videos_path = data_path / 'manipulated_sequences'
        
        if not real_videos_path.exists():
            logger.error(f"Real videos path not found: {real_videos_path}")
            return False
        
        if not fake_videos_path.exists():
            logger.error(f"Fake videos path not found: {fake_videos_path}")
            return False
        
        # Count available videos
        real_videos = list(real_videos_path.glob('**/*.mp4'))
        fake_methods = ['Deepfakes', 'Face2Face', 'FaceSwapper', 'NeuralTextures']
        fake_videos = []
        
        for method in fake_methods:
            method_path = fake_videos_path / method
            if method_path.exists():
                fake_videos.extend(list(method_path.glob('**/*.mp4')))
        
        logger.info(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos")
        
        if len(real_videos) == 0 or len(fake_videos) == 0:
            logger.error("No training videos found. Please check the dataset structure.")
            return False
        
        # Create dataset
        logger.info("Creating training dataset...")
        dataset = FaceForensicsDataset(
            real_videos=real_videos[:min(1000, len(real_videos))],  # Limit for demo
            fake_videos=fake_videos[:min(1000, len(fake_videos))],  # Limit for demo
            transform=detector.get_transforms()
        )
        
        logger.info(f"Dataset created with {len(dataset)} samples")
        
        # Train the model
        logger.info("Starting training...")
        training_stats = detector.train(
            dataset=dataset,
            batch_size=config['batch_size'],
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            num_workers=config['num_workers'],
            train_split=config['train_split'],
            val_split=config['val_split']
        )
        
        # Save the trained model
        model_save_path = output_dir / f"faceforensics_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        detector.save_model(str(model_save_path))
        
        logger.info(f"Model saved to: {model_save_path}")
        logger.info("Training completed successfully!")
        logger.info(f"Final training stats: {training_stats}")
        
        # Save training configuration and stats
        import json
        stats_path = output_dir / f"training_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_path, 'w') as f:
            json.dump({
                'config': config,
                'training_stats': training_stats,
                'model_path': str(model_save_path),
                'dataset_info': {
                    'real_videos': len(real_videos),
                    'fake_videos': len(fake_videos),
                    'total_samples': len(dataset)
                }
            }, f, indent=2)
        
        logger.info(f"Training stats saved to: {stats_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_test():
    """
    Quick test to verify the model architecture works
    """
    try:
        logger.info("=== Quick Model Test ===")
        
        # Test model initialization
        detector = FaceForensicsDetector(
            model_path=None,
            device='cpu'  # Use CPU for quick test
        )
        
        logger.info("✓ Model initialization successful")
        
        # Test model forward pass with dummy data
        import torch
        dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, RGB image 224x224
        
        with torch.no_grad():
            output = detector.model(dummy_input)
            if isinstance(output, tuple):
                main_output, aux_output = output
                logger.info(f"✓ Model forward pass successful, main output shape: {main_output.shape}, aux output shape: {aux_output.shape}")
            else:
                logger.info(f"✓ Model forward pass successful, output shape: {output.shape}")
        
        # Test data loading (if dataset exists)
        data_path = Path('G:/FaceForensics++')
        if data_path.exists():
            logger.info("✓ FaceForensics++ dataset found")
        else:
            logger.info("⚠ FaceForensics++ dataset not found (this is OK for architecture test)")
        
        logger.info("Quick test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during quick test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("FaceForensics++ Model Training Script")
    print("=====================================")
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Quick test mode
        success = quick_test()
    else:
        # Full training mode
        print("Starting full training...")
        print("Use 'python train_faceforensics_model.py test' for quick architecture test")
        print()
        
        success = train_model()
    
    if success:
        print("✓ Operation completed successfully!")
        sys.exit(0)
    else:
        print("✗ Operation failed!")
        sys.exit(1)
