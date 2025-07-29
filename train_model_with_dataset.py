#!/usr/bin/env python3
"""
FaceForensics++ Model Training Script
Analyzes the actual dataset and trains the deepfake detection model
"""

import os
import sys
import django
from pathlib import Path
import torch
import logging
from datetime import datetime
import json
from collections import defaultdict

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


def analyze_dataset():
    """
    Analyze the FaceForensics++ dataset structure and count available videos
    """
    logger.info("=== Dataset Analysis ===")
    
    dataset_path = Path("G:/Deefake_detection_app/dataset")
    
    # Check if dataset exists
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return None
    
    analysis = {
        'dataset_path': str(dataset_path),
        'original_videos': [],
        'manipulated_videos': defaultdict(list),
        'total_real': 0,
        'total_fake': 0,
        'compression_level': 'c23'
    }
    
    # Analyze original videos
    original_path = dataset_path / "original_sequences" / "youtube" / "c23" / "videos"
    if original_path.exists():
        original_videos = list(original_path.glob("*.mp4"))
        analysis['original_videos'] = [str(v) for v in original_videos]
        analysis['total_real'] = len(original_videos)
        logger.info(f"Found {len(original_videos)} original videos")
    else:
        logger.warning(f"Original videos path not found: {original_path}")
    
    # Analyze manipulated videos by method
    manipulated_base = dataset_path / "manipulated_sequences"
    if manipulated_base.exists():
        for method_dir in manipulated_base.iterdir():
            if method_dir.is_dir():
                method_name = method_dir.name
                videos_path = method_dir / "c23" / "videos"
                
                if videos_path.exists():
                    fake_videos = list(videos_path.glob("*.mp4"))
                    analysis['manipulated_videos'][method_name] = [str(v) for v in fake_videos]
                    analysis['total_fake'] += len(fake_videos)
                    logger.info(f"Found {len(fake_videos)} videos for method: {method_name}")
                else:
                    logger.warning(f"Videos path not found for method {method_name}: {videos_path}")
    
    logger.info(f"Dataset Analysis Summary:")
    logger.info(f"  - Total Real Videos: {analysis['total_real']}")
    logger.info(f"  - Total Fake Videos: {analysis['total_fake']}")
    logger.info(f"  - Fake Video Methods: {list(analysis['manipulated_videos'].keys())}")
    
    # Save analysis to file
    analysis_file = dataset_path / "dataset_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Dataset analysis saved to: {analysis_file}")
    
    return analysis


def prepare_training_data(analysis):
    """
    Prepare training data lists from the dataset analysis
    """
    if not analysis:
        return None, None
    
    real_videos = analysis['original_videos']
    fake_videos = []
    
    # Combine all fake videos from different methods
    for method, videos in analysis['manipulated_videos'].items():
        fake_videos.extend(videos)
    
    logger.info(f"Prepared training data:")
    logger.info(f"  - Real videos: {len(real_videos)}")
    logger.info(f"  - Fake videos: {len(fake_videos)}")
    
    return real_videos, fake_videos


def train_model(real_videos, fake_videos):
    """
    Train the FaceForensics++ deepfake detection model
    """
    try:
        # Configuration
        config = {
            'batch_size': 8,  # Reduced for limited dataset
            'num_epochs': 20,  # Reduced for demo
            'learning_rate': 0.001,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_workers': 2,  # Reduced for stability
            'train_split': 0.7,
            'val_split': 0.2,
            'test_split': 0.1,
            'max_videos_per_class': 40  # Limit for demo training
        }
        
        logger.info("=== FaceForensics++ Model Training ===")
        logger.info(f"Configuration: {config}")
        logger.info(f"Device: {config['device']}")
        
        # Create output directory
        output_dir = Path("G:/Deefake_detection_app/models")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Limit dataset size for demo
        if len(real_videos) > config['max_videos_per_class']:
            real_videos = real_videos[:config['max_videos_per_class']]
        if len(fake_videos) > config['max_videos_per_class']:
            fake_videos = fake_videos[:config['max_videos_per_class']]
        
        logger.info(f"Using {len(real_videos)} real and {len(fake_videos)} fake videos for training")
        
        # Initialize detector
        logger.info("Initializing FaceForensics++ detector...")
        detector = FaceForensicsDetector(
            model_path=None,  # Start with untrained model
            device=config['device']
        )
        
        # Create dataset
        logger.info("Creating training dataset...")
        train_transform, val_transform = detector.get_transforms()
        dataset = FaceForensicsDataset(
            real_videos=real_videos,
            fake_videos=fake_videos,
            transform=val_transform  # Use validation transform for consistency
        )
        
        logger.info(f"Dataset created with {len(dataset)} samples")
        
        if len(dataset) == 0:
            logger.error("No valid samples in dataset. Check video paths and face detection.")
            return False
        
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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_save_path = output_dir / f"faceforensics_model_{timestamp}.pth"
        detector.save_model(str(model_save_path))
        
        logger.info(f"Model saved to: {model_save_path}")
        logger.info("Training completed successfully!")
        logger.info(f"Final training stats: {training_stats}")
        
        # Save training configuration and stats
        stats_path = output_dir / f"training_stats_{timestamp}.json"
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
    Quick test to verify the model architecture and dataset
    """
    try:
        logger.info("=== Quick Test ===")
        
        # Test dataset analysis
        analysis = analyze_dataset()
        if not analysis:
            logger.error("Dataset analysis failed")
            return False
        
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
                logger.info(f"✓ Model forward pass successful")
                logger.info(f"  Main output shape: {main_output.shape}")
                logger.info(f"  Auxiliary output shape: {aux_output.shape}")
            else:
                logger.info(f"✓ Model forward pass successful, output shape: {output.shape}")
        
        logger.info("✓ Quick test completed successfully!")
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
    elif len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        # Analysis only mode
        analysis = analyze_dataset()
        success = analysis is not None
    else:
        # Full training mode
        print("Starting full training...")
        print("Use 'python train_model_with_dataset.py test' for quick test")
        print("Use 'python train_model_with_dataset.py analyze' for dataset analysis only")
        print()
        
        # Analyze dataset
        analysis = analyze_dataset()
        if not analysis:
            print("✗ Dataset analysis failed!")
            sys.exit(1)
        
        # Prepare training data
        real_videos, fake_videos = prepare_training_data(analysis)
        if not real_videos or not fake_videos:
            print("✗ No training data available!")
            sys.exit(1)
        
        # Train model
        success = train_model(real_videos, fake_videos)
    
    if success:
        print("✓ Operation completed successfully!")
        sys.exit(0)
    else:
        print("✗ Operation failed!")
        sys.exit(1)
