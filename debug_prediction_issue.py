#!/usr/bin/env python3
"""
Debug script to identify the prediction inversion issue
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from enhanced_xception_predictor import EnhancedXceptionNetPredictor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_prediction_logic():
    """Test prediction logic with known samples"""
    print("=" * 60)
    print("DEBUGGING PREDICTION ISSUE")
    print("=" * 60)
    
    # Initialize predictor
    predictor = EnhancedXceptionNetPredictor()
    
    if predictor.model is None:
        print("❌ No model loaded!")
        return
    
    print(f"✅ Model loaded: {predictor.model_type}")
    print(f"📱 Device: {predictor.device}")
    
    # Test with dummy data first
    print("\n🧪 Testing with dummy data:")
    dummy_input = torch.randn(1, 3, 224, 224).to(predictor.device)
    
    with torch.no_grad():
        outputs = predictor.model(dummy_input)
        probabilities = torch.softmax(outputs, dim=1)
        
        real_prob = probabilities[0][0].item()
        fake_prob = probabilities[0][1].item()
        
        print(f"Raw output: {outputs}")
        print(f"Probabilities: {probabilities}")
        print(f"Real probability (index 0): {real_prob:.4f}")
        print(f"Fake probability (index 1): {fake_prob:.4f}")
        print(f"Current prediction logic: {'FAKE' if fake_prob > 0.5 else 'REAL'}")
    
    # Test with actual videos if available
    dataset_path = Path("dataset")
    if not dataset_path.exists():
        print("\n❌ Dataset folder not found - skipping video tests")
        return
    
    real_path = dataset_path / "real"
    fake_path = dataset_path / "fake"
    
    if real_path.exists() and fake_path.exists():
        print("\n🎬 Testing with actual videos:")
        
        # Test a few real videos
        real_videos = list(real_path.glob("*.mp4"))[:3]
        fake_videos = list(fake_path.glob("*.mp4"))[:3]
        
        print("\n📹 Testing REAL videos (should predict REAL):")
        for i, video in enumerate(real_videos):
            result = predictor.predict_video(video, max_frames=5)
            if result.get('success'):
                pred = result['prediction']
                conf = result['confidence']
                print(f"  Video {i+1}: {video.name}")
                print(f"    Prediction: {pred} (confidence: {conf:.3f})")
                print(f"    Expected: real, Got: {pred} - {'✅ CORRECT' if pred == 'real' else '❌ WRONG'}")
            else:
                print(f"  Video {i+1}: {video.name} - ERROR: {result.get('error')}")
        
        print("\n🤖 Testing FAKE videos (should predict FAKE):")
        for i, video in enumerate(fake_videos):
            result = predictor.predict_video(video, max_frames=5)
            if result.get('success'):
                pred = result['prediction']
                conf = result['confidence']
                print(f"  Video {i+1}: {video.name}")
                print(f"    Prediction: {pred} (confidence: {conf:.3f})")
                print(f"    Expected: deepfake, Got: {pred} - {'✅ CORRECT' if pred == 'deepfake' else '❌ WRONG'}")
            else:
                print(f"  Video {i+1}: {video.name} - ERROR: {result.get('error')}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS:")
    print("=" * 60)
    
    # Check training logs for pattern
    log_files = list(Path(".").glob("improved_xception_training_*.log"))
    if log_files:
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        print(f"📊 Latest training log: {latest_log}")
        
        with open(latest_log, 'r') as f:
            content = f.read()
            
        # Extract key metrics
        if "Best validation accuracy: 52.33%" in content:
            print("❌ CRITICAL ISSUE: Model accuracy is 52.33% (random chance)")
            print("   This means the model didn't learn to distinguish real from fake!")
            
        if "Train Acc: 49" in content or "Train Acc: 50" in content or "Train Acc: 51" in content:
            print("❌ Training accuracy around 50% indicates model is not learning")
            
        print("\n💡 LIKELY CAUSES:")
        print("   1. Dataset quality issues (mislabeled or too similar)")
        print("   2. Model architecture problems")
        print("   3. Training hyperparameters need adjustment")
        print("   4. Insufficient data augmentation")
        print("   5. Face extraction quality issues")
    
    print("\n🔧 RECOMMENDED FIXES:")
    print("   1. Check dataset labels and quality")
    print("   2. Retrain with better hyperparameters")
    print("   3. Use data augmentation")
    print("   4. Try different model architecture")
    print("   5. Use transfer learning from pretrained model")

def check_dataset_labels():
    """Check if dataset labels are correct"""
    print("\n" + "=" * 60)
    print("CHECKING DATASET LABELS")
    print("=" * 60)
    
    dataset_path = Path("dataset")
    if not dataset_path.exists():
        print("❌ Dataset folder not found")
        return
    
    real_path = dataset_path / "real"
    fake_path = dataset_path / "fake"
    
    if real_path.exists():
        real_videos = list(real_path.glob("*.mp4"))
        print(f"📁 Real videos folder: {len(real_videos)} videos")
        for i, video in enumerate(real_videos[:5]):
            print(f"   {i+1}. {video.name}")
        if len(real_videos) > 5:
            print(f"   ... and {len(real_videos) - 5} more")
    else:
        print("❌ Real videos folder not found")
    
    if fake_path.exists():
        fake_videos = list(fake_path.glob("*.mp4"))
        print(f"📁 Fake videos folder: {len(fake_videos)} videos")
        for i, video in enumerate(fake_videos[:5]):
            print(f"   {i+1}. {video.name}")
        if len(fake_videos) > 5:
            print(f"   ... and {len(fake_videos) - 5} more")
    else:
        print("❌ Fake videos folder not found")

if __name__ == "__main__":
    test_prediction_logic()
    check_dataset_labels()
