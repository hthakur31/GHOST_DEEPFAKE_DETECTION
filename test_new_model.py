#!/usr/bin/env python3
"""
Quick test script to verify the new model works correctly
"""

import sys
import time
from pathlib import Path
from enhanced_xception_predictor import EnhancedXceptionNetPredictor

def test_new_model():
    """Test the newly trained model"""
    print("🧪 Testing New Simple XceptionNet Model")
    print("=" * 50)
    
    # Look for the latest simple_xception model
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found")
        return
    
    simple_models = list(models_dir.glob("simple_xception*.pth"))
    if not simple_models:
        print("❌ No SimpleXceptionNet models found")
        print("   Training might still be in progress...")
        return
    
    latest_model = max(simple_models, key=lambda x: x.stat().st_mtime)
    print(f"📱 Found model: {latest_model}")
    
    # Initialize predictor with the specific model
    predictor = EnhancedXceptionNetPredictor(model_path=str(latest_model))
    
    if predictor.model is None:
        print("❌ Failed to load model")
        return
    
    print(f"✅ Model loaded successfully!")
    print(f"   Architecture: {predictor.model_type}")
    print(f"   Device: {predictor.device}")
    
    # Test with dummy data
    import torch
    dummy_input = torch.randn(1, 3, 224, 224).to(predictor.device)
    
    with torch.no_grad():
        outputs = predictor.model(dummy_input)
        probabilities = torch.softmax(outputs, dim=1)
        
        real_prob = probabilities[0][0].item()
        fake_prob = probabilities[0][1].item()
        
        print(f"\n🔍 Model Test Results:")
        print(f"   Real probability: {real_prob:.4f}")
        print(f"   Fake probability: {fake_prob:.4f}")
        print(f"   Prediction: {'FAKE' if fake_prob > 0.5 else 'REAL'}")
    
    # Test with actual videos if available
    real_videos = list(Path("dataset/original_sequences/youtube/c23/videos").glob("*.mp4"))[:2]
    fake_videos = list(Path("dataset/manipulated_sequences/Deepfakes/c23/videos").glob("*.mp4"))[:2]
    
    if real_videos:
        print(f"\n🎬 Testing REAL videos:")
        for video in real_videos:
            result = predictor.predict_video(video, max_frames=5)
            if result.get('success'):
                pred = result['prediction']
                conf = result['confidence']
                print(f"   {video.name}: {pred} ({conf:.3f}) - {'✅' if pred == 'real' else '❌'}")
    
    if fake_videos:
        print(f"\n🤖 Testing FAKE videos:")
        for video in fake_videos:
            result = predictor.predict_video(video, max_frames=5)
            if result.get('success'):
                pred = result['prediction']
                conf = result['confidence']
                print(f"   {video.name}: {pred} ({conf:.3f}) - {'✅' if pred == 'deepfake' else '❌'}")
    
    print("\n" + "=" * 50)
    return predictor

def wait_for_training():
    """Wait for training to complete"""
    print("⏳ Waiting for training to complete...")
    
    while True:
        # Check if training log shows completion
        log_files = list(Path(".").glob("simple_xception_training_*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_log, 'r') as f:
                content = f.read()
            
            if "Training completed successfully!" in content:
                print("✅ Training completed!")
                return True
            elif "ERROR" in content or "Failed" in content:
                print("❌ Training failed!")
                return False
        
        # Check if model files exist
        models_dir = Path("models")
        if models_dir.exists():
            simple_models = list(models_dir.glob("simple_xception_final_*.pth"))
            if simple_models:
                print("✅ Model file found!")
                return True
        
        print("   Still training... (checking again in 30 seconds)")
        time.sleep(30)

if __name__ == "__main__":
    # Check if training is complete
    models_dir = Path("models")
    simple_models = list(models_dir.glob("simple_xception*.pth")) if models_dir.exists() else []
    
    if simple_models:
        test_new_model()
    else:
        print("🚀 Simple XceptionNet training appears to be in progress...")
        print("   Run this script again once training completes")
        print("   You can check progress with: Get-Content simple_xception_training_*.log -Tail 20")
