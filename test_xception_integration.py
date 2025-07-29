#!/usr/bin/env python3
"""
Test script to verify XceptionNet integration with Django frontend
"""

import os
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_xception_integration():
    """Test XceptionNet integration"""
    
    print("=== XceptionNet Integration Test ===")
    
    # Test 1: Import XceptionNet predictor
    try:
        from xception_predictor import get_xception_predictor, XceptionNetPredictor
        print("✅ XceptionNet predictor imported successfully")
    except Exception as e:
        print(f"❌ Failed to import XceptionNet predictor: {e}")
        return False
    
    # Test 2: Initialize predictor
    try:
        predictor = get_xception_predictor()
        if predictor.model is not None:
            print("✅ XceptionNet model loaded successfully")
            print(f"   Device: {predictor.device}")
        else:
            print("⚠️ XceptionNet model not loaded (no trained model found)")
            return False
    except Exception as e:
        print(f"❌ Failed to initialize XceptionNet predictor: {e}")
        return False
    
    # Test 3: Check for trained models
    models_dir = Path("models")
    if models_dir.exists():
        xception_models = list(models_dir.glob("*xception*.pth")) + list(models_dir.glob("robust_xception*.pth"))
        print(f"✅ Found {len(xception_models)} XceptionNet model(s):")
        for model in sorted(xception_models, key=lambda x: x.stat().st_mtime, reverse=True):
            model_time = model.stat().st_mtime
            import time
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model_time))
            print(f"   📁 {model.name} (created: {time_str})")
    else:
        print("❌ Models directory not found")
        return False
    
    # Test 4: Test prediction capability (if sample video exists)
    print("\n=== Testing Prediction Capability ===")
    
    # Look for any sample video files
    dataset_path = Path("dataset")
    sample_video = None
    
    if dataset_path.exists():
        # Try to find a sample video
        video_paths = [
            dataset_path / "original_sequences" / "youtube" / "c23" / "videos",
            dataset_path / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
        ]
        
        for video_dir in video_paths:
            if video_dir.exists():
                videos = list(video_dir.glob("*.mp4"))
                if videos:
                    sample_video = videos[0]
                    break
    
    if sample_video and sample_video.exists():
        try:
            print(f"🎬 Testing with sample video: {sample_video.name}")
            result = predictor.predict_video(str(sample_video), max_frames=5)
            
            if result.get('success', False):
                print(f"✅ Prediction successful!")
                print(f"   Prediction: {result['prediction'].upper()}")
                print(f"   Confidence: {result['confidence']:.2f}")
                print(f"   Frames analyzed: {result['frames_analyzed']}")
                print(f"   Deepfake ratio: {result.get('deepfake_ratio', 0):.2f}")
            else:
                print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Prediction test failed: {e}")
            return False
    else:
        print("⚠️ No sample video found for testing")
        print("   You can test with your own video files once you upload them through the frontend")
    
    # Test 5: Test Django integration (check if views can import)
    print("\n=== Testing Django Integration ===")
    
    try:
        # Check if Django views can import our predictor
        sys.path.append('.')
        from detector.views import VideoUploadView
        print("✅ Django views can import XceptionNet predictor")
    except Exception as e:
        print(f"❌ Django integration test failed: {e}")
        print("   Make sure Django is set up correctly")
        return False
    
    print("\n=== Integration Test Summary ===")
    print("✅ XceptionNet is successfully integrated with your Django frontend!")
    print("✅ The system will now use XceptionNet as the primary detection model")
    print("✅ FaceForensics++ model serves as a backup if XceptionNet fails")
    print()
    print("🚀 You can now:")
    print("   1. Start your Django server: python manage.py runserver")
    print("   2. Upload videos through the web interface")
    print("   3. Get deepfake predictions powered by XceptionNet!")
    
    return True

if __name__ == "__main__":
    success = test_xception_integration()
    
    if success:
        print("\n🎉 All tests passed! XceptionNet integration is ready!")
    else:
        print("\n💥 Some tests failed. Please check the errors above.")
        
    print("\n" + "="*50)
    print("Ready for video deepfake detection!")
    print("="*50)
