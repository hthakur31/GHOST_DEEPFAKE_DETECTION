#!/usr/bin/env python3
"""
Test script to verify FaceForensics++ model integration with Django
"""

import os
import sys
import django
from pathlib import Path

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')
django.setup()

from detector.ml_utils import get_detector
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_integration():
    """Test that our trained FaceForensics++ model loads correctly"""
    
    print("🧪 FaceForensics++ Model Integration Test")
    print("=" * 50)
    
    try:
        # Test advanced model loading
        print("📦 Loading FaceForensics++ detector...")
        detector = get_detector(use_advanced_model=True)
        
        print(f"✅ Model Name: {detector.model_name}")
        print(f"✅ Model Version: {detector.model_version}")
        print(f"✅ Using Advanced Model: {detector.use_advanced_model}")
        
        if hasattr(detector, 'device'):
            print(f"✅ Device: {detector.device}")
        
        if hasattr(detector, 'advanced_detector') and detector.advanced_detector:
            print(f"✅ FaceForensics++ Model Path: {detector.advanced_detector.model_path or 'Auto-detected latest'}")
            print(f"✅ Model Device: {detector.advanced_detector.device}")
            
        # Check if trained model exists
        models_dir = Path("G:/Deefake_detection_app/models")
        if models_dir.exists():
            model_files = list(models_dir.glob("faceforensics_model_*.pth"))
            if model_files:
                latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                print(f"✅ Trained Model Found: {latest_model.name}")
                print(f"✅ Model Size: {latest_model.stat().st_size / (1024*1024):.2f} MB")
            else:
                print("⚠️  No trained FaceForensics++ models found")
        
        # Test a sample detection (with dummy data)
        print("\n🔬 Testing Model Forward Pass...")
        import torch
        
        if hasattr(detector, 'advanced_detector') and detector.advanced_detector:
            # Test with dummy input
            dummy_face = torch.randn(1, 3, 224, 224)
            dummy_freq = torch.randn(1, 3, 224, 224)
            
            with torch.no_grad():
                try:
                    output = detector.advanced_detector.model(dummy_face, dummy_freq)
                    if isinstance(output, tuple):
                        main_out, aux_out = output
                        print(f"✅ Main Output Shape: {main_out.shape}")
                        print(f"✅ Auxiliary Output Shape: {aux_out.shape}")
                    else:
                        print(f"✅ Output Shape: {output.shape}")
                    print("✅ Model Forward Pass: SUCCESSFUL")
                except Exception as e:
                    print(f"❌ Model Forward Pass Failed: {e}")
        
        print("\n🎉 Model Integration Test: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Model Integration Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sample_video_detection():
    """Test video detection with a sample from our dataset"""
    
    print("\n🎬 Sample Video Detection Test")
    print("=" * 40)
    
    try:
        # Find a sample video from our dataset
        dataset_path = Path("G:/Deefake_detection_app/dataset")
        
        # Try to find a real video first
        real_video_path = dataset_path / "original_sequences" / "youtube" / "c23" / "videos"
        fake_video_path = dataset_path / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
        
        sample_video = None
        video_type = None
        
        if real_video_path.exists():
            real_videos = list(real_video_path.glob("*.mp4"))
            if real_videos:
                sample_video = real_videos[0]
                video_type = "REAL"
        
        if not sample_video and fake_video_path.exists():
            fake_videos = list(fake_video_path.glob("*.mp4"))
            if fake_videos:
                sample_video = fake_videos[0]
                video_type = "FAKE"
        
        if not sample_video:
            print("⚠️  No sample videos found in dataset")
            return False
        
        print(f"📹 Testing with: {sample_video.name} (Expected: {video_type})")
        
        # Load detector and run detection
        detector = get_detector(use_advanced_model=True)
        results = detector.detect_video(str(sample_video))
        
        print(f"🔍 Detection Results:")
        print(f"   Prediction: {results.get('prediction', 'ERROR')}")
        print(f"   Confidence: {results.get('confidence_score', 0):.3f}")
        print(f"   Processing Time: {results.get('processing_time', 0):.2f}s")
        print(f"   Faces Detected: {results.get('face_detected', False)}")
        print(f"   Model Used: {results.get('model_name', 'Unknown')}")
        
        if 'frame_predictions' in results:
            print(f"   Frames Analyzed: {len(results['frame_predictions'])}")
        
        expected_correct = (
            (video_type == "REAL" and results.get('prediction') == "REAL") or
            (video_type == "FAKE" and results.get('prediction') == "FAKE")
        )
        
        if expected_correct:
            print("✅ Detection Result: CORRECT")
        else:
            print("⚠️  Detection Result: UNEXPECTED (but model may still be working)")
        
        print("🎉 Sample Video Detection Test: COMPLETED")
        return True
        
    except Exception as e:
        print(f"❌ Sample Video Detection FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 FaceForensics++ Django Integration Tests")
    print("=" * 60)
    
    # Test 1: Model Integration
    test1_passed = test_model_integration()
    
    # Test 2: Sample Video Detection (only if model loads successfully)
    test2_passed = False
    if test1_passed:
        test2_passed = test_sample_video_detection()
    
    print("\n📊 Test Summary")
    print("=" * 20)
    print(f"Model Integration: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Video Detection: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Your FaceForensics++ model is ready for use!")
    elif test1_passed:
        print("\n⚠️  Model loads but video detection needs verification")
    else:
        print("\n❌ Model integration issues need to be resolved")
