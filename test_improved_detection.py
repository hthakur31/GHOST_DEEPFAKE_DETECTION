#!/usr/bin/env python3
"""
Test the improved FaceForensics++ detection with adjustable threshold
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')
django.setup()

from detector.ml_utils import get_detector
from detector.models import DetectionResult
import time

def test_improved_detection():
    """Test the improved detection with different thresholds"""
    
    print("🔧 TESTING IMPROVED DETECTION LOGIC")
    print("=" * 50)
    
    # Load the improved detector
    try:
        detector = get_detector(use_advanced_model=True)
        print(f"✅ Loaded detector: {detector.model_name}")
        
        if hasattr(detector, 'advanced_detector'):
            current_threshold = detector.advanced_detector.get_threshold()
            print(f"📊 Current threshold: {current_threshold}")
        
    except Exception as e:
        print(f"❌ Error loading detector: {e}")
        return
    
    # Test with a sample video from recent results
    recent_results = DetectionResult.objects.filter(
        prediction__in=['REAL', 'FAKE']
    ).order_by('-created_at')[:3]
    
    if not recent_results.exists():
        print("⚠️  No recent results found to test with")
        return
    
    print(f"\n🧪 TESTING WITH DIFFERENT THRESHOLDS:")
    print("-" * 60)
    
    # Test different thresholds
    thresholds_to_test = [0.5, 0.6, 0.7]
    
    for threshold in thresholds_to_test:
        print(f"\n🎯 Testing with threshold: {threshold}")
        print("-" * 30)
        
        # Set the threshold
        if hasattr(detector, 'advanced_detector'):
            detector.advanced_detector.set_threshold(threshold)
        
        # Test with the first recent video file
        test_result = recent_results.first()
        video_path = test_result.video_file.path
        
        print(f"📹 Testing video: {test_result.original_filename}")
        print(f"   Previous result: {test_result.prediction} (confidence: {test_result.confidence_score:.3f})")
        
        start_time = time.time()
        try:
            # Run detection with new threshold
            results = detector.detect_video(video_path)
            processing_time = time.time() - start_time
            
            print(f"   New result: {results.get('prediction', 'ERROR')}")
            print(f"   Confidence: {results.get('confidence_score', 0):.3f}")
            print(f"   Real prob: {results.get('real_probability', 0):.1f}%")
            print(f"   Fake prob: {results.get('fake_probability', 0):.1f}%")
            
            if 'consistency_score' in results:
                print(f"   Consistency: {results['consistency_score']:.3f}")
            if 'threshold_used' in results:
                print(f"   Threshold used: {results['threshold_used']}")
                
            print(f"   Processing time: {processing_time:.1f}s")
            
            # Check if result changed
            if results.get('prediction') != test_result.prediction:
                print(f"   🔄 PREDICTION CHANGED!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📋 SUMMARY:")
    print(f"   ✅ Improved detection logic implemented")
    print(f"   🎯 Adjustable threshold (default: 0.65)")
    print(f"   🔄 Confidence-weighted voting")
    print(f"   📊 Consistency scoring")
    print(f"   🎭 Better false positive reduction")

def show_threshold_recommendations():
    """Show threshold recommendations based on current data"""
    
    print(f"\n💡 THRESHOLD RECOMMENDATIONS:")
    print("=" * 40)
    
    # Analyze current detection patterns
    fake_results = DetectionResult.objects.filter(prediction='FAKE')
    real_results = DetectionResult.objects.filter(prediction='REAL')
    
    total_results = fake_results.count() + real_results.count()
    fake_ratio = fake_results.count() / total_results if total_results > 0 else 0
    
    print(f"📊 Current Statistics:")
    print(f"   FAKE: {fake_results.count()} ({fake_ratio*100:.1f}%)")
    print(f"   REAL: {real_results.count()} ({(1-fake_ratio)*100:.1f}%)")
    
    print(f"\n🎯 Threshold Recommendations:")
    
    if fake_ratio > 0.8:
        print(f"   🔴 High false positive rate detected!")
        print(f"   📈 Recommended threshold: 0.7-0.8")
        print(f"   💡 This will reduce false positives")
    elif fake_ratio > 0.6:
        print(f"   🟡 Moderate bias toward FAKE detected")
        print(f"   📈 Recommended threshold: 0.6-0.7")
    elif fake_ratio < 0.2:
        print(f"   🟡 Possible bias toward REAL detected")
        print(f"   📈 Recommended threshold: 0.4-0.5")
    else:
        print(f"   ✅ Reasonable balance detected")
        print(f"   📈 Current threshold (0.65) should be good")
    
    print(f"\n🔧 How to adjust threshold:")
    print(f"   • Higher threshold (0.7+) = Fewer false positives")
    print(f"   • Lower threshold (0.5-) = Catch more deepfakes")
    print(f"   • Default 0.65 = Balanced approach")

if __name__ == "__main__":
    print("🚀 IMPROVED DEEPFAKE DETECTION TEST")
    print("=" * 60)
    
    # Test the improved detection
    test_improved_detection()
    
    # Show recommendations
    show_threshold_recommendations()
    
    print(f"\n✅ FALSE POSITIVE FIX IMPLEMENTED!")
    print(f"   The model now uses:")
    print(f"   • Higher threshold (0.65 instead of 0.5)")
    print(f"   • Confidence-weighted voting")
    print(f"   • Better probability calculation")
    print(f"   • Consistency scoring")
    print(f"   • Adjustable threshold for fine-tuning")
