#!/usr/bin/env python3
"""
Test the new XceptionNet ensemble detector to see if it reduces false positives
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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ensemble_vs_current():
    """Compare ensemble model performance vs current model"""
    
    print("🚀 TESTING ENSEMBLE MODEL (ResNet50 + XceptionNet)")
    print("=" * 70)
    
    # Test current advanced model
    print("\n📊 Testing Current FaceForensics++ Model:")
    print("-" * 50)
    
    current_detector = get_detector(use_ensemble=False, use_advanced_model=True)
    
    # Test ensemble model  
    print("\n🎯 Testing New Ensemble Model (ResNet50 + XceptionNet):")
    print("-" * 60)
    
    ensemble_detector = get_detector(use_ensemble=True, use_advanced_model=True)
    
    # Get recent failed predictions for testing
    recent_results = DetectionResult.objects.filter(
        prediction='FAKE'
    ).order_by('-created_at')[:5]
    
    if not recent_results.exists():
        print("❌ No recent FAKE predictions found for testing")
        return
    
    print(f"\n🔍 Testing on {recent_results.count()} recent videos that were predicted as FAKE:")
    print("=" * 70)
    
    improvements = 0
    total_tests = 0
    
    for result in recent_results:
        if not result.video_file:
            continue
            
        video_path = result.video_file.path
        if not os.path.exists(video_path):
            continue
            
        total_tests += 1
        
        print(f"\n🎬 Testing: {result.original_filename}")
        print(f"   Original prediction: {result.prediction} (confidence: {result.confidence_score:.3f})")
        
        # Test current model
        print("   🔄 Current model predicting...")
        current_result = current_detector.detect_video(video_path)
        current_pred = current_result.get('prediction', 'ERROR')
        current_conf = current_result.get('confidence_score', 0.0)
        
        # Test ensemble model
        print("   🔄 Ensemble model predicting...")
        ensemble_result = ensemble_detector.detect_video(video_path)
        ensemble_pred = ensemble_result.get('prediction', 'ERROR')
        ensemble_conf = ensemble_result.get('confidence_score', 0.0)
        
        print(f"   📊 Current:  {current_pred} (confidence: {current_conf:.3f})")
        print(f"   🎯 Ensemble: {ensemble_pred} (confidence: {ensemble_conf:.3f})")
        
        # Check if ensemble shows improvement (predicts REAL instead of FAKE)
        if result.prediction == 'FAKE' and ensemble_pred == 'REAL':
            improvements += 1
            print("   ✅ IMPROVEMENT: Ensemble correctly identifies as REAL!")
        elif ensemble_pred == 'REAL':
            print("   ✅ GOOD: Ensemble predicts REAL")
        elif current_pred == 'FAKE' and ensemble_pred == 'FAKE':
            print("   ⚠️  SAME: Both models predict FAKE")
        else:
            print("   ❓ MIXED: Different behaviors")
    
    print(f"\n📋 RESULTS SUMMARY:")
    print("=" * 30)
    print(f"   Total videos tested: {total_tests}")
    print(f"   Improvements found: {improvements}")
    if total_tests > 0:
        improvement_rate = (improvements / total_tests) * 100
        print(f"   Improvement rate: {improvement_rate:.1f}%")
        
        if improvement_rate > 50:
            print("   🎉 EXCELLENT: Ensemble shows significant improvement!")
        elif improvement_rate > 25:
            print("   👍 GOOD: Ensemble shows moderate improvement")
        elif improvement_rate > 0:
            print("   😐 MILD: Some improvement detected")
        else:
            print("   😞 NO IMPROVEMENT: Consider other approaches")

def test_threshold_sensitivity():
    """Test different thresholds with ensemble model"""
    
    print(f"\n🎛️  THRESHOLD SENSITIVITY TEST")
    print("=" * 40)
    
    # Test with different thresholds
    thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
    
    # Get a sample video that was predicted as FAKE
    sample_result = DetectionResult.objects.filter(prediction='FAKE').first()
    
    if not sample_result or not sample_result.video_file:
        print("❌ No sample video available for threshold testing")
        return
    
    video_path = sample_result.video_file.path
    if not os.path.exists(video_path):
        print("❌ Sample video file not found")
        return
    
    print(f"🎬 Testing with: {sample_result.original_filename}")
    print(f"   Original prediction: {sample_result.prediction}")
    print()
    
    for threshold in thresholds:
        from detector.xception_ensemble import EnsembleDeepfakeDetector
        
        detector = EnsembleDeepfakeDetector(threshold=threshold)
        result = detector.detect_video(video_path)
        
        pred = result.get('prediction', 'ERROR')
        conf = result.get('confidence_score', 0.0)
        fake_prob = result.get('fake_probability', 0.0)
        
        print(f"   Threshold {threshold:.2f}: {pred} (conf: {conf:.3f}, fake_prob: {fake_prob:.1f}%)")

if __name__ == "__main__":
    print("🔬 ENSEMBLE MODEL EVALUATION")
    print("=" * 50)
    
    try:
        # Test ensemble vs current
        test_ensemble_vs_current()
        
        # Test threshold sensitivity
        test_threshold_sensitivity()
        
        print(f"\n✅ EVALUATION COMPLETE!")
        print(f"\n💡 RECOMMENDATIONS:")
        print("   1. If ensemble shows >50% improvement, switch to ensemble model")
        print("   2. Fine-tune threshold based on sensitivity test results")
        print("   3. Consider ensemble threshold between 0.65-0.75 for best balance")
        print("   4. Monitor production performance and adjust as needed")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
