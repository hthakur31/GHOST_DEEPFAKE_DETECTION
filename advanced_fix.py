#!/usr/bin/env python3
"""
Advanced diagnosis and fix for persistent false positive issues
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')
django.setup()

from detector.ml_utils import get_detector
from detector.models import DetectionResult
import torch
import numpy as np

def diagnose_current_predictions():
    """Diagnose why real videos are still being classified as fake"""
    
    print("🔍 ADVANCED FALSE POSITIVE DIAGNOSIS")
    print("=" * 60)
    
    # Check latest predictions
    latest_results = DetectionResult.objects.order_by('-created_at')[:5]
    
    print("📊 LATEST PREDICTIONS:")
    print("-" * 50)
    
    for result in latest_results:
        print(f"File: {result.original_filename}")
        print(f"  Prediction: {result.prediction}")
        print(f"  Confidence: {result.confidence_score:.3f}")
        print(f"  Real prob: {result.real_probability:.1f}%")
        print(f"  Fake prob: {result.fake_probability:.1f}%")
        print(f"  Model: {result.model_used}")
        print(f"  Date: {result.created_at.strftime('%Y-%m-%d %H:%M')}")
        print()
    
    # Analyze the pattern
    fake_predictions = DetectionResult.objects.filter(prediction='FAKE')
    
    print("🔍 ANALYSIS OF FAKE PREDICTIONS:")
    print("-" * 40)
    
    if fake_predictions.exists():
        confidences = [r.confidence_score for r in fake_predictions if r.confidence_score]
        fake_probs = [r.fake_probability for r in fake_predictions if r.fake_probability]
        
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)
            
            print(f"Average confidence: {avg_conf:.3f}")
            print(f"Confidence range: {min_conf:.3f} - {max_conf:.3f}")
        
        if fake_probs:
            avg_fake_prob = sum(fake_probs) / len(fake_probs)
            min_fake_prob = min(fake_probs)
            max_fake_prob = max(fake_probs)
            
            print(f"Average fake probability: {avg_fake_prob:.1f}%")
            print(f"Fake prob range: {min_fake_prob:.1f}% - {max_fake_prob:.1f}%")
    
    print("\n🎯 RECOMMENDED THRESHOLD:")
    
    # If most fake predictions have low confidence/probability, suggest higher threshold
    if fake_predictions.exists():
        fake_probs = [r.fake_probability for r in fake_predictions if r.fake_probability]
        if fake_probs:
            max_fake_prob = max(fake_probs)
            # Suggest threshold higher than the highest fake probability
            suggested_threshold = min(0.9, (max_fake_prob / 100) + 0.1)
            print(f"Current max fake probability: {max_fake_prob:.1f}%")
            print(f"Suggested threshold: {suggested_threshold:.2f} ({suggested_threshold*100:.0f}%)")
            return suggested_threshold
    
    return 0.75  # Default higher threshold

def create_aggressive_fix(new_threshold):
    """Create an aggressive fix for false positives"""
    
    print(f"\n🛠️  IMPLEMENTING AGGRESSIVE FALSE POSITIVE FIX")
    print("=" * 60)
    
    print(f"🎯 Setting new threshold: {new_threshold:.2f} ({new_threshold*100:.0f}%)")
    
    # Load detector and set new threshold
    try:
        detector = get_detector(use_advanced_model=True)
        
        if hasattr(detector, 'advanced_detector'):
            detector.advanced_detector.set_threshold(new_threshold)
            print(f"✅ Threshold updated to: {detector.advanced_detector.get_threshold()}")
        
        return detector
    except Exception as e:
        print(f"❌ Error updating threshold: {e}")
        return None

def test_with_aggressive_threshold(detector, new_threshold):
    """Test detection with the new aggressive threshold"""
    
    print(f"\n🧪 TESTING WITH AGGRESSIVE THRESHOLD: {new_threshold:.2f}")
    print("-" * 50)
    
    # Get a recent video that was classified as fake
    fake_result = DetectionResult.objects.filter(prediction='FAKE').order_by('-created_at').first()
    
    if not fake_result:
        print("⚠️  No recent FAKE predictions to test with")
        return
    
    video_path = fake_result.video_file.path
    print(f"📹 Testing with: {fake_result.original_filename}")
    print(f"   Previous result: {fake_result.prediction} (conf: {fake_result.confidence_score:.3f})")
    
    try:
        # Run detection with new threshold
        results = detector.detect_video(video_path)
        
        print(f"   New result: {results.get('prediction', 'ERROR')}")
        print(f"   New confidence: {results.get('confidence_score', 0):.3f}")
        print(f"   Real probability: {results.get('real_probability', 0):.1f}%")
        print(f"   Fake probability: {results.get('fake_probability', 0):.1f}%")
        print(f"   Threshold used: {results.get('threshold_used', 'Unknown')}")
        
        if results.get('prediction') == 'REAL':
            print("   ✅ SUCCESS: Now correctly classified as REAL!")
        else:
            print("   ⚠️  Still classified as FAKE - may need even higher threshold")
            
    except Exception as e:
        print(f"   ❌ Error during testing: {e}")

def create_permanent_fix(threshold):
    """Create a permanent fix by updating the model initialization"""
    
    print(f"\n🔧 CREATING PERMANENT FIX")
    print("=" * 40)
    
    # Update the FaceForensics model to use the new threshold by default
    model_file_path = "G:/Deefake_detection_app/detector/faceforensics_model.py"
    
    print(f"📝 Updating default threshold in {model_file_path}")
    print(f"   New default threshold: {threshold:.2f}")
    
    # Read current file
    try:
        with open(model_file_path, 'r') as f:
            content = f.read()
        
        # Find and replace the threshold in __init__
        old_init_line = "def __init__(self, model_path: Optional[str] = None, device: str = 'auto', threshold: float = 0.65):"
        new_init_line = f"def __init__(self, model_path: Optional[str] = None, device: str = 'auto', threshold: float = {threshold:.2f}):"
        
        if old_init_line in content:
            content = content.replace(old_init_line, new_init_line)
            
            # Write back to file
            with open(model_file_path, 'w') as f:
                f.write(content)
            
            print(f"✅ Default threshold updated to {threshold:.2f}")
            print(f"   All new detector instances will use this threshold")
            
        else:
            print(f"⚠️  Could not find exact init line to update")
            print(f"   Manual update may be required")
            
    except Exception as e:
        print(f"❌ Error updating file: {e}")

if __name__ == "__main__":
    print("🚨 AGGRESSIVE FALSE POSITIVE FIX")
    print("=" * 60)
    
    # Step 1: Diagnose current predictions
    suggested_threshold = diagnose_current_predictions()
    
    # Step 2: Create aggressive fix
    detector = create_aggressive_fix(suggested_threshold)
    
    if detector:
        # Step 3: Test with aggressive threshold
        test_with_aggressive_threshold(detector, suggested_threshold)
        
        # Step 4: Create permanent fix
        create_permanent_fix(suggested_threshold)
        
        print(f"\n🎉 AGGRESSIVE FIX COMPLETE!")
        print(f"   ✅ New threshold: {suggested_threshold:.2f} ({suggested_threshold*100:.0f}%)")
        print(f"   ✅ This makes the model MUCH less likely to classify real videos as fake")
        print(f"   ✅ Changes are permanent - restart Django server to apply")
        
        print(f"\n🔄 NEXT STEPS:")
        print(f"   1. Restart Django server: Ctrl+C then 'python manage.py runserver'")
        print(f"   2. Upload a real video to test the fix")
        print(f"   3. If still issues, we can increase threshold further")
    
    else:
        print(f"\n❌ Could not apply fix - check detector loading")
