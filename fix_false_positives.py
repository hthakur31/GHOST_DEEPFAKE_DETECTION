#!/usr/bin/env python3
"""
Immediate fix for false positive issues using conservative threshold adjustment
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')
django.setup()

from detector.models import DetectionResult
from detector.ml_utils import get_detector
import logging

logger = logging.getLogger(__name__)

def analyze_current_predictions():
    """Analyze current predictions to understand the problem"""
    
    print("🔍 ANALYZING CURRENT PREDICTIONS")
    print("=" * 50)
    
    # Get all predictions
    all_results = DetectionResult.objects.all().order_by('-created_at')
    recent_results = all_results[:20]  # Get recent results first
    
    fake_predictions = all_results.filter(prediction='FAKE')[:10]
    real_predictions = all_results.filter(prediction='REAL')[:10]
    
    print(f"Recent predictions breakdown:")
    print(f"   FAKE: {fake_predictions.count()}")
    print(f"   REAL: {real_predictions.count()}")
    
    if fake_predictions.exists():
        fake_confidences = [r.confidence_score for r in fake_predictions if r.confidence_score]
        if fake_confidences:
            avg_fake_conf = sum(fake_confidences) / len(fake_confidences)
            max_fake_conf = max(fake_confidences)
            min_fake_conf = min(fake_confidences)
            
            print(f"\nFAKE prediction confidence analysis:")
            print(f"   Average: {avg_fake_conf:.3f}")
            print(f"   Range: {min_fake_conf:.3f} - {max_fake_conf:.3f}")
            
            # If most fake predictions have low confidence, this suggests a threshold issue
            if avg_fake_conf < 0.6:
                print(f"   🚨 ISSUE: Low confidence suggests threshold too low")
                return max_fake_conf + 0.1  # Suggest threshold higher than max fake confidence
    
    return 0.7  # Default safe threshold

def create_improved_detection():
    """Create improved detection logic with higher threshold"""
    
    print(f"\n🛠️  CREATING IMPROVED DETECTION")
    print("=" * 40)
    
    # Create improved detector code that patches the existing system
    improved_code = '''
# PATCH: detector/faceforensics_model.py - Line ~470
# Replace the threshold decision logic with this improved version:

# BEFORE (problematic logic):
# if weighted_fake_score > self.threshold:
#     final_prediction = 1  # Fake

# AFTER (improved logic):
# Conservative approach - require HIGH confidence for FAKE classification
conservative_threshold = max(0.75, self.threshold)  # At least 75% confidence required

# Additional checks for better accuracy
uncertainty_margin = 0.05  # 5% margin for uncertainty
confidence_gap = abs(weighted_fake_score - weighted_real_score)

if weighted_fake_score > conservative_threshold and confidence_gap > uncertainty_margin:
    # Only classify as FAKE if we're very confident AND there's a clear gap
    final_prediction = 1  # Fake
    final_confidence = weighted_fake_score
else:
    # Default to REAL for uncertain cases (reduces false positives)
    final_prediction = 0  # Real
    final_confidence = weighted_real_score
'''
    
    print("Improved detection logic:")
    print("• Higher threshold (75% minimum)")
    print("• Confidence gap requirement")
    print("• Default to REAL for uncertain cases")
    
    return improved_code

def apply_quick_fix():
    """Apply a quick fix to reduce false positives"""
    
    print(f"\n⚡ APPLYING QUICK FIX")
    print("=" * 30)
    
    try:
        # Read the current file
        faceforensics_path = "G:/Deefake_detection_app/detector/faceforensics_model.py"
        
        with open(faceforensics_path, 'r') as f:
            content = f.read()
        
        # Find and replace the problematic threshold logic
        old_logic = '''                # Apply improved threshold logic - higher threshold reduces false positives
                if weighted_fake_score > self.threshold:
                    final_prediction = 1  # Fake
                    final_confidence = weighted_fake_score
                else:
                    final_prediction = 0  # Real
                    final_confidence = weighted_real_score'''
        
        new_logic = '''                # Apply conservative threshold logic - significantly reduce false positives
                conservative_threshold = max(0.75, self.threshold)  # At least 75% confidence required
                uncertainty_margin = 0.05  # 5% margin for uncertainty
                confidence_gap = abs(weighted_fake_score - weighted_real_score)
                
                # Only classify as FAKE if we're very confident AND there's a clear gap
                if (weighted_fake_score > conservative_threshold and 
                    confidence_gap > uncertainty_margin and 
                    weighted_fake_score > 0.7):  # Triple check
                    final_prediction = 1  # Fake
                    final_confidence = weighted_fake_score
                else:
                    # Default to REAL for uncertain cases (reduces false positives)
                    final_prediction = 0  # Real
                    final_confidence = weighted_real_score'''
        
        if old_logic in content:
            new_content = content.replace(old_logic, new_logic)
            
            # Backup the original file
            backup_path = faceforensics_path + ".backup"
            with open(backup_path, 'w') as f:
                f.write(content)
            print(f"✅ Backup created: {backup_path}")
            
            # Write the improved version
            with open(faceforensics_path, 'w') as f:
                f.write(new_content)
            print(f"✅ Applied improved threshold logic")
            
            return True
        else:
            print("❌ Could not find target logic to replace")
            return False
            
    except Exception as e:
        print(f"❌ Error applying fix: {e}")
        return False

def test_improved_model():
    """Test the improved model"""
    
    print(f"\n🧪 TESTING IMPROVED MODEL")
    print("=" * 35)
    
    try:
        # Get an improved detector
        detector = get_detector(use_ensemble=False, use_advanced_model=True)
        
        # Test on a recent FAKE prediction
        recent_fake = DetectionResult.objects.filter(prediction='FAKE').first()
        
        if not recent_fake or not recent_fake.video_file:
            print("❌ No test video available")
            return
        
        video_path = recent_fake.video_file.path
        if not os.path.exists(video_path):
            print("❌ Test video not found")
            return
        
        print(f"🎬 Testing with: {recent_fake.original_filename}")
        print(f"   Original prediction: {recent_fake.prediction} (confidence: {recent_fake.confidence_score:.3f})")
        
        # Test improved model
        result = detector.detect_video(video_path)
        
        new_pred = result.get('prediction', 'ERROR')
        new_conf = result.get('confidence_score', 0.0)
        
        print(f"   New prediction: {new_pred} (confidence: {new_conf:.3f})")
        
        if recent_fake.prediction == 'FAKE' and new_pred == 'Real':
            print("   ✅ SUCCESS: False positive corrected!")
        elif new_pred == 'Real':
            print("   ✅ GOOD: Predicts REAL")
        else:
            print("   ⚠️  Still predicting FAKE")
        
    except Exception as e:
        print(f"❌ Error testing: {e}")

if __name__ == "__main__":
    print("🚨 FALSE POSITIVE FIX")
    print("=" * 30)
    
    # Step 1: Analyze current predictions
    suggested_threshold = analyze_current_predictions()
    print(f"\n💡 Suggested threshold: {suggested_threshold:.2f}")
    
    # Step 2: Show improved logic
    improved_code = create_improved_detection()
    
    # Step 3: Apply quick fix
    if apply_quick_fix():
        print(f"\n🎉 FALSE POSITIVE FIX APPLIED!")
        
        # Step 4: Test the improved model
        test_improved_model()
        
        print(f"\n📋 SUMMARY:")
        print("✅ Applied conservative threshold logic")
        print("✅ Increased minimum confidence requirement to 75%")
        print("✅ Added confidence gap requirement")
        print("✅ Default to REAL for uncertain predictions")
        print()
        print("🎯 Expected Results:")
        print("• Significantly fewer false positives")
        print("• Higher confidence in FAKE predictions")
        print("• More conservative classification")
        
    else:
        print(f"\n❌ Could not apply automatic fix")
        print("Please manually update the threshold logic in:")
        print("detector/faceforensics_model.py around line 470")
        print(improved_code)
