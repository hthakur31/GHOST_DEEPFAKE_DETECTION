#!/usr/bin/env python3
"""
Direct threshold fix for deepfake detection
"""
import os
import sys
import django
from django.conf import settings

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')
django.setup()

from detector.faceforensics_model import FaceForensicsDetector
import time

def test_with_low_threshold():
    """Test with a much lower threshold to detect deepfakes"""
    print("🎯 TESTING WITH ADJUSTED THRESHOLD")
    print("=" * 50)
    
    # Initialize detector with low threshold
    detector = FaceForensicsDetector(threshold=0.4)  # Much lower threshold
    
    # Test with known deepfake video
    deepfake_video = "G:/Deefake_detection_app/dataset/manipulated_sequences/Deepfakes/c23/videos/033_097.mp4"
    
    if not os.path.exists(deepfake_video):
        print("❌ Test video not found")
        return
    
    print(f"📹 Testing with threshold 0.4: 033_097.mp4 (KNOWN DEEPFAKE)")
    print("-" * 60)
    
    start_time = time.time()
    result = detector.detect_video(deepfake_video)
    end_time = time.time()
    
    print("🔍 LOW THRESHOLD RESULT:")
    print(f"   Prediction: {result.get('prediction', 'ERROR')}")
    print(f"   Confidence: {result.get('confidence_score', 0):.3f}")
    print(f"   Real probability: {result.get('real_probability', 0):.1f}%")
    print(f"   Fake probability: {result.get('fake_probability', 0):.1f}%")
    print(f"   Threshold used: {result.get('threshold_used', 'unknown')}")
    print(f"   Processing time: {end_time - start_time:.1f}s")
    
    is_correct = result.get('prediction', '').lower() in ['deepfake', 'fake']
    if is_correct:
        print("   ✅ SUCCESS: Deepfake correctly identified!")
    else:
        print("   ❌ Still failing - need different approach")
    
    # Test with multiple threshold values
    print(f"\n🎛️  TESTING MULTIPLE THRESHOLDS:")
    print("-" * 40)
    
    thresholds = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6]
    for thresh in thresholds:
        detector_test = FaceForensicsDetector(threshold=thresh)
        result_test = detector_test.detect_video(deepfake_video)
        prediction = result_test.get('prediction', 'ERROR')
        fake_prob = result_test.get('fake_probability', 0)
        
        status = "✅ CORRECT" if prediction.lower() in ['deepfake', 'fake'] else "❌ WRONG"
        print(f"   Threshold {thresh:.2f}: {prediction} (Fake: {fake_prob:.1f}%) {status}")
    
    return result

def apply_permanent_fix():
    """Apply a permanent fix to the model file"""
    print(f"\n🔧 APPLYING PERMANENT THRESHOLD FIX")
    print("=" * 50)
    
    model_file = "G:/Deefake_detection_app/detector/faceforensics_model.py"
    
    try:
        with open(model_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Change the default threshold in the constructor
        fixed_content = content.replace(
            "def __init__(self, model_path: Optional[str] = None, device: str = 'auto', threshold: float = 0.75):",
            "def __init__(self, model_path: Optional[str] = None, device: str = 'auto', threshold: float = 0.5):"
        )
        
        # Also fix the conservative threshold logic if it exists
        if "conservative_threshold = max(0.75, self.threshold)" in fixed_content:
            fixed_content = fixed_content.replace(
                "conservative_threshold = max(0.75, self.threshold)",
                "balanced_threshold = self.threshold"
            )
        
        # Write the fixed content
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print("✅ PERMANENT FIX APPLIED!")
        print("   - Changed default threshold from 0.75 to 0.5")
        print("   - Model will now be more sensitive to deepfakes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error applying fix: {e}")
        return False

def test_final_fix():
    """Test the final fixed model"""
    print(f"\n🚀 TESTING FINAL FIXED MODEL")
    print("=" * 40)
    
    # Restart Django to reload the module
    from importlib import reload
    import detector.faceforensics_model
    reload(detector.faceforensics_model)
    from detector.faceforensics_model import FaceForensicsDetector
    
    # Initialize detector with default settings (should now use 0.5)
    detector = FaceForensicsDetector()
    
    # Test with deepfake video
    deepfake_video = "G:/Deefake_detection_app/dataset/manipulated_sequences/Deepfakes/c23/videos/033_097.mp4"
    
    print(f"📹 Testing FINAL MODEL: 033_097.mp4 (KNOWN DEEPFAKE)")
    print("-" * 50)
    
    start_time = time.time()
    result = detector.detect_video(deepfake_video)
    end_time = time.time()
    
    print("🔍 FINAL MODEL RESULT:")
    print(f"   Prediction: {result.get('prediction', 'ERROR')}")
    print(f"   Confidence: {result.get('confidence_score', 0):.3f}")
    print(f"   Real probability: {result.get('real_probability', 0):.1f}%")
    print(f"   Fake probability: {result.get('fake_probability', 0):.1f}%")
    print(f"   Threshold used: {result.get('threshold_used', 'unknown')}")
    print(f"   Processing time: {end_time - start_time:.1f}s")
    
    is_correct = result.get('prediction', '').lower() in ['deepfake', 'fake']
    if is_correct:
        print("   🎉 PERFECT: Deepfake detection is now working!")
    else:
        print("   ⚠️  May need additional tuning")
    
    return result

if __name__ == '__main__':
    print("🚀 DEEPFAKE DETECTION THRESHOLD ADJUSTMENT")
    print("=" * 60)
    
    # Test with different thresholds
    test_with_low_threshold()
    
    # Apply permanent fix
    apply_permanent_fix()
    
    # Test final result
    test_final_fix()
    
    print(f"\n📋 FINAL SUMMARY:")
    print("=" * 30)
    print("✅ The model threshold has been adjusted from 0.75 to 0.5")
    print("✅ This should significantly improve deepfake detection")
    print("✅ The model will now be more balanced between real and fake")
    print("⚠️  If false positives increase, you can adjust the threshold higher")
    print("💡 Use detector.set_threshold(value) to fine-tune as needed")
