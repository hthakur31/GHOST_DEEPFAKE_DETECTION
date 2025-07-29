#!/usr/bin/env python3
"""
Test the final fixed deepfake detection model
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

def test_final_model():
    """Test the final fixed model on deepfake videos"""
    print("🎉 TESTING FINAL FIXED MODEL")
    print("=" * 50)
    
    # Restart Django to reload the module
    from importlib import reload
    import detector.faceforensics_model
    reload(detector.faceforensics_model)
    from detector.faceforensics_model import FaceForensicsDetector
    
    # Initialize detector
    detector = FaceForensicsDetector()
    
    # Test multiple deepfake videos
    deepfake_folder = "G:/Deefake_detection_app/dataset/manipulated_sequences/Deepfakes/c23/videos"
    test_videos = [
        "033_097.mp4",
        "035_036.mp4", 
        "044_945.mp4"
    ]
    
    results = []
    for video_name in test_videos:
        video_path = os.path.join(deepfake_folder, video_name)
        if os.path.exists(video_path):
            print(f"\n📹 Testing DEEPFAKE: {video_name}")
            print("-" * 40)
            
            start_time = time.time()
            result = detector.detect_video(video_path)
            end_time = time.time()
            
            prediction = result.get('prediction', 'ERROR')
            confidence = result.get('confidence_score', 0)
            real_prob = result.get('real_probability', 0)
            fake_prob = result.get('fake_probability', 0)
            
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Real prob: {real_prob:.1f}%")
            print(f"   Fake prob: {fake_prob:.1f}%")
            print(f"   Time: {end_time - start_time:.1f}s")
            
            is_correct = prediction.lower() in ['deepfake', 'fake']
            if is_correct:
                print("   ✅ SUCCESS: Deepfake correctly identified!")
            else:
                print("   ❌ FAILED: Still predicting as real")
            
            results.append({
                'video': video_name,
                'prediction': prediction,
                'correct': is_correct,
                'fake_prob': fake_prob
            })
    
    # Summary
    print(f"\n📊 FINAL TEST SUMMARY:")
    print("=" * 40)
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    print(f"   Deepfake videos tested: {total_count}")
    print(f"   Correctly identified: {correct_count}")
    print(f"   Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 80:
        print("\n🎉 EXCELLENT: Model is now working well!")
        print("   ✅ Deepfake detection has been successfully fixed")
    elif accuracy >= 50:
        print("\n✅ GOOD: Model is improved but may need more tuning")
    else:
        print("\n⚠️  NEEDS MORE WORK: Model still struggling with deepfakes")
    
    return results

if __name__ == '__main__':
    print("🚀 FINAL DEEPFAKE DETECTION TEST")
    print("=" * 60)
    
    results = test_final_model()
    
    print(f"\n💡 CONCLUSION:")
    print("=" * 20)
    print("The model's decision logic has been simplified to use direct threshold comparison.")
    print("This should provide more reliable and predictable deepfake detection.")
