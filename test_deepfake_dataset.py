#!/usr/bin/env python3
"""
Test the model specifically on known deepfake videos from the dataset
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

def test_deepfake_videos():
    """Test known deepfake videos from dataset"""
    print("🎭 TESTING KNOWN DEEPFAKE VIDEOS")
    print("=" * 50)
    
    # Initialize detector
    detector = FaceForensicsDetector()
    
    # Known deepfake videos from dataset
    deepfake_folder = "G:/Deefake_detection_app/dataset/manipulated_sequences/Deepfakes/c23/videos"
    test_videos = [
        "033_097.mp4",
        "035_036.mp4", 
        "044_945.mp4",
        "055_147.mp4",
        "097_033.mp4"
    ]
    
    results = []
    for video_name in test_videos:
        video_path = os.path.join(deepfake_folder, video_name)
        if os.path.exists(video_path):
            print(f"\n📹 Testing DEEPFAKE video: {video_name}")
            print("-" * 40)
            
            start_time = time.time()
            try:
                # Test with current model
                result = detector.detect_video(video_path)
                end_time = time.time()
                
                prediction = result['prediction']
                confidence = result['confidence']
                probabilities = result.get('probabilities', {})
                threshold = result.get('threshold_used', 'unknown')
                
                print(f"   Prediction: {prediction}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Real prob: {probabilities.get('real', 0)*100:.1f}%")
                print(f"   Fake prob: {probabilities.get('fake', 0)*100:.1f}%")
                print(f"   Threshold: {threshold}")
                print(f"   Time: {end_time - start_time:.1f}s")
                
                # Check if correctly identified as deepfake
                is_correct = (prediction.lower() in ['deepfake', 'fake'])
                if is_correct:
                    print("   ✅ CORRECT: Properly identified as deepfake")
                else:
                    print("   ❌ FALSE NEGATIVE: Deepfake predicted as real!")
                
                results.append({
                    'video': video_name,
                    'prediction': prediction,
                    'confidence': confidence,
                    'is_correct': is_correct,
                    'probabilities': probabilities
                })
                
            except Exception as e:
                print(f"   ❌ Error processing {video_name}: {str(e)}")
                results.append({
                    'video': video_name,
                    'prediction': 'ERROR',
                    'confidence': 0,
                    'is_correct': False,
                    'error': str(e)
                })
        else:
            print(f"❌ Video not found: {video_path}")
    
    # Summary
    print("\n📊 DEEPFAKE DETECTION SUMMARY:")
    print("=" * 50)
    correct_count = sum(1 for r in results if r['is_correct'])
    total_count = len(results)
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    print(f"   Total deepfake videos tested: {total_count}")
    print(f"   Correctly identified: {correct_count}")
    print(f"   False negatives: {total_count - correct_count}")
    print(f"   Accuracy: {accuracy:.1f}%")
    
    if accuracy < 50:
        print("\n🚨 CRITICAL ISSUE: Model is failing to detect deepfakes!")
        print("   The model is too conservative and needs threshold adjustment.")
        print("\n💡 RECOMMENDED FIXES:")
        print("   1. Lower the detection threshold (current appears to be ~0.65)")
        print("   2. Adjust the balanced threshold logic")
        print("   3. Consider ensemble model calibration")
    elif accuracy < 80:
        print("\n⚠️  WARNING: Model accuracy is below optimal level")
        print("   Consider fine-tuning the threshold or model parameters")
    else:
        print("\n✅ Model is performing well on deepfake detection")
    
    return results

def test_with_different_thresholds():
    """Test the same deepfake video with different thresholds"""
    print("\n🎛️  THRESHOLD SENSITIVITY TEST")
    print("=" * 50)
    
    detector = FaceForensicsDetector()
    test_video = "G:/Deefake_detection_app/dataset/manipulated_sequences/Deepfakes/c23/videos/033_097.mp4"
    
    if not os.path.exists(test_video):
        print("❌ Test video not found")
        return
    
    # Test different thresholds
    test_thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    
    print(f"📹 Testing with video: 033_097.mp4 (KNOWN DEEPFAKE)")
    print("-" * 60)
    
    for threshold in test_thresholds:
        try:
            # Temporarily modify threshold (if the detector supports it)
            result = detector.detect_video(test_video)
            confidence = result['confidence']
            probabilities = result.get('probabilities', {})
            
            # Simulate different threshold decisions
            fake_prob = probabilities.get('fake', 0)
            would_be_fake = fake_prob > threshold
            
            status = "✅ CORRECT (Fake)" if would_be_fake else "❌ WRONG (Real)"
            print(f"   Threshold {threshold:.2f}: Fake={fake_prob:.3f} → {'FAKE' if would_be_fake else 'REAL'} {status}")
            
        except Exception as e:
            print(f"   Threshold {threshold:.2f}: Error - {str(e)}")
    
    print(f"\n💡 OPTIMAL THRESHOLD ANALYSIS:")
    print(f"   Current fake probability: {probabilities.get('fake', 0):.3f}")
    print(f"   For correct detection, threshold should be < {probabilities.get('fake', 0):.3f}")

if __name__ == '__main__':
    print("🚀 DEEPFAKE DATASET TESTING")
    print("=" * 60)
    
    # Test known deepfake videos
    results = test_deepfake_videos()
    
    # Test threshold sensitivity
    test_with_different_thresholds()
    
    print("\n✅ TESTING COMPLETE!")
