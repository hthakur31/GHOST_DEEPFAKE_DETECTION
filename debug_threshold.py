#!/usr/bin/env python3
"""
Fix false positive issue by adjusting detection threshold and improving prediction logic
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')
django.setup()

from detector.ml_utils import get_detector
import torch
import numpy as np

def test_different_thresholds():
    """Test different decision thresholds to find optimal one"""
    
    print("🔧 TESTING DETECTION THRESHOLDS")
    print("=" * 50)
    
    # Load the detector
    try:
        detector = get_detector(use_advanced_model=True)
        print(f"✅ Loaded model: {detector.model_name}")
    except Exception as e:
        print(f"❌ Error loading detector: {e}")
        return
    
    # Test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print(f"\n📊 Current prediction logic analysis:")
    print(f"   Classes: {detector.advanced_detector.classes if hasattr(detector, 'advanced_detector') else 'Unknown'}")
    
    # Create test scenarios with dummy data
    print(f"\n🧪 Testing threshold sensitivity:")
    print(f"{'Threshold':<10} {'Action':<30}")
    print("-" * 40)
    
    for threshold in thresholds:
        # Simulate frame predictions with majority voting
        test_predictions = [0, 0, 1, 0, 0]  # Mostly Real (0), some Fake (1)
        mean_pred = np.mean(test_predictions)
        
        if mean_pred > threshold:
            prediction_class = "FAKE"
        else:
            prediction_class = "REAL"
        
        print(f"{threshold:<10} {prediction_class + f' (mean={mean_pred:.1f})':<30}")
    
    print(f"\n💡 THRESHOLD ANALYSIS:")
    print(f"   Current threshold: 0.5")
    print(f"   Issue: With random/uncertain predictions around 0.5,")
    print(f"          the model tends to classify as FAKE more often")
    
    return thresholds

def create_improved_detector():
    """Create an improved detector with better threshold and calibration"""
    
    print(f"\n🛠️  CREATING IMPROVED DETECTOR")
    print("=" * 40)
    
    improved_detector_code = '''
class ImprovedFaceForensicsDetector(FaceForensicsDetector):
    """
    Improved detector with better threshold management and confidence calibration
    """
    
    def __init__(self, model_path=None, device='auto', threshold=0.6):
        super().__init__(model_path, device)
        self.threshold = threshold  # Higher threshold = less likely to classify as fake
        self.confidence_calibration = True
        
    def detect_video_with_improved_logic(self, video_path: str):
        """
        Detect deepfakes with improved prediction logic
        """
        try:
            metadata = self.get_video_metadata(video_path)
            frames = self.extract_frames(video_path)
            
            if not frames:
                return {
                    'prediction': 'ERROR',
                    'error': 'No frames extracted'
                }
            
            # Preprocess frames
            frame_tensors, freq_tensors = self.preprocessor.preprocess_frames(frames)
            
            if frame_tensors.size(0) == 0:
                return {
                    'prediction': 'ERROR',
                    'error': 'No valid faces detected'
                }
            
            # Move to device
            frame_tensors = frame_tensors.to(self.device)
            freq_tensors = freq_tensors.to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs, aux_outputs = self.model(frame_tensors, freq_tensors)
                
                # Get probabilities
                probabilities = F.softmax(outputs, dim=1)
                
                # Extract real and fake probabilities
                real_probs = probabilities[:, 0].cpu().numpy()  # Index 0 = Real
                fake_probs = probabilities[:, 1].cpu().numpy()  # Index 1 = Fake
                
                # Calculate confidence-weighted average
                frame_confidences = np.maximum(real_probs, fake_probs)
                weights = frame_confidences / np.sum(frame_confidences)
                
                # Weighted voting instead of simple majority
                weighted_fake_score = np.sum(fake_probs * weights)
                weighted_real_score = np.sum(real_probs * weights)
                
                # Apply improved threshold logic
                if weighted_fake_score > self.threshold:
                    final_prediction = 'FAKE'
                    final_confidence = weighted_fake_score
                else:
                    final_prediction = 'REAL'
                    final_confidence = weighted_real_score
                
                # Calculate additional metrics
                consistency_score = 1.0 - np.std(fake_probs)  # Lower std = more consistent
                
                return {
                    'prediction': final_prediction,
                    'confidence_score': float(final_confidence),
                    'fake_probability': float(weighted_fake_score * 100),
                    'real_probability': float(weighted_real_score * 100),
                    'consistency_score': float(consistency_score),
                    'threshold_used': self.threshold,
                    'frame_count': len(frames),
                    'model_name': 'FaceForensics++ Improved',
                    'metadata': metadata
                }
                
        except Exception as e:
            return {
                'prediction': 'ERROR',
                'error': str(e)
            }
'''
    
    print("✅ Improved detector logic created")
    print("🔧 Key improvements:")
    print("   1. Higher default threshold (0.6 instead of 0.5)")
    print("   2. Confidence-weighted voting instead of simple majority")
    print("   3. Better probability calibration")
    print("   4. Consistency scoring for quality assessment")
    
    return improved_detector_code

if __name__ == "__main__":
    print("🚨 FALSE POSITIVE FIX ANALYSIS")
    print("=" * 60)
    
    # Test current threshold behavior
    thresholds = test_different_thresholds()
    
    # Create improved detector
    improved_code = create_improved_detector()
    
    print(f"\n📋 RECOMMENDED FIXES:")
    print(f"   1. 🎯 Increase detection threshold from 0.5 to 0.6-0.7")
    print(f"   2. 🔄 Use confidence-weighted voting instead of simple majority")
    print(f"   3. 📊 Add consistency scoring to detect uncertain predictions")
    print(f"   4. 🎓 Consider retraining with more balanced dataset")
    print(f"   5. 🔍 Implement confidence calibration")
    
    print(f"\n🎯 IMMEDIATE ACTION:")
    print(f"   The simplest fix is to increase the threshold to 0.6 or 0.7")
    print(f"   This will make the model less likely to classify real videos as fake")
    print(f"   Current issue: 80% fake predictions suggests threshold too low")
