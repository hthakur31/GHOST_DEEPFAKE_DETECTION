#!/usr/bin/env python3
"""
Analyze detection accuracy and identify false positive issues
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')
django.setup()

from detector.models import DetectionResult, FrameAnalysis

def analyze_detection_accuracy():
    """Analyze detection results to identify false positive patterns"""
    
    print("🔍 DETECTION ACCURACY ANALYSIS")
    print("=" * 50)
    
    # Get all completed results
    all_results = DetectionResult.objects.filter(
        prediction__in=['REAL', 'FAKE']
    ).order_by('-created_at')
    
    print(f"Total completed detections: {all_results.count()}")
    
    # Show recent results with details
    print("\n📊 RECENT DETECTION RESULTS:")
    print("-" * 80)
    print(f"{'Filename':<30} {'Prediction':<10} {'Confidence':<12} {'Model':<20} {'Date'}")
    print("-" * 80)
    
    for result in all_results[:15]:  # Show last 15 results
        confidence_str = f"{result.confidence_score:.3f}" if result.confidence_score else "N/A"
        model_str = result.model_used[:18] if result.model_used else "Unknown"
        date_str = result.created_at.strftime('%m-%d %H:%M')
        filename = result.original_filename[:28] if result.original_filename else "Unknown"
        
        print(f"{filename:<30} {result.prediction:<10} {confidence_str:<12} {model_str:<20} {date_str}")
    
    # Analyze confidence distribution
    fake_results = all_results.filter(prediction='FAKE')
    real_results = all_results.filter(prediction='REAL')
    
    print(f"\n📈 CONFIDENCE SCORE ANALYSIS:")
    print(f"FAKE predictions: {fake_results.count()}")
    if fake_results.exists():
        fake_confidences = [r.confidence_score for r in fake_results if r.confidence_score]
        if fake_confidences:
            avg_fake_conf = sum(fake_confidences) / len(fake_confidences)
            min_fake_conf = min(fake_confidences)
            max_fake_conf = max(fake_confidences)
            print(f"  Average confidence: {avg_fake_conf:.3f}")
            print(f"  Range: {min_fake_conf:.3f} - {max_fake_conf:.3f}")
    
    print(f"\nREAL predictions: {real_results.count()}")
    if real_results.exists():
        real_confidences = [r.confidence_score for r in real_results if r.confidence_score]
        if real_confidences:
            avg_real_conf = sum(real_confidences) / len(real_confidences)
            min_real_conf = min(real_confidences)
            max_real_conf = max(real_confidences)
            print(f"  Average confidence: {avg_real_conf:.3f}")
            print(f"  Range: {min_real_conf:.3f} - {max_real_conf:.3f}")
    
    # Check for problematic patterns
    print(f"\n⚠️  POTENTIAL ISSUES DETECTED:")
    
    # Check if confidence scores are consistently around 0.5 (uncertain)
    uncertain_results = all_results.filter(
        confidence_score__gte=0.45, 
        confidence_score__lte=0.55
    )
    if uncertain_results.count() > all_results.count() * 0.7:  # More than 70% uncertain
        print(f"   🔴 High uncertainty: {uncertain_results.count()}/{all_results.count()} results have confidence ~0.5")
        print(f"      This suggests the model is not well-trained or has issues")
    
    # Check if mostly predicting FAKE (bias toward fake)
    fake_ratio = fake_results.count() / all_results.count() if all_results.count() > 0 else 0
    if fake_ratio > 0.8:  # More than 80% fake
        print(f"   🔴 Bias toward FAKE: {fake_ratio*100:.1f}% of predictions are FAKE")
        print(f"      This suggests false positive bias - real videos being classified as fake")
    
    # Check for consistent low confidence
    low_conf_results = all_results.filter(confidence_score__lt=0.6)
    if low_conf_results.count() > all_results.count() * 0.8:  # More than 80% low confidence
        print(f"   🔴 Low confidence pattern: {low_conf_results.count()}/{all_results.count()} have confidence < 0.6")
        print(f"      This suggests model uncertainty or poor training")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    if fake_ratio > 0.8:
        print(f"   1. 🔧 Adjust decision threshold (currently seems to favor FAKE)")
        print(f"   2. 🔧 Retrain model with more balanced dataset")
        print(f"   3. 🔧 Review preprocessing pipeline for bias")
    
    if uncertain_results.count() > all_results.count() * 0.7:
        print(f"   1. 🔧 Model needs more training or better architecture")
        print(f"   2. 🔧 Check if training data quality is sufficient")
        print(f"   3. 🔧 Consider ensemble methods or confidence calibration")
    
    return {
        'total_results': all_results.count(),
        'fake_results': fake_results.count(),
        'real_results': real_results.count(),
        'fake_ratio': fake_ratio,
        'uncertain_count': uncertain_results.count()
    }

if __name__ == "__main__":
    stats = analyze_detection_accuracy()
    
    print(f"\n📋 SUMMARY:")
    print(f"   Total detections: {stats['total_results']}")
    print(f"   FAKE predictions: {stats['fake_results']} ({stats['fake_ratio']*100:.1f}%)")
    print(f"   REAL predictions: {stats['real_results']} ({(1-stats['fake_ratio'])*100:.1f}%)")
    print(f"   Uncertain predictions: {stats['uncertain_count']}")
    
    if stats['fake_ratio'] > 0.8:
        print(f"\n🚨 CRITICAL: Strong bias toward FAKE classification detected!")
        print(f"   Your model is likely producing false positives for real videos.")
