#!/usr/bin/env python3
"""
Check current database statistics for the home page
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')
django.setup()

from detector.models import DetectionResult, ModelPerformance

def check_current_stats():
    """Check current database statistics"""
    
    print("🔍 Current Database Statistics")
    print("=" * 40)
    
    # Total detection results
    total_results = DetectionResult.objects.count()
    print(f"Total Detection Results: {total_results}")
    
    # By prediction status
    print("\nBy Prediction Status:")
    for status, label in DetectionResult.DETECTION_CHOICES:
        count = DetectionResult.objects.filter(prediction=status).count()
        print(f"  {label} ({status}): {count}")
    
    # Calculate current statistics that would appear on home page
    total_predictions = DetectionResult.objects.exclude(prediction='PROCESSING').count()
    fake_predictions = DetectionResult.objects.filter(prediction='FAKE').count()
    real_predictions = DetectionResult.objects.filter(prediction='REAL').count()
    error_predictions = DetectionResult.objects.filter(prediction='ERROR').count()
    
    print(f"\nHome Page Statistics:")
    print(f"  Total Videos Analyzed: {total_predictions}")
    print(f"  Real Videos Detected: {real_predictions}")
    print(f"  Deepfakes Detected: {fake_predictions}")
    print(f"  Errors: {error_predictions}")
    print(f"  Deepfake Rate: {(fake_predictions / total_predictions * 100) if total_predictions > 0 else 0:.1f}%")
    
    # Check model performance
    try:
        performance = ModelPerformance.objects.latest('last_updated')
        print(f"\nModel Performance:")
        print(f"  Model: {performance.model_name} v{performance.model_version}")
        print(f"  Total Predictions: {performance.total_predictions}")
        print(f"  Accuracy: {performance.accuracy:.1f}%" if performance.accuracy else "  Accuracy: Not calculated")
    except ModelPerformance.DoesNotExist:
        print("\nNo Model Performance data found")
    
    # Recent results
    recent_results = DetectionResult.objects.exclude(prediction='PROCESSING').order_by('-created_at')[:5]
    print(f"\nRecent Results (last 5):")
    for result in recent_results:
        confidence_str = f"{result.confidence_score:.3f}" if result.confidence_score else "N/A"
        print(f"  {result.original_filename[:30]:<30} | {result.prediction:<10} | {confidence_str:<8} | {result.created_at.strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    check_current_stats()
