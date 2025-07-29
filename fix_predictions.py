#!/usr/bin/env python3
"""
Check and fix prediction values in the database
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')
django.setup()

from detector.models import DetectionResult

def check_and_fix_predictions():
    """Check and fix prediction values in the database"""
    
    print("🔍 Checking Prediction Values in Database")
    print("=" * 50)
    
    # Check all unique prediction values
    all_results = DetectionResult.objects.all()
    unique_predictions = set(result.prediction for result in all_results)
    
    print(f"Unique prediction values found: {unique_predictions}")
    
    # Show detailed breakdown
    print("\nDetailed breakdown:")
    for prediction in unique_predictions:
        count = DetectionResult.objects.filter(prediction=prediction).count()
        print(f"  '{prediction}': {count} results")
    
    # Check for case issues and fix them
    print("\n🔧 Fixing Case Issues...")
    
    # Fix "Fake" -> "FAKE"
    fake_lowercase = DetectionResult.objects.filter(prediction='Fake')
    if fake_lowercase.exists():
        count = fake_lowercase.count()
        fake_lowercase.update(prediction='FAKE')
        print(f"✅ Fixed {count} 'Fake' -> 'FAKE'")
    
    # Fix "Real" -> "REAL"
    real_lowercase = DetectionResult.objects.filter(prediction='Real')
    if real_lowercase.exists():
        count = real_lowercase.count()
        real_lowercase.update(prediction='REAL')
        print(f"✅ Fixed {count} 'Real' -> 'REAL'")
    
    # Fix other potential issues
    for old_val, new_val in [
        ('fake', 'FAKE'),
        ('real', 'REAL'),
        ('error', 'ERROR'),
        ('processing', 'PROCESSING'),
        ('Error', 'ERROR'),
        ('Processing', 'PROCESSING')
    ]:
        results = DetectionResult.objects.filter(prediction=old_val)
        if results.exists():
            count = results.count()
            results.update(prediction=new_val)
            print(f"✅ Fixed {count} '{old_val}' -> '{new_val}'")
    
    print("\n📊 Updated Statistics:")
    
    # Calculate updated statistics
    total_predictions = DetectionResult.objects.exclude(prediction='PROCESSING').count()
    fake_predictions = DetectionResult.objects.filter(prediction='FAKE').count()
    real_predictions = DetectionResult.objects.filter(prediction='REAL').count()
    error_predictions = DetectionResult.objects.filter(prediction='ERROR').count()
    processing_predictions = DetectionResult.objects.filter(prediction='PROCESSING').count()
    
    print(f"  Total Videos Analyzed: {total_predictions}")
    print(f"  Real Videos Detected: {real_predictions}")
    print(f"  Deepfakes Detected: {fake_predictions}")
    print(f"  Errors: {error_predictions}")
    print(f"  Currently Processing: {processing_predictions}")
    print(f"  Deepfake Rate: {(fake_predictions / total_predictions * 100) if total_predictions > 0 else 0:.1f}%")

if __name__ == "__main__":
    check_and_fix_predictions()
