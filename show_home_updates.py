#!/usr/bin/env python3
"""
Summary of Home Page Statistics Updates
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')
django.setup()

from detector.models import DetectionResult, ModelPerformance
from django.db import models

def show_updated_statistics():
    """Show the updated statistics that now appear on the home page"""
    
    print("🎯 UPDATED HOME PAGE STATISTICS")
    print("=" * 60)
    
    # Calculate all the metrics that now show on the home page
    all_results = DetectionResult.objects.all().count()
    total_predictions = DetectionResult.objects.exclude(prediction='PROCESSING').count()
    fake_predictions = DetectionResult.objects.filter(prediction='FAKE').count()
    real_predictions = DetectionResult.objects.filter(prediction='REAL').count()
    error_predictions = DetectionResult.objects.filter(prediction='ERROR').count()
    processing_predictions = DetectionResult.objects.filter(prediction='PROCESSING').count()
    
    # Additional metrics now shown
    successful_predictions = fake_predictions + real_predictions
    success_rate = (successful_predictions / total_predictions * 100) if total_predictions > 0 else 0
    fake_percentage = (fake_predictions / successful_predictions * 100) if successful_predictions > 0 else 0
    real_percentage = (real_predictions / successful_predictions * 100) if successful_predictions > 0 else 0
    
    # Average confidence score for successful predictions
    successful_results = DetectionResult.objects.filter(
        prediction__in=['REAL', 'FAKE'],
        confidence_score__isnull=False
    )
    avg_confidence = successful_results.aggregate(
        avg_conf=models.Avg('confidence_score')
    )['avg_conf'] or 0
    
    # Processing time statistics
    completed_results = DetectionResult.objects.filter(
        prediction__in=['REAL', 'FAKE'],
        processing_time__isnull=False
    )
    avg_processing_time = completed_results.aggregate(
        avg_time=models.Avg('processing_time')
    )['avg_time'] or 0
    
    print("📊 MAIN STATISTICS CARDS:")
    print(f"   📹 Videos Analyzed Successfully: {successful_predictions}")
    print(f"      (Total uploaded: {all_results})")
    print(f"   ✅ Real Videos Detected: {real_predictions}")
    print(f"      ({real_predictions}/{successful_predictions} - {real_percentage:.1f}%)")
    print(f"   ⚠️  Deepfakes Detected: {fake_predictions}")
    print(f"      ({fake_predictions}/{successful_predictions} - {fake_percentage:.1f}%)")
    print(f"   📈 Deepfake Detection Rate: {fake_percentage:.1f}%")
    print(f"      (Avg Confidence: {avg_confidence * 100:.1f}%)")
    
    print("\n📊 ADDITIONAL METRICS ROW:")
    print(f"   📈 Success Rate: {success_rate:.1f}%")
    print(f"      ({error_predictions} errors)")
    print(f"   ⏱️  Avg Processing Time: {avg_processing_time:.1f}s")
    print(f"   🛡️  Avg Detection Confidence: {avg_confidence * 100:.1f}%")
    
    print("\n🔧 WHAT WAS FIXED:")
    print("   ✅ Fixed case sensitivity issue (Fake/Real → FAKE/REAL)")
    print("   ✅ Added comprehensive statistics calculation")
    print("   ✅ Separated successful vs total uploads")
    print("   ✅ Added success rate and error tracking")
    print("   ✅ Added average confidence and processing time")
    print("   ✅ Improved visual layout with better color coding")
    print("   ✅ Added detailed breakdowns with percentages")
    
    # Show model performance if available
    try:
        performance = ModelPerformance.objects.latest('last_updated')
        print(f"\n🤖 MODEL PERFORMANCE:")
        print(f"   Model: {performance.model_name} v{performance.model_version}")
        print(f"   Total Predictions: {performance.total_predictions}")
        print(f"   Accuracy: {performance.accuracy:.1f}%" if performance.accuracy else "   Accuracy: Not calculated")
    except ModelPerformance.DoesNotExist:
        print("\n🤖 MODEL PERFORMANCE: No data available")
    
    print(f"\n🎉 HOME PAGE NOW SHOWS ACCURATE DATA:")
    print(f"   • Real-time statistics from actual database")
    print(f"   • Proper distinction between successful vs failed analysis")
    print(f"   • Detailed breakdowns and percentages")
    print(f"   • Performance metrics and confidence scores")
    print(f"   • Visual improvements for better readability")

if __name__ == "__main__":
    show_updated_statistics()
