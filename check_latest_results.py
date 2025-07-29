#!/usr/bin/env python
import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')
django.setup()

from detector.models import DetectionResult

print("=== LATEST COMPLETED DETECTIONS ===")
results = DetectionResult.objects.filter(prediction__in=['REAL', 'FAKE']).order_by('-created_at')[:10]

for result in results:
    print(f"File: {result.original_filename}")
    print(f"Prediction: {result.prediction}")
    if result.confidence_score:
        print(f"Confidence: {result.confidence_score:.3f}")
        print(f"Real Prob: {result.real_probability:.3f}")
        print(f"Fake Prob: {result.fake_probability:.3f}")
    print(f"Model Version: {result.model_version or 'old'}")
    print(f"Method: {result.detection_method or 'old'}")
    print(f"Created: {result.created_at}")
    print("---")

print(f"\nTotal results found: {results.count()}")

# Check current processing
processing = DetectionResult.objects.filter(prediction='PROCESSING').count()
print(f"Currently processing: {processing}")
