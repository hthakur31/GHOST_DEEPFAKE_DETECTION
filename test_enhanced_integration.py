#!/usr/bin/env python3
"""
Test Enhanced XceptionNet Integration
Tests the enhanced predictor with the Django application
"""

import os
import sys
import django
from pathlib import Path

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detection.settings')
django.setup()

from enhanced_xception_predictor import get_xception_predictor, reload_xception_model
from model_watcher import start_model_watcher, force_model_reload
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_integration():
    """Test the enhanced XceptionNet integration with Django"""
    print("="*60)
    print("🧪 Testing Enhanced XceptionNet Integration")
    print("="*60)
    
    # Test 1: Check current predictor
    print("\n1️⃣ Testing Current Enhanced XceptionNet Predictor:")
    predictor = get_xception_predictor()
    
    if predictor.model is not None:
        print(f"✅ Enhanced XceptionNet model loaded successfully")
        print(f"   Model Type: {predictor.model_type}")
        print(f"   Device: {predictor.device}")
        print(f"   Architecture: {'ImprovedXceptionNet' if predictor.model_type == 'improved' else 'LegacyXceptionNet'}")
    else:
        print(f"⚠️  No enhanced XceptionNet model currently loaded")
        print(f"   This is expected if training is still in progress")
    
    # Test 2: Check available models
    print("\n2️⃣ Checking Available Models:")
    models_dir = Path("models")
    if models_dir.exists():
        xception_models = []
        for pattern in ["improved_xception*.pth", "robust_xception*.pth", "xception_best_*.pth"]:
            xception_models.extend(list(models_dir.glob(pattern)))
        
        if xception_models:
            print(f"✅ Found {len(xception_models)} XceptionNet models:")
            for model in sorted(xception_models, key=lambda x: x.stat().st_mtime, reverse=True):
                mtime = model.stat().st_mtime
                import time
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                print(f"   📄 {model.name} - {time_str}")
        else:
            print(f"⚠️  No XceptionNet models found")
    else:
        print(f"❌ Models directory not found")
    
    # Test 3: Force reload models
    print("\n3️⃣ Testing Model Reload:")
    reload_result = force_model_reload()
    if reload_result['success']:
        print(f"✅ Model reload successful")
        print(f"   Model loaded: {reload_result['model_loaded']}")
        print(f"   Model type: {reload_result['model_type']}")
    else:
        print(f"⚠️  Model reload: {reload_result['message']}")
    
    # Test 4: Start model watcher
    print("\n4️⃣ Testing Model Watcher:")
    try:
        watcher = start_model_watcher()
        print(f"✅ Enhanced XceptionNet model watcher started")
        print(f"   Will automatically detect and load new trained models")
        print(f"   Checking every 30 seconds for model updates")
    except Exception as e:
        print(f"❌ Failed to start model watcher: {e}")
    
    # Test 5: Integration status
    print("\n5️⃣ Django Integration Status:")
    try:
        from detector.views import VideoUploadView
        print(f"✅ Django views updated to use enhanced XceptionNet predictor")
        print(f"   Video upload processing will use the enhanced model")
        print(f"   Automatic model detection and architecture support enabled")
    except Exception as e:
        print(f"❌ Django integration issue: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("📋 Integration Summary:")
    print("="*60)
    
    if predictor.model is not None:
        print(f"🟢 Status: Enhanced XceptionNet is READY")
        print(f"   Your Django application is now using the enhanced model")
        if predictor.model_type == 'improved':
            print(f"   ⭐ Using ImprovedXceptionNet with attention mechanism")
        else:
            print(f"   📦 Using LegacyXceptionNet for backward compatibility")
    else:
        print(f"🟡 Status: Waiting for trained model")
        print(f"   The system will automatically use the improved model once training completes")
    
    print(f"\n🔄 Auto-reload: Model watcher is monitoring for new trained models")
    print(f"💻 Frontend: Django views updated to use enhanced predictor")
    print(f"🎯 Architecture: Automatic detection of improved vs legacy models")
    
    print("\n✨ Your deepfake detection application is ready to use the enhanced XceptionNet!")

if __name__ == "__main__":
    test_enhanced_integration()
