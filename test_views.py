#!/usr/bin/env python
"""
Simple view test to ensure templates work with views
"""
import os
import django
from django.test import Client
from django.conf import settings

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')
django.setup()

def test_views():
    """Test that views render correctly with templates"""
    client = Client()
    
    print("🌐 Testing Django Views with Templates...")
    
    try:
        # Test home page
        print("  Testing home page...", end=' ')
        response = client.get('/')
        if response.status_code == 200:
            print("✅ PASS")
        else:
            print(f"❌ FAIL (Status: {response.status_code})")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == '__main__':
    success = test_views()
    if success:
        print("\n🎉 All view tests PASSED!")
    else:
        print("\n❌ View tests FAILED!")
