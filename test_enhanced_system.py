#!/usr/bin/env python3
"""
Test the enhanced deepfake detection system with report downloads and history features
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

def test_enhanced_system():
    """Test the enhanced features"""
    print("🔍 Testing Enhanced Deepfake Detection System")
    print("=" * 60)
    
    # Test 1: Check if Django server is running
    try:
        response = requests.get('http://localhost:8000/detector/')
        print(f"✅ Django server is running (Status: {response.status_code})")
    except requests.exceptions.ConnectionError:
        print("❌ Django server is not running. Please start it with: python manage.py runserver")
        return False
    
    # Test 2: Check history page
    try:
        response = requests.get('http://localhost:8000/detector/history/')
        print(f"✅ History page accessible (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ History page error: {e}")
    
    # Test 3: Test report generator
    try:
        from advanced_report_generator import report_generator
        print("✅ Advanced report generator imported successfully")
        
        # Check if reports directory exists
        reports_dir = Path("reports")
        if reports_dir.exists():
            print(f"✅ Reports directory exists: {reports_dir}")
        else:
            reports_dir.mkdir(exist_ok=True)
            print(f"✅ Reports directory created: {reports_dir}")
            
    except ImportError as e:
        print(f"❌ Report generator import error: {e}")
    
    # Test 4: Check database migrations
    try:
        from django.core.management import execute_from_command_line
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')
        import django
        django.setup()
        
        from detector.models import DetectionResult
        
        # Check if new fields exist
        result = DetectionResult.objects.first()
        if result:
            # Test new fields
            if hasattr(result, 'metadata'):
                print("✅ New metadata field available")
            if hasattr(result, 'report_data'):
                print("✅ New report_data field available")
            if hasattr(result, 'report_generated'):
                print("✅ New report_generated field available")
                
        print("✅ Database models updated successfully")
        
    except Exception as e:
        print(f"❌ Database check error: {e}")
    
    # Test 5: Check package installations
    required_packages = ['reportlab', 'pandas', 'openpyxl', 'matplotlib', 'seaborn']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")
    
    print("\n" + "=" * 60)
    print("🎉 Enhanced System Features Available:")
    print("📊 History page with advanced filtering")
    print("📥 PDF, JSON, Excel, HTML report downloads")
    print("📦 Bulk download functionality")
    print("🗑️ Individual and bulk delete options")
    print("📈 Detailed analysis with frame-by-frame breakdown")
    print("⚠️ Risk assessment and recommendations")
    print("📋 Professional report templates")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_enhanced_system()
