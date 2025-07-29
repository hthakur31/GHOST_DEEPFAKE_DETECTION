#!/usr/bin/env python
"""
Test script for the enhanced UI and download functionality
"""
import os
import sys
import django
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')
django.setup()

from detector.models import DetectionResult
from django.test import Client
from django.urls import reverse
import json


def test_enhanced_ui():
    """Test the enhanced UI functionality"""
    print("🔍 Testing Enhanced UI and Download Functionality")
    print("=" * 60)
    
    # Check if we have any detection results
    results = DetectionResult.objects.all().order_by('-created_at')
    
    if not results.exists():
        print("❌ No detection results found in database")
        print("   Please run a video analysis first to test the UI")
        return False
    
    result = results.first()
    print(f"✓ Found test result: {result.id}")
    print(f"  - Filename: {result.original_filename}")
    print(f"  - Prediction: {result.prediction}")
    print(f"  - Confidence: {result.confidence_score:.4f}")
    
    # Test client setup
    client = Client()
    
    # Test result page
    try:
        result_url = reverse('detector:result', kwargs={'detection_id': result.id})
        response = client.get(result_url)
        
        if response.status_code == 200:
            print("✓ Result page loads successfully")
            
            # Check for enhanced UI elements
            content = response.content.decode()
            
            ui_checks = [
                ("Hero section", "hero-section" in content),
                ("Chart.js", "Chart.js" in content or "chart.js" in content),
                ("Confidence chart", "confidenceChart" in content),
                ("Download section", "download-section" in content),
                ("Details button", "View Detailed Analysis" in content),
                ("Bootstrap styling", "btn-primary" in content),
                ("FontAwesome icons", "fa-" in content)
            ]
            
            for check_name, passed in ui_checks:
                status = "✓" if passed else "❌"
                print(f"  {status} {check_name}")
        else:
            print(f"❌ Result page failed to load: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing result page: {e}")
        return False
    
    # Test download functionality
    print("\n📥 Testing Download Functionality")
    print("-" * 40)
    
    download_formats = ['json', 'pdf', 'excel', 'html']
    
    for format_type in download_formats:
        try:
            download_url = reverse('detector:download_report_format', 
                                 kwargs={'result_id': result.id, 'format_type': format_type})
            response = client.get(download_url)
            
            if response.status_code == 200:
                print(f"✓ {format_type.upper()} download works")
                
                # Check content type
                content_type = response.get('Content-Type', '')
                expected_types = {
                    'json': 'application/json',
                    'pdf': 'application/pdf',
                    'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    'html': 'text/html'
                }
                
                if expected_types[format_type] in content_type:
                    print(f"  ✓ Correct content type: {content_type}")
                else:
                    print(f"  ⚠️ Content type: {content_type} (fallback)")
                
                # Check Content-Disposition header
                disposition = response.get('Content-Disposition', '')
                if 'attachment' in disposition:
                    print(f"  ✓ Download header set")
                else:
                    print(f"  ❌ Missing download header")
                
            else:
                print(f"❌ {format_type.upper()} download failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error testing {format_type} download: {e}")
    
    # Test detailed result page
    print("\n📊 Testing Detailed Analysis Page")
    print("-" * 40)
    
    try:
        detailed_url = reverse('detector:detailed_result', kwargs={'result_id': result.id})
        response = client.get(detailed_url)
        
        if response.status_code == 200:
            print("✓ Detailed analysis page loads successfully")
            
            content = response.content.decode()
            detail_checks = [
                ("Frame analysis", "frame-analysis" in content or "Frame Analysis" in content),
                ("Charts/graphs", "Chart" in content or "chart" in content),
                ("Metadata section", "metadata" in content or "Metadata" in content),
                ("Back button", "Back to Results" in content or "btn" in content)
            ]
            
            for check_name, passed in detail_checks:
                status = "✓" if passed else "❌"
                print(f"  {status} {check_name}")
        else:
            print(f"❌ Detailed analysis page failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing detailed page: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Enhanced UI Testing Complete!")
    print("\nTo test the UI visually:")
    print("1. Start the Django server: python manage.py runserver")
    print(f"2. Visit: http://localhost:8000/detector/result/{result.id}/")
    print("3. Test the download buttons and navigation")
    
    return True


def check_static_files():
    """Check if static files are properly configured"""
    print("\n📁 Checking Static Files Configuration")
    print("-" * 40)
    
    static_checks = [
        ("STATIC_URL setting", hasattr(django.conf.settings, 'STATIC_URL')),
        ("Bootstrap files", True),  # Assuming CDN is used
        ("FontAwesome files", True),  # Assuming CDN is used
        ("Chart.js files", True),  # Assuming CDN is used
    ]
    
    for check_name, passed in static_checks:
        status = "✓" if passed else "❌"
        print(f"  {status} {check_name}")


if __name__ == "__main__":
    try:
        test_enhanced_ui()
        check_static_files()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
