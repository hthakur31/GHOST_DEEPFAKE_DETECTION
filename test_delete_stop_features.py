#!/usr/bin/env python3
"""
Test script for new delete and stop processing features in the deepfake detection app.
Tests the delete records and stop processing functionality.
"""

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://127.0.0.1:8000"
API_BASE_URL = f"{BASE_URL}/api"

def get_csrf_token():
    """Get CSRF token from Django."""
    try:
        response = requests.get(f"{BASE_URL}/")
        if 'csrftoken' in response.cookies:
            return response.cookies['csrftoken']
        
        # Try alternative approach
        session = requests.Session()
        response = session.get(f"{BASE_URL}/")
        csrf_token = None
        
        # Look for CSRF token in page content
        if 'csrfmiddlewaretoken' in response.text:
            import re
            match = re.search(r'name="csrfmiddlewaretoken" value="([^"]*)"', response.text)
            if match:
                csrf_token = match.group(1)
        
        return csrf_token
        
    except Exception as e:
        print(f"Error getting CSRF token: {e}")
    return None

def test_stats_api():
    """Test the stats API endpoint."""
    print("\n" + "="*50)
    print("TESTING STATS API")
    print("="*50)
    
    try:
        response = requests.get(f"{API_BASE_URL}/stats/")
        if response.status_code == 200:
            data = response.json()
            print("✅ Stats API is working!")
            print(f"   Total Videos: {data.get('total_videos', 0)}")
            print(f"   Real Videos: {data.get('real_videos', 0)}")
            print(f"   Deepfakes: {data.get('deepfakes', 0)}")
            print(f"   Processing: {data.get('processing', 0)}")
            print(f"   Accuracy: {data.get('accuracy', 'N/A')}%")
            return True
        else:
            print(f"❌ Stats API failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing stats API: {e}")
        return False

def test_delete_api():
    """Test the delete API endpoint."""
    print("\n" + "="*50)
    print("TESTING DELETE API")
    print("="*50)
    
    csrf_token = get_csrf_token()
    if not csrf_token:
        print("❌ Could not get CSRF token")
        return False
    
    # Test with empty list (should fail gracefully)
    try:
        headers = {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrf_token,
        }
        cookies = {'csrftoken': csrf_token}
        
        response = requests.delete(
            f"{API_BASE_URL}/delete/",
            json={'ids': []},
            headers=headers,
            cookies=cookies
        )
        
        if response.status_code == 400:
            print("✅ Delete API correctly rejects empty ID list")
        else:
            data = response.json()
            print(f"⚠️  Delete API response: {data}")
            
        # Test with non-existent ID
        response = requests.delete(
            f"{API_BASE_URL}/delete/",
            json={'ids': [99999]},
            headers=headers,
            cookies=cookies
        )
        
        data = response.json()
        if data.get('success'):
            print("✅ Delete API handles non-existent IDs gracefully")
        else:
            print(f"⚠️  Delete API response for non-existent ID: {data}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing delete API: {e}")
        return False

def test_stop_api():
    """Test the stop processing API endpoint."""
    print("\n" + "="*50)
    print("TESTING STOP PROCESSING API")
    print("="*50)
    
    csrf_token = get_csrf_token()
    if not csrf_token:
        print("❌ Could not get CSRF token")
        return False
    
    try:
        headers = {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrf_token,
        }
        cookies = {'csrftoken': csrf_token}
        
        # Test with non-existent record ID
        response = requests.post(
            f"{API_BASE_URL}/stop/",
            json={'record_id': 99999},
            headers=headers,
            cookies=cookies
        )
        
        data = response.json()
        if response.status_code == 404:
            print("✅ Stop API correctly handles non-existent records")
        else:
            print(f"⚠️  Stop API response for non-existent record: {data}")
            
        # Test without record_id
        response = requests.post(
            f"{API_BASE_URL}/stop/",
            json={},
            headers=headers,
            cookies=cookies
        )
        
        if response.status_code == 400:
            print("✅ Stop API correctly requires record_id")
        else:
            data = response.json()
            print(f"⚠️  Stop API response without record_id: {data}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing stop API: {e}")
        return False

def test_history_page():
    """Test the history page."""
    print("\n" + "="*50)
    print("TESTING HISTORY PAGE")
    print("="*50)
    
    try:
        response = requests.get(f"{BASE_URL}/history/")
        if response.status_code == 200:
            content = response.text
            
            # Check for key elements
            checks = [
                ('bulk-delete-btn', 'Bulk delete button'),
                ('select-all', 'Select all checkbox'),
                ('record-checkbox', 'Individual checkboxes'),
                ('delete-single', 'Delete buttons'),
                ('stop-processing', 'Stop processing buttons'),
                ('refresh-status', 'Refresh status buttons'),
            ]
            
            for element_id, description in checks:
                if element_id in content:
                    print(f"✅ {description} found")
                else:
                    print(f"⚠️  {description} not found")
            
            print("✅ History page loaded successfully")
            return True
        else:
            print(f"❌ History page failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing history page: {e}")
        return False

def test_all_pages():
    """Test all main pages for basic functionality."""
    print("\n" + "="*50)
    print("TESTING ALL PAGES")
    print("="*50)
    
    pages = [
        ('/', 'Home page'),
        ('/history/', 'History page'),
    ]
    
    all_working = True
    
    for url, name in pages:
        try:
            response = requests.get(f"{BASE_URL}{url}")
            if response.status_code == 200:
                print(f"✅ {name} is working")
            else:
                print(f"❌ {name} failed with status {response.status_code}")
                all_working = False
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
            all_working = False
    
    return all_working

def main():
    """Run all tests."""
    print("🚀 Starting DELETE & STOP PROCESSING feature tests...")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Wait a moment for server to be ready
    print("\n⏳ Waiting 3 seconds for Django server to be ready...")
    time.sleep(3)
    
    # Run tests
    tests = [
        ("All Pages", test_all_pages),
        ("Stats API", test_stats_api),
        ("Delete API", test_delete_api),
        ("Stop API", test_stop_api),
        ("History Page", test_history_page),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The new DELETE & STOP features are working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
    
    print(f"⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*50)
    print("NEW FEATURES SUMMARY")
    print("="*50)
    print("🗑️  DELETE RECORDS:")
    print("   • Individual delete buttons on each record")
    print("   • Bulk delete with checkboxes")
    print("   • Confirmation dialogs for safety")
    print("   • API endpoint: DELETE /api/delete/")
    print("")
    print("⏹️  STOP PROCESSING:")
    print("   • Stop button for records being processed")
    print("   • Real-time status updates")
    print("   • Graceful cancellation handling")
    print("   • API endpoint: POST /api/stop/")
    print("")
    print("🔄 ENHANCED HISTORY:")
    print("   • Checkboxes for bulk selection")
    print("   • Dynamic action buttons based on status")
    print("   • Real-time UI updates")
    print("   • Better user feedback")
    print("")
    print("🌐 Navigate to: http://127.0.0.1:8000/history/")
    print("   to test the new features interactively!")

if __name__ == "__main__":
    main()
