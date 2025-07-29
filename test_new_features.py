#!/usr/bin/env python
"""
Test script for the new API endpoints and real-time statistics
"""
import requests
import json
from datetime import datetime

# Base URL
base_url = "http://127.0.0.1:8000"

def test_stats_api():
    """Test the new stats API endpoint"""
    print("=== Testing Stats API ===")
    try:
        response = requests.get(f"{base_url}/api/stats/")
        if response.status_code == 200:
            data = response.json()
            print("✅ Stats API working!")
            print(f"Status: {data['status']}")
            print(f"Timestamp: {data['timestamp']}")
            
            stats = data['stats']
            print("\n📊 Current Statistics:")
            print(f"  Total Successful Predictions: {stats['total_predictions']}")
            print(f"  Real Videos: {stats['real_predictions']} ({stats['real_percentage']:.1f}%)")
            print(f"  Fake Videos: {stats['fake_predictions']} ({stats['fake_percentage']:.1f}%)")
            print(f"  Success Rate: {stats['success_rate']:.1f}%")
            print(f"  Average Confidence: {stats['avg_confidence']:.1f}%")
            print(f"  Model Accuracy: {stats.get('model_accuracy', 'N/A') if stats.get('model_accuracy') else 'N/A'}")
            print(f"  Average Processing Time: {stats['avg_processing_time']:.2f}s")
            print(f"  Currently Processing: {stats['processing_predictions']}")
            print(f"  Total Uploaded: {stats['all_videos_uploaded']}")
            
        else:
            print(f"❌ Stats API failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error testing stats API: {e}")

def test_recent_api():
    """Test the recent detections API"""
    print("\n=== Testing Recent Detections API ===")
    try:
        response = requests.get(f"{base_url}/api/recent/?limit=5")
        if response.status_code == 200:
            data = response.json()
            print("✅ Recent API working!")
            print(f"Results: {len(data)}")
            
            for i, result in enumerate(data[:3], 1):
                print(f"\n{i}. {result['original_filename']}")
                print(f"   Status: {result['prediction']}")
                if result['confidence_score']:
                    print(f"   Confidence: {result['confidence_score']:.3f}")
                print(f"   Date: {result['created_at']}")
                
        else:
            print(f"❌ Recent API failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error testing recent API: {e}")

def check_pages():
    """Check if the new pages are accessible"""
    print("\n=== Testing Web Pages ===")
    
    # Test home page
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Home page accessible")
        else:
            print(f"❌ Home page failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error accessing home page: {e}")
    
    # Test history page
    try:
        response = requests.get(f"{base_url}/history/")
        if response.status_code == 200:
            print("✅ History page accessible")
        else:
            print(f"❌ History page failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error accessing history page: {e}")

def main():
    print("🧪 Testing New Features - Real-time Stats & History")
    print("=" * 50)
    
    test_stats_api()
    test_recent_api()
    check_pages()
    
    print("\n" + "=" * 50)
    print("✨ Test completed!")
    print("\n🔄 The home page now updates statistics every 10 seconds automatically")
    print("📜 The history page shows all detections with filtering and pagination")
    print("🚀 Navigate to http://127.0.0.1:8000/ to see the live updates!")

if __name__ == "__main__":
    main()
