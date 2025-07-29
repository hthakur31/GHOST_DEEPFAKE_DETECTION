#!/usr/bin/env python3
"""
Quick test script for delete functionality
"""
import requests
import json

def get_csrf_token():
    """Get CSRF token from Django"""
    response = requests.get('http://127.0.0.1:8000/')
    if response.status_code == 200:
        # Extract CSRF token from cookies
        csrf_token = response.cookies.get('csrftoken')
        return csrf_token
    return None

def test_delete_record(record_id):
    """Test deleting a specific record"""
    csrf_token = get_csrf_token()
    if not csrf_token:
        print("❌ Could not get CSRF token")
        return False
    
    print(f"🔑 CSRF Token: {csrf_token}")
    
    # Test delete API
    headers = {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrf_token,
        'Referer': 'http://127.0.0.1:8000/history/'
    }
    
    data = {'ids': [record_id]}
    
    print(f"🗑️  Attempting to delete record: {record_id}")
    response = requests.delete('http://127.0.0.1:8000/api/delete/', 
                              headers=headers, 
                              json=data)
    
    print(f"📡 Response Status: {response.status_code}")
    try:
        response_data = response.json()
        print(f"📄 Response: {json.dumps(response_data, indent=2)}")
        
        if response_data.get('success'):
            print("✅ Delete successful!")
            return True
        else:
            print(f"❌ Delete failed: {response_data.get('error', 'Unknown error')}")
            return False
    except:
        print(f"❌ Could not parse response: {response.text}")
        return False

if __name__ == '__main__':
    # Test with the record we just created
    record_id = '85ac780c-1e3a-4833-a4ea-562e20992265'
    test_delete_record(record_id)
