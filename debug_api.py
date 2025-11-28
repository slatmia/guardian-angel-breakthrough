#!/usr/bin/env python3
"""
Guardian API Debug Test
Find out what's causing the Internal Server Error
"""

import requests
import json

def test_guardian_api():
    """Test the Guardian API endpoint step by step."""
    
    url = "http://127.0.0.1:11435/guardian/analyze"
    
    # Simple test message
    payload = {
        "text": "Hello world"
    }
    
    try:
        print("Testing Guardian API...")
        print(f"URL: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS!")
            print(json.dumps(result, indent=2))
        else:
            print("FAILED!")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_guardian_api()