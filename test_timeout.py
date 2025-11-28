#!/usr/bin/env python3
"""
Quick test to see what's happening with Guardian analyze endpoint
"""

import requests
import json

def test_simple():
    url = "http://127.0.0.1:11436/guardian/analyze"
    data = {"text": "Hello world"}
    
    print("Testing simple text...")
    try:
        response = requests.post(url, json=data, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except requests.exceptions.Timeout:
        print("TIMEOUT - Server hung!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_simple()