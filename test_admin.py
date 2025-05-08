import os
import requests

# Admin key
api_key = "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"

print(f"Using admin key: {api_key[:10]}...")

# Make a direct API call
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Try different endpoints
endpoints = [
    ("models", "https://api.openai.com/v1/models"),
    ("usage", "https://api.openai.com/v1/dashboard/billing/usage"),
    ("subscription", "https://api.openai.com/v1/dashboard/billing/subscription")
]

for name, url in endpoints:
    try:
        print(f"\nTesting {name} endpoint...")
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:200]}...")  # Print first 200 chars
    except Exception as e:
        print(f"Error: {str(e)}") 