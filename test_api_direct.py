import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("No API key found in environment variables!")
    exit(1)

print(f"Using API key: {api_key[:10]}...")

# Make a direct API call
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Try to list available models
response = requests.get(
    "https://api.openai.com/v1/models",
    headers=headers
)

print("\nAPI Response:")
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")

# If that fails, try to get account info
if response.status_code != 200:
    print("\nTrying to get account info...")
    account_response = requests.get(
        "https://api.openai.com/v1/dashboard/billing/usage",
        headers=headers
    )
    print(f"Account Status Code: {account_response.status_code}")
    print(f"Account Response: {account_response.text}") 