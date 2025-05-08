import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("No API key found in environment variables!")
    exit(1)

print(f"Using API key: {api_key[:10]}...")

# Initialize client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.openai.com/v1"
)

# Test different API endpoints
endpoints = [
    ("chat", lambda: client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )),
    ("models", lambda: client.models.list()),
    ("audio", lambda: client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input="test",
        response_format="wav"
    ))
]

for endpoint_name, api_call in endpoints:
    try:
        print(f"\nTesting {endpoint_name} endpoint...")
        response = api_call()
        print(f"{endpoint_name} endpoint works! Response type: {type(response)}")
    except Exception as e:
        print(f"{endpoint_name} endpoint failed: {str(e)}")
        print(f"Error details: {e.__class__.__name__}")
        if hasattr(e, 'response'):
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")

# Test the API
try:
    print("\nTesting API key...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )
    print("API key works! Response:", response.choices[0].message.content)
except Exception as e:
    print(f"API test failed: {str(e)}")
    print("\nTroubleshooting steps:")
    print("1. Check your account status at https://platform.openai.com/account")
    print("2. Verify your API key at https://platform.openai.com/account/api-keys")
    print("3. Make sure you have sufficient credits/balance")
    print("4. Try creating a new API key if the current one isn't working") 