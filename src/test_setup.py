"""
Test script to verify OpenAI API is working
Run this to make sure everything is set up correctly
"""

import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

# Check if API key exists
if not api_key:
    print("❌ ERROR: OPENAI_API_KEY not found in .env file")
    print("Please check your .env file and make sure it contains:")
    print("OPENAI_API_KEY=your-key-here")
    exit(1)

print("✅ API key found")
print(f"Key starts with: {api_key[:10]}...")

# Set API key
openai.api_key = api_key

# Test API call
print("\n🔄 Testing API connection...")
print("Sending test message to GPT-3.5-turbo...")

try:
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Say 'API test successful!' if you can read this."}
        ],
        max_tokens=50
    )
    
    # Get response
    answer = response.choices[0].message.content
    
    print("\n✅ API Test SUCCESSFUL!")
    print(f"Response from OpenAI: {answer}")
    print(f"Tokens used: {response.usage.total_tokens}")
    
except Exception as e:
    print("\n❌ API Test FAILED!")
    print(f"Error: {e}")
    print("\nPossible issues:")
    print("1. API key is invalid")
    print("2. No credit in OpenAI account")
    print("3. Internet connection issue")