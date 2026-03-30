"""
Test embedding generation
"""

import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

print("🧮 Testing Embedding Generation")
print("=" * 50)

# Test text
test_text = "Machine learning is a subset of artificial intelligence."

print(f"\nTest text: '{test_text}'")
print("\nGenerating embedding...")

try:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=test_text
    )
    
    embedding = response.data[0].embedding
    
    print(f"\n✅ Embedding generated successfully!")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Embedding type: {type(embedding)}")
    
    # Verify it's a list of floats
    assert isinstance(embedding, list), "Embedding should be a list"
    assert all(isinstance(x, float) for x in embedding), "All values should be floats"
    assert len(embedding) == 1536, "text-embedding-3-small should return 1536 dimensions"
    
    print("\n✅ All checks passed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")