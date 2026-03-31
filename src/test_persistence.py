"""
Test that ChromaDB actually persists data
"""

from vector_store import VectorStore
import random

print("\n🧪 TESTING PERSISTENCE")
print("=" * 70)

# Step 1: Add data
print("\n📝 Step 1: Adding test data...")
store1 = VectorStore()

sample_chunks = [
    {
        'text': f'Test document {i}',
        'source': 'test.pdf',
        'chunk_id': i,
        'word_count': 3,
        'char_count': 15
    }
    for i in range(5)
]

sample_embeddings = [[random.random() for _ in range(1536)] for _ in range(5)]

store1.add_documents(sample_chunks, sample_embeddings)
print(f"✅ Added 5 documents")

# Step 2: Create new instance (simulates restart)
print("\n📝 Step 2: Creating new instance (simulating restart)...")
store2 = VectorStore()

# Step 3: Check if data persisted
count = store2.collection.count()
print(f"📊 Documents found after 'restart': {count}")

if count == 5:
    print("✅ PERSISTENCE TEST PASSED!")
else:
    print(f"❌ PERSISTENCE TEST FAILED! Expected 5, found {count}")

# Clean up
store2.delete_all()
print("\n🗑️  Cleaned up test data")