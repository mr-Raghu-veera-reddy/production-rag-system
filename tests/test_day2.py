"""
Complete Day 2 test suite
Tests all components together
"""

print("\n" + "=" * 70)
print("🧪 DAY 2 COMPREHENSIVE TEST")
print("=" * 70)

# Test 1: Check all modules import correctly
print("\n📝 Test 1: Module imports")
try:
    from src.embeddings import EmbeddingGenerator
    from src.vector_store import VectorStore
    from src.ingestion_pipeline import IngestionPipeline
    print("✅ All modules import successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Check vector store has data
print("\n📝 Test 2: Vector store data")
store = VectorStore()
count = store.collection.count()
print(f"   Documents in store: {count}")

if count == 0:
    print("❌ No documents in vector store!")
    print("   Run: python src/ingestion_pipeline.py")
    exit(1)
else:
    print(f"✅ Vector store contains {count} documents")

# Test 3: Test embedding generation
print("\n📝 Test 3: Embedding generation")
embedder = EmbeddingGenerator()
test_text = "This is a test sentence."
embedding = embedder.get_embedding(test_text)

if embedding and len(embedding) == 1536:
    print(f"✅ Embedding generated (dimension: {len(embedding)})")
else:
    print("❌ Embedding generation failed")
    exit(1)

# Test 4: Test retrieval
print("\n📝 Test 4: Search functionality")
query = "machine learning"
query_embedding = embedder.get_embedding(query)
results = store.search(query_embedding, top_k=5)

if len(results['documents'][0]) > 0:
    print(f"✅ Search returned {len(results['documents'][0])} results")
else:
    print("❌ Search returned no results")
    exit(1)

# Test 5: Verify result quality
print("\n📝 Test 5: Result relevance")
top_result = results['documents'][0][0]
distance = results['distances'][0][0]

print(f"   Query: '{query}'")
print(f"   Top result distance: {distance:.4f}")
print(f"   Top result preview: {top_result[:100]}...")

if distance < 1.0:  # Reasonable similarity threshold
    print("✅ Results appear relevant")
else:
    print("⚠️  Results may not be very relevant (high distance)")

# Summary
print("\n" + "=" * 70)
print("✅ ALL DAY 2 TESTS PASSED!")
print("=" * 70)
print("\nDay 2 Components Working:")
print("  ✅ Embedding generation")
print("  ✅ Vector store")
print("  ✅ Document ingestion pipeline")
print("  ✅ Search/retrieval")
print("\n🎉 Ready for Day 3!")