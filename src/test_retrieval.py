"""
Test basic retrieval from vector store
"""

from embeddings import EmbeddingGenerator
from vector_store import VectorStore

print("\n🧪 TESTING RETRIEVAL")
print("=" * 70)

# Initialize components
print("\n📝 Initializing components...")
embedder = EmbeddingGenerator()
store = VectorStore()

# Check if we have data
count = store.collection.count()
print(f"📊 Vector store contains {count} documents")

if count == 0:
    print("\n❌ No documents in vector store!")
    print("Run: python src/ingestion_pipeline.py first")
    exit(1)

# Test queries
test_queries = [
    "What is machine learning?",
    "Explain neural networks",
    "How does deep learning work?",
    "What are transformers in AI?"
]

print("\n" + "=" * 70)
print("🔍 TESTING SEARCH")
print("=" * 70)

for i, query in enumerate(test_queries, 1):
    print(f"\n[Query {i}] {query}")
    print("─" * 70)
    
    # Generate query embedding
    query_embedding = embedder.get_embedding(query)
    
    # Search
    results = store.search(query_embedding, top_k=3)
    
    # Display results
    for j in range(len(results['documents'][0])):
        print(f"\n  Result {j+1}:")
        print(f"  Source: {results['metadatas'][0][j]['source']}")
        print(f"  Distance: {results['distances'][0][j]:.4f}")
        print(f"  Text: {results['documents'][0][j][:150]}...")

print("\n" + "=" * 70)
print("✅ RETRIEVAL TEST PASSED!")
print("=" * 70)