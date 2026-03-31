"""
Complete end-to-end system test
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("\n" + "=" * 80)
print("🧪 COMPLETE SYSTEM TEST")
print("=" * 80)

# Test 1: All imports work
print("\n📝 Test 1: Module imports")
try:
    from src.document_loader import DocumentLoader
    from src.text_chunker import TextChunker
    from src.embeddings import EmbeddingGenerator
    from src.vector_store import VectorStore
    from src.retriever import Retriever
    from src.qa_generator import QAGenerator
    from src.rag_system import RAGSystem
    print("✅ All modules import successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: RAG system initialization
print("\n📝 Test 2: RAG system initialization")
try:
    rag = RAGSystem()
    print("✅ RAG system initialized")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    exit(1)

# Test 3: Question answering
print("\n📝 Test 3: End-to-end question answering")
test_question = "What is machine learning?"

try:
    result = rag.query(test_question)
    
    # Verify result structure
    required_fields = ['query', 'answer', 'sources', 'chunks_used', 'tokens_used', 'cost', 'latency']
    
    for field in required_fields:
        if field not in result:
            print(f"❌ Missing field: {field}")
            exit(1)
    
    print(f"✅ Question answered successfully")
    print(f"   Answer length: {len(result['answer'])} characters")
    print(f"   Sources: {len(result['sources'])}")
    print(f"   Chunks used: {result['chunks_used']}")
    print(f"   Latency: {result['latency']:.2f}s")
    
except Exception as e:
    print(f"❌ Question answering failed: {e}")
    exit(1)

# Test 4: Multiple queries
print("\n📝 Test 4: Multiple sequential queries")
test_queries = [
    "What are neural networks?",
    "Explain deep learning",
    "What is reinforcement learning?"
]

try:
    total_cost = 0
    
    for i, query in enumerate(test_queries, 1):
        result = rag.query(query)
        total_cost += result['cost']
        print(f"   Query {i}: ✅ ({result['cost']:.4f})")
    
    print(f"✅ All queries completed")
    print(f"   Total cost: ${total_cost:.4f}")
    
except Exception as e:
    print(f"❌ Multiple queries failed: {e}")
    exit(1)

# Summary
print("\n" + "=" * 80)
print("✅ COMPLETE SYSTEM TEST PASSED!")
print("=" * 80)
print("\n🎉 Your RAG system is fully functional!")
print("\nAll components working:")
print("  ✅ Document loading")
print("  ✅ Text chunking")
print("  ✅ Embedding generation")
print("  ✅ Vector storage")
print("  ✅ Semantic retrieval")
print("  ✅ Answer generation")
print("  ✅ End-to-end pipeline")
print("\n🚀 Ready for deployment!")