"""
Day 4 comprehensive test
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("\n" + "=" * 80)
print("🧪 DAY 4 COMPREHENSIVE TEST")
print("=" * 80)

# Test imports
print("\n📝 Test 1: New module imports")
try:
    from src.query_rewriter import QueryRewriter
    from src.reranker import Reranker
    from src.advanced_retriever import AdvancedRetriever
    from src.monitoring import Monitor
    print("✅ All new modules import successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test query rewriter
print("\n📝 Test 2: Query rewriter")
rewriter = QueryRewriter()
variants = rewriter.rewrite_query("What's ML?")
if len(variants) > 1:
    print(f"✅ Query rewriting works ({len(variants)} variants)")
else:
    print("❌ Query rewriting failed")
    exit(1)

# Test re-ranker
print("\n📝 Test 3: Re-ranker")
reranker = Reranker()
sample_chunks = [
    {'text': 'ML is AI subset', 'source': 'd1', 'chunk_id': '0', 'distance': 0.3},
    {'text': 'Weather is sunny', 'source': 'd2', 'chunk_id': '1', 'distance': 0.4}
]
reranked = reranker.rerank("What is ML?", sample_chunks, top_k=1)
if len(reranked) == 1:
    print(f"✅ Re-ranking works (score: {reranked[0].get('rerank_score', 0):.1f})")
else:
    print("❌ Re-ranking failed")
    exit(1)

# Test advanced retriever
print("\n📝 Test 4: Advanced retriever")
try:
    adv_ret = AdvancedRetriever(use_query_rewriting=True, use_reranking=True, top_k=3)
    chunks = adv_ret.retrieve("What is machine learning?")
    if len(chunks) > 0:
        print(f"✅ Advanced retrieval works ({len(chunks)} chunks)")
    else:
        print("❌ Advanced retrieval returned no chunks")
        exit(1)
except Exception as e:
    print(f"❌ Advanced retrieval failed: {e}")
    exit(1)

# Test monitoring
print("\n📝 Test 5: Monitoring")
monitor = Monitor(log_file="logs/test_day4.jsonl")
monitor.log_query(
    query="test",
    answer="test answer",
    result={'sources': ['d1'], 'chunks_used': 3, 'tokens_used': 100, 'cost': 0.0001, 'latency': 2.0, 'model': 'gpt-3.5-turbo'}
)
stats = monitor.get_stats()
if stats.get('total_queries', 0) > 0:
    print(f"✅ Monitoring works ({stats['total_queries']} queries logged)")
else:
    print("❌ Monitoring failed")
    exit(1)

# Clean up test logs
import os
if os.path.exists("logs/test_day4.jsonl"):
    os.remove("logs/test_day4.jsonl")

# Summary
print("\n" + "=" * 80)
print("✅ ALL DAY 4 TESTS PASSED!")
print("=" * 80)
print("\nDay 4 Features Working:")
print("  ✅ Query rewriting")
print("  ✅ Re-ranking")
print("  ✅ Advanced retrieval")
print("  ✅ Monitoring")
print("\n🎉 Ready for Day 5!")