"""
Advanced Retriever Module
Combines query rewriting and re-ranking for better retrieval
"""

from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from query_rewriter import QueryRewriter
from reranker import Reranker
from typing import List, Dict

class AdvancedRetriever:
    """
    Advanced retrieval with query rewriting and re-ranking
    """
    
    def __init__(
        self,
        use_query_rewriting: bool = True,
        use_reranking: bool = True,
        top_k: int = 5
    ):
        """
        Initialize advanced retriever
        
        Args:
            use_query_rewriting: Whether to rewrite queries
            use_reranking: Whether to re-rank results
            top_k: Final number of chunks to return
        """
        
        print("=" * 80)
        print("🚀 INITIALIZING ADVANCED RETRIEVER")
        print("=" * 80)
        
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()
        
        self.use_query_rewriting = use_query_rewriting
        self.use_reranking = use_reranking
        self.top_k = top_k
        
        if use_query_rewriting:
            self.query_rewriter = QueryRewriter()
            print("✅ Query rewriting: ENABLED")
        else:
            self.query_rewriter = None
            print("⚪ Query rewriting: DISABLED")
        
        if use_reranking:
            self.reranker = Reranker()
            print("✅ Re-ranking: ENABLED")
        else:
            self.reranker = None
            print("⚪ Re-ranking: DISABLED")
        
        print("=" * 80)
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Advanced retrieval with optional query rewriting and re-ranking
        
        Args:
            query: User's question
            top_k: Number of final chunks (overrides default)
            
        Returns:
            Retrieved and optionally re-ranked chunks
        """
        
        if top_k is None:
            top_k = self.top_k
        
        print(f"\n{'=' * 80}")
        print(f"🔍 ADVANCED RETRIEVAL")
        print('=' * 80)
        print(f"Query: '{query}'")
        print(f"Target chunks: {top_k}")
        
        # Step 1: Query rewriting (if enabled)
        if self.use_query_rewriting and self.query_rewriter:
            print(f"\n[Step 1/3] Query rewriting...")
            query_variants = self.query_rewriter.rewrite_query(query, num_variants=2)
        else:
            print(f"\n[Step 1/3] Skipping query rewriting...")
            query_variants = [query]
        
        # Step 2: Retrieve chunks for all query variants
        print(f"\n[Step 2/3] Retrieving chunks...")
        
        all_chunks = []
        seen_chunk_ids = set()
        
        # If re-ranking, get more chunks initially
        retrieve_k = top_k * 2 if self.use_reranking else top_k
        
        for i, variant in enumerate(query_variants, 1):
            print(f"   Variant {i}: '{variant}'")
            
            # Generate embedding
            embedding = self.embedder.get_embedding(variant)
            
            # Search
            results = self.vector_store.search(embedding, top_k=retrieve_k)
            
            # Add unique chunks
            for j in range(len(results['documents'][0])):
                chunk_id = f"{results['metadatas'][0][j]['source']}_{results['metadatas'][0][j]['chunk_id']}"
                
                if chunk_id not in seen_chunk_ids:
                    chunk = {
                        'text': results['documents'][0][j],
                        'source': results['metadatas'][0][j]['source'],
                        'chunk_id': results['metadatas'][0][j]['chunk_id'],
                        'distance': results['distances'][0][j],
                        'retrieved_by': variant
                    }
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)
        
        print(f"   Retrieved {len(all_chunks)} unique chunks")
        
        # Step 3: Re-ranking (if enabled)
        if self.use_reranking and self.reranker and len(all_chunks) > top_k:
            print(f"\n[Step 3/3] Re-ranking...")
            final_chunks = self.reranker.rerank(query, all_chunks, top_k=top_k)
        else:
            print(f"\n[Step 3/3] Skipping re-ranking...")
            # Just take top k by distance
            all_chunks.sort(key=lambda x: x['distance'])
            final_chunks = all_chunks[:top_k]
        
        print(f"\n✅ Final result: {len(final_chunks)} chunks")
        print('=' * 80)
        
        return final_chunks
    
    def print_results(self, chunks: List[Dict]):
        """Pretty print retrieved chunks"""
        
        print("\n" + "=" * 80)
        print("📋 RETRIEVED CHUNKS")
        print("=" * 80)
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\n[{i}] {chunk['source']} (Chunk {chunk['chunk_id']})")
            
            if 'rerank_score' in chunk:
                print(f"    Re-rank score: {chunk['rerank_score']:.1f}/10")
            
            print(f"    Distance: {chunk['distance']:.4f}")
            
            if 'retrieved_by' in chunk:
                print(f"    Retrieved by: '{chunk['retrieved_by']}'")
            
            print(f"    Text: {chunk['text'][:200]}...")
            print("-" * 80)


# Test code
if __name__ == "__main__":
    print("\n🧪 TESTING ADVANCED RETRIEVER")
    print("=" * 80)
    
    # Test 1: Basic retrieval (no advanced features)
    print("\n" + "#" * 80)
    print("# TEST 1: Basic Retrieval")
    print("#" * 80)
    
    basic_retriever = AdvancedRetriever(
        use_query_rewriting=False,
        use_reranking=False,
        top_k=3
    )
    
    chunks1 = basic_retriever.retrieve("What is ML?")
    basic_retriever.print_results(chunks1)
    
    # Test 2: With query rewriting only
    print("\n" + "#" * 80)
    print("# TEST 2: With Query Rewriting")
    print("#" * 80)
    
    rewrite_retriever = AdvancedRetriever(
        use_query_rewriting=True,
        use_reranking=False,
        top_k=3
    )
    
    chunks2 = rewrite_retriever.retrieve("What is ML?")
    rewrite_retriever.print_results(chunks2)
    
    # Test 3: Full advanced retrieval
    print("\n" + "#" * 80)
    print("# TEST 3: Full Advanced Retrieval")
    print("#" * 80)
    
    advanced_retriever = AdvancedRetriever(
        use_query_rewriting=True,
        use_reranking=True,
        top_k=3
    )
    
    chunks3 = advanced_retriever.retrieve("What is ML?")
    advanced_retriever.print_results(chunks3)
    
    print("\n✅ ADVANCED RETRIEVER TEST PASSED!")
    print("=" * 80)