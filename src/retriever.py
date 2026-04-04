"""
Retriever Module
Retrieve relevant chunks for a given query
"""

from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from typing import List, Dict

class Retriever:
    """
    Retrieve relevant document chunks for queries
    """
    
    def __init__(self, top_k: int = 5):
        """
        Initialize retriever
        
        Args:
            top_k: Number of chunks to retrieve per query
        """
        
        print("🔍 Initializing Retriever...")
        
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.top_k = top_k
        
        # Check if vector store has data
        count = self.vector_store.collection.count()
        if count == 0:
            print("⚠️  WARNING: Vector store is empty!")
            print("   Run: python src/ingestion_pipeline.py")
        else:
            print(f"✅ Retriever ready ({count} documents available)")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve most relevant chunks for a query
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve (overrides default)
            
        Returns:
            List of retrieved chunks with metadata
        """
        
        if top_k is None:
            top_k = self.top_k
        
        print(f"\n🔍 Searching for: '{query}'")
        print(f"   Retrieving top {top_k} chunks...")
        
        # Step 1: Generate query embedding
        query_embedding = self.embedder.get_embedding(query)
        
        if not query_embedding:
            print("❌ Failed to generate query embedding")
            return []
        
        # Step 2: Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # Step 3: Format results
        retrieved_chunks = []
        
        for i in range(len(results['documents'][0])):
            chunk = {
                'text': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source'],
                'chunk_id': results['metadatas'][0][i]['chunk_id'],
                'distance': results['distances'][0][i],
                'rank': i + 1
            }
            retrieved_chunks.append(chunk)
        
        print(f"✅ Retrieved {len(retrieved_chunks)} chunks")
        
        return retrieved_chunks
    
    def print_results(self, chunks: List[Dict]):
        """
        Pretty print retrieved chunks
        
        Args:
            chunks: List of retrieved chunks
        """
        
        print("\n" + "=" * 80)
        print("📋 RETRIEVED CHUNKS")
        print("=" * 80)
        
        for chunk in chunks:
            print(f"\n[Rank {chunk['rank']}] {chunk['source']} (Chunk {chunk['chunk_id']})")
            print(f"Distance: {chunk['distance']:.4f}")
            print(f"Text: {chunk['text'][:300]}...")
            print("-" * 80)
    
    def get_context_string(self, chunks: List[Dict]) -> str:
        """
        Combine retrieved chunks into a single context string
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Combined context string
        """
        
        context_parts = []
        
        for chunk in chunks:
            # Format: [Source: filename] text
            context_parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        return context


# Test code
if __name__ == "__main__":
    print("\n🧪 TESTING RETRIEVER")
    print("=" * 80)
    
    # Create retriever
    retriever = Retriever(top_k=5)
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does deep learning work?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST QUERY {i}/{len(test_queries)}")
        print('=' * 80)
        
        # Retrieve
        chunks = retriever.retrieve(query, top_k=3)
        
        # Print results
        retriever.print_results(chunks)
        
        # Show context string
        print("\n📝 COMBINED CONTEXT:")
        print("-" * 80)
        context = retriever.get_context_string(chunks)
        print(context[:500] + "...")
    
    print("\n" + "=" * 80)
    print("✅ RETRIEVER TEST PASSED!")
    print("=" * 80)