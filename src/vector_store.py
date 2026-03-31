"""
Vector Store Module
Manage vector database using ChromaDB
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os

class VectorStore:
    """
    Manage document embeddings in ChromaDB
    """
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "rag_documents"):
        """
        Initialize ChromaDB vector store
        
        Args:
            persist_directory: Directory to persist database
            collection_name: Name of the collection
        """
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        print(f"💾 Initializing Vector Store...")
        print(f"   Directory: {persist_directory}")
        print(f"   Collection: {collection_name}")
        
        # Create ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG system document chunks with embeddings"}
        )
        
        print(f"✅ Vector Store initialized")
        print(f"   Current documents: {self.collection.count()}")
    
    def add_documents(
        self, 
        chunks: List[Dict], 
        embeddings: List[List[float]]
    ):
        """
        Add document chunks with embeddings to vector store
        
        Args:
            chunks: List of chunk dictionaries from TextChunker
            embeddings: List of embedding vectors from EmbeddingGenerator
        """
        
        if len(chunks) != len(embeddings):
            raise ValueError(f"Number of chunks ({len(chunks)}) must match number of embeddings ({len(embeddings)})")
        
        print(f"\n💾 Adding {len(chunks)} documents to vector store...")
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings_list = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create unique ID
            chunk_id = f"{chunk['source']}_{chunk['chunk_id']}"
            
            # Get document text
            document_text = chunk['text']
            
            # Create metadata
            metadata = {
                'source': chunk['source'],
                'chunk_id': str(chunk['chunk_id']),
                'word_count': chunk['word_count'],
                'char_count': chunk['char_count']
            }
            
            ids.append(chunk_id)
            documents.append(document_text)
            metadatas.append(metadata)
            embeddings_list.append(embedding)
        
        # Add to collection (ChromaDB handles batching internally)
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"✅ Added {len(chunks)} chunks to vector store")
        print(f"   Total documents now: {self.collection.count()}")
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> Dict:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query_embedding: Embedding vector of the query
            top_k: Number of results to return
            
        Returns:
            Dictionary with results
        """
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results
    
    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5
    ) -> Dict:
        """
        Search using text query (ChromaDB will embed it)
        
        Args:
            query_text: Text query
            top_k: Number of results to return
            
        Returns:
            Dictionary with results
        """
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        
        return results
    
    def get_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Get specific chunk by ID
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk data or None
        """
        
        try:
            result = self.collection.get(
                ids=[chunk_id]
            )
            return result
        except:
            return None
    
    def delete_all(self):
        """
        Delete all documents from collection
        """
        
        # Delete collection
        self.client.delete_collection(name=self.collection_name)
        
        # Recreate empty collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG system document chunks with embeddings"}
        )
        
        print(f"🗑️  Deleted all documents from collection")
    
    def get_stats(self) -> Dict:
        """
        Get vector store statistics
        
        Returns:
            Dictionary with stats
        """
        
        count = self.collection.count()
        
        return {
            'collection_name': self.collection_name,
            'total_documents': count,
            'persist_directory': self.persist_directory
        }
    
    def print_stats(self):
        """
        Print vector store statistics
        """
        
        stats = self.get_stats()
        
        print("\n📊 VECTOR STORE STATISTICS")
        print("=" * 70)
        print(f"Collection: {stats['collection_name']}")
        print(f"Total documents: {stats['total_documents']}")
        print(f"Storage location: {stats['persist_directory']}")
        print("=" * 70)


# Test code
if __name__ == "__main__":
    print("\n🧪 TESTING VECTOR STORE")
    print("=" * 70)
    
    # Test 1: Initialize store
    print("\n📝 Test 1: Initialize vector store")
    store = VectorStore()
    
    # Test 2: Add sample documents
    print("\n📝 Test 2: Add sample documents")
    
    # Create sample chunks
    sample_chunks = [
        {
            'text': 'Machine learning is a subset of AI.',
            'source': 'test.pdf',
            'chunk_id': 0,
            'word_count': 8,
            'char_count': 38
        },
        {
            'text': 'Deep learning uses neural networks.',
            'source': 'test.pdf',
            'chunk_id': 1,
            'word_count': 6,
            'char_count': 36
        }
    ]
    
    # Generate sample embeddings (simple random for testing)
    import random
    sample_embeddings = [[random.random() for _ in range(1536)] for _ in range(2)]
    
    # Add to store
    store.add_documents(sample_chunks, sample_embeddings)
    
    # Test 3: Search
    print("\n📝 Test 3: Search functionality")
    query_embedding = [random.random() for _ in range(1536)]
    results = store.search(query_embedding, top_k=2)
    
    print(f"✅ Search returned {len(results['documents'][0])} results")
    
    # Test 4: Stats
    print("\n📝 Test 4: Get statistics")
    store.print_stats()
    
    # Test 5: Clean up
    print("\n📝 Test 5: Clean up test data")
    store.delete_all()
    print(f"✅ Cleaned up. Documents remaining: {store.collection.count()}")
    
    print("\n✅ VECTOR STORE TEST PASSED!")
    print("=" * 70)