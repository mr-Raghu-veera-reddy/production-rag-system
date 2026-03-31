"""
Complete Document Ingestion Pipeline
PDF → Chunks → Embeddings → Vector Store
"""

# from document_loader import DocumentLoader
# from text_chunker import TextChunker
# from embeddings import EmbeddingGenerator
# from vector_store import VectorStore
# ✅ Fixed - works correctly as part of the src package
from .document_loader import DocumentLoader
from .text_chunker import TextChunker
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
import time

class IngestionPipeline:
    """
    Complete pipeline to ingest documents into RAG system
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize ingestion pipeline
        
        Args:
            chunk_size: Number of words per chunk
            chunk_overlap: Number of overlapping words
            embedding_model: OpenAI embedding model
        """
        
        print("=" * 70)
        print("🚀 INITIALIZING INGESTION PIPELINE")
        print("=" * 70)
        
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = EmbeddingGenerator(model=embedding_model)
        self.vector_store = VectorStore()
        
        print("\n✅ All components initialized")
    
    def ingest_directory(self, directory_path: str, clear_existing: bool = False):
        """
        Ingest all PDFs from a directory
        
        Args:
            directory_path: Path to directory containing PDFs
            clear_existing: Whether to clear existing data first
        """
        
        start_time = time.time()
        
        print("\n" + "=" * 70)
        print("📂 STARTING DOCUMENT INGESTION")
        print("=" * 70)
        print(f"Directory: {directory_path}")
        print(f"Clear existing: {clear_existing}")
        
        # Optional: Clear existing data
        if clear_existing:
            print("\n🗑️  Clearing existing data...")
            self.vector_store.delete_all()
        
        # Step 1: Load documents
        print("\n" + "─" * 70)
        print("📖 STEP 1/4: Loading Documents")
        print("─" * 70)
        
        documents = self.loader.load_directory(directory_path)
        
        if not documents:
            print("❌ No documents found! Aborting.")
            return
        
        print(f"✅ Loaded {len(documents)} documents")
        
        # Step 2: Chunk documents
        print("\n" + "─" * 70)
        print("✂️  STEP 2/4: Chunking Documents")
        print("─" * 70)
        
        chunks = self.chunker.chunk_documents(documents)
        
        print(f"✅ Created {len(chunks)} chunks")
        
        # Step 3: Generate embeddings
        print("\n" + "─" * 70)
        print("🧮 STEP 3/4: Generating Embeddings")
        print("─" * 70)
        
        # Extract just the text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches
        embeddings = self.embedder.get_embeddings_batch(texts, batch_size=100)
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        # Step 4: Store in vector database
        print("\n" + "─" * 70)
        print("💾 STEP 4/4: Storing in Vector Database")
        print("─" * 70)
        
        self.vector_store.add_documents(chunks, embeddings)
        
        print(f"✅ Stored {len(chunks)} chunks in vector database")
        
        # Summary
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "=" * 70)
        print("✅ INGESTION COMPLETE!")
        print("=" * 70)
        
        # Print statistics
        self._print_summary(documents, chunks, duration)
    
    def _print_summary(self, documents, chunks, duration):
        """Print ingestion summary"""
        
        # Document stats
        total_pages = sum(doc['num_pages'] for doc in documents)
        total_words = sum(doc['word_count'] for doc in documents)
        
        # Embedding stats
        embedding_stats = self.embedder.get_stats()
        
        # Vector store stats
        vector_stats = self.vector_store.get_stats()
        
        print("\n📊 INGESTION SUMMARY")
        print("─" * 70)
        print(f"Documents processed: {len(documents)}")
        print(f"Total pages: {total_pages}")
        print(f"Total words: {total_words:,}")
        print(f"Chunks created: {len(chunks)}")
        print(f"Embeddings generated: {embedding_stats['total_tokens_used']:,} tokens")
        print(f"Estimated cost: ${embedding_stats['estimated_cost']:.4f}")
        print(f"Vector store documents: {vector_stats['total_documents']}")
        print(f"Time taken: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print("=" * 70)


# Run ingestion
if __name__ == "__main__":
    print("\n🧪 RUNNING INGESTION PIPELINE")
    print("=" * 70)
    
    # Create pipeline
    pipeline = IngestionPipeline(
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Ingest documents from data folder
    pipeline.ingest_directory(
        directory_path="./data",
        clear_existing=True  # Clear old data first
    )
    
    print("\n🎉 Pipeline execution complete!")