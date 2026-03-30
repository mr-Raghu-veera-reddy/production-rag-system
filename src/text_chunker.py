"""
Text Chunker Module
Splits documents into smaller chunks for embedding and retrieval
"""

from typing import List, Dict

class TextChunker:
    """
    Split documents into overlapping chunks
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize chunker with size parameters
        
        Args:
            chunk_size: Number of words per chunk (default 500)
            chunk_overlap: Number of overlapping words between chunks (default 50)
        """
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        print(f"✂️  Text Chunker initialized")
        print(f"   Chunk size: {chunk_size} words")
        print(f"   Overlap: {chunk_overlap} words")
    
    def chunk_text(self, text: str, source: str = "") -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to split
            source: Source document name (for reference)
            
        Returns:
            List of chunk dictionaries
        """
        
        # Split text into words
        words = text.split()
        
        if not words:
            print(f"   ⚠️  Warning: Empty text from {source}")
            return []
        
        # Calculate step size (how many words to move forward)
        step = self.chunk_size - self.chunk_overlap
        
        # Create chunks
        chunks = []
        for i in range(0, len(words), step):
            # Get chunk words
            chunk_words = words[i:i + self.chunk_size]
            
            # Join words back into text
            chunk_text = ' '.join(chunk_words)
            
            # Create chunk dictionary
            chunk = {
                'text': chunk_text,
                'source': source,
                'chunk_id': len(chunks),
                'start_word': i,
                'end_word': i + len(chunk_words),
                'word_count': len(chunk_words),
                'char_count': len(chunk_text)
            }
            
            chunks.append(chunk)
            
            # Stop if this was the last chunk
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of document dictionaries from DocumentLoader
            
        Returns:
            List of all chunks from all documents
        """
        
        print("\n" + "=" * 70)
        print("✂️  CHUNKING DOCUMENTS")
        print("=" * 70)
        
        all_chunks = []
        
        for doc in documents:
            print(f"\nChunking: {doc['filename']}")
            
            # Chunk this document
            chunks = self.chunk_text(doc['text'], doc['filename'])
            
            print(f"   Created {len(chunks)} chunks")
            
            # Add to all chunks
            all_chunks.extend(chunks)
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 CHUNKING SUMMARY")
        print("=" * 70)
        print(f"Documents chunked: {len(documents)}")
        print(f"Total chunks created: {len(all_chunks)}")
        
        if all_chunks:
            avg_words = sum(c['word_count'] for c in all_chunks) / len(all_chunks)
            print(f"Average words per chunk: {avg_words:.0f}")
        
        print("=" * 70)
        
        return all_chunks
    
    def print_chunk_preview(self, chunks: List[Dict], num_chunks: int = 3):
        """
        Print preview of first few chunks
        
        Args:
            chunks: List of chunks
            num_chunks: Number of chunks to preview
        """
        
        print("\n" + "=" * 70)
        print(f"📄 CHUNK PREVIEW (First {num_chunks} chunks)")
        print("=" * 70)
        
        for i, chunk in enumerate(chunks[:num_chunks], 1):
            print(f"\n[Chunk {i}]")
            print(f"Source: {chunk['source']}")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Words: {chunk['word_count']}")
            print(f"Text preview: {chunk['text'][:200]}...")
            print("-" * 70)


# Test code
if __name__ == "__main__":
    print("\n🧪 TESTING TEXT CHUNKER")
    print("=" * 70)
    
    # First, load documents
    from document_loader import DocumentLoader
    
    loader = DocumentLoader()
    documents = loader.load_directory("./data")
    
    if not documents:
        print("❌ No documents to chunk. Add PDFs to ./data folder first.")
        exit(1)
    
    # Create chunker
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    
    # Chunk documents
    chunks = chunker.chunk_documents(documents)
    
    # Show preview
    if chunks:
        chunker.print_chunk_preview(chunks, num_chunks=3)
        
        print("\n✅ Text Chunker Test PASSED!")
    else:
        print("\n❌ Text Chunker Test FAILED!")