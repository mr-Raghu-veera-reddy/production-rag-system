"""
Text Cleaner Module
Clean text to remove problematic characters
"""

import re
import unicodedata

class TextCleaner:
    """
    Clean text to remove problematic characters and normalize
    """
    
    def __init__(self):
        print("🧹 Text Cleaner initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing/replacing problematic characters
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        
        if not text:
            return ""
        
        # Step 1: Remove surrogate pairs (mathematical symbols)
        # These are characters in the range U+D800 to U+DFFF
        text = re.sub(r'[\ud800-\udfff]', '', text)
        
        # Step 2: Normalize unicode (convert special chars to closest ASCII)
        # NFD = Canonical Decomposition
        text = unicodedata.normalize('NFKD', text)
        
        # Step 3: Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        # Step 4: Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Step 5: Remove any remaining problematic Unicode
        # Only keep ASCII and common extended characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Step 6: Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_chunk(self, chunk: dict) -> dict:
        """
        Clean a chunk dictionary
        
        Args:
            chunk: Chunk dictionary with 'text' field
            
        Returns:
            Chunk with cleaned text
        """
        
        if 'text' in chunk:
            original_length = len(chunk['text'])
            chunk['text'] = self.clean_text(chunk['text'])
            cleaned_length = len(chunk['text'])
            
            # Update character count
            chunk['char_count'] = cleaned_length
            
            # Optionally track how much was removed
            if original_length != cleaned_length:
                removed = original_length - cleaned_length
                # You can log this if needed
                pass
        
        return chunk
    
    def clean_chunks(self, chunks: list) -> list:
        """
        Clean a list of chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of cleaned chunks
        """
        
        print(f"🧹 Cleaning {len(chunks)} chunks...")
        
        cleaned_chunks = []
        total_removed = 0
        
        for chunk in chunks:
            original_length = len(chunk.get('text', ''))
            cleaned_chunk = self.clean_chunk(chunk)
            cleaned_length = len(cleaned_chunk.get('text', ''))
            
            total_removed += (original_length - cleaned_length)
            cleaned_chunks.append(cleaned_chunk)
        
        if total_removed > 0:
            print(f"   Removed {total_removed:,} problematic characters")
        
        print(f"✅ Cleaned {len(cleaned_chunks)} chunks")
        
        return cleaned_chunks


# Test
if __name__ == "__main__":
    print("\n🧪 TESTING TEXT CLEANER")
    print("=" * 70)
    
    cleaner = TextCleaner()
    
    # Test with problematic text
    test_text = "This is \ud835\udc65 and \ud835\udf03 some math symbols"
    
    print(f"\nOriginal: {repr(test_text)}")
    
    cleaned = cleaner.clean_text(test_text)
    
    print(f"Cleaned: {repr(cleaned)}")
    print(f"Length: {len(test_text)} → {len(cleaned)}")
    
    # Test with chunk
    test_chunk = {
        'text': test_text,
        'source': 'test.pdf',
        'chunk_id': 0,
        'char_count': len(test_text)
    }
    
    cleaned_chunk = cleaner.clean_chunk(test_chunk)
    
    print(f"\nCleaned chunk text: {cleaned_chunk['text']}")
    print(f"Updated char_count: {cleaned_chunk['char_count']}")
    
    print("\n✅ TEXT CLEANER TEST PASSED!")