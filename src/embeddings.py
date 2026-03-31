"""
Embeddings Module
Generate vector embeddings for text chunks using OpenAI API
"""

import openai
import os
from dotenv import load_dotenv
from typing import List, Dict
import time
import json

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class EmbeddingGenerator:
    """
    Generate embeddings using OpenAI's embedding models
    """
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize embedding generator
        
        Args:
            model: OpenAI embedding model to use
                - text-embedding-3-small: $0.02 per 1M tokens (1536 dimensions)
                - text-embedding-3-large: $0.13 per 1M tokens (3072 dimensions)
        """
        
        self.model = model
        self.embedding_cache = {}  # Cache to avoid re-embedding same text
        self.total_tokens_used = 0
        
        print(f"🧮 Embedding Generator initialized")
        print(f"   Model: {model}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats (embedding vector)
        """
        
        # Check cache first
        if text in self.embedding_cache:
            print("   📦 Using cached embedding")
            return self.embedding_cache[text]
        
        try:
            # Call OpenAI API
            response = openai.embeddings.create(
                model=self.model,
                input=text
            )
            
            # Extract embedding
            embedding = response.data[0].embedding
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            
            # Cache the embedding
            self.embedding_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            print(f"   ❌ Error generating embedding: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches
        More efficient than calling API for each text individually
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to embed in one API call (max 2048)
            
        Returns:
            List of embedding vectors
        """
        
        print(f"\n🧮 Generating embeddings for {len(texts)} texts...")
        print(f"   Batch size: {batch_size}")
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) - 1) // batch_size + 1
            
            print(f"\n   Processing batch {batch_num}/{total_batches} ({len(batch)} texts)...")
            
            try:
                # Make API call for entire batch
                response = openai.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                # Extract embeddings (maintain order)
                batch_embeddings = [item.embedding for item in response.data]
                
                # Track tokens
                tokens_used = response.usage.total_tokens
                self.total_tokens_used += tokens_used
                
                print(f"   ✅ Batch complete ({tokens_used} tokens)")
                
                # Add to results
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting - be nice to API
                # OpenAI allows 3,000 requests/min for tier 1
                time.sleep(0.5)  # Small delay between batches
                
            except Exception as e:
                print(f"   ❌ Error in batch {batch_num}: {e}")
                
                # If batch fails, try one by one
                print(f"   🔄 Retrying texts individually...")
                for text in batch:
                    embedding = self.get_embedding(text)
                    if embedding:
                        all_embeddings.append(embedding)
                    else:
                        # Use zero vector as fallback
                        all_embeddings.append([0.0] * 1536)
        
        print(f"\n✅ Generated {len(all_embeddings)} embeddings")
        print(f"   Total tokens used: {self.total_tokens_used:,}")
        print(f"   Estimated cost: ${self.total_tokens_used * 0.00002:.4f}")
        
        return all_embeddings
    
    def save_cache(self, cache_file: str = "cache/embedding_cache.json"):
        """
        Save embedding cache to file
        
        Args:
            cache_file: Path to cache file
        """
        
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        with open(cache_file, 'w') as f:
            json.dump(self.embedding_cache, f)
        
        print(f"💾 Saved {len(self.embedding_cache)} embeddings to cache")
    
    def load_cache(self, cache_file: str = "cache/embedding_cache.json"):
        """
        Load embedding cache from file
        
        Args:
            cache_file: Path to cache file
        """
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.embedding_cache = json.load(f)
            
            print(f"📦 Loaded {len(self.embedding_cache)} embeddings from cache")
        else:
            print("📦 No cache file found, starting fresh")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about embedding generation
        
        Returns:
            Dictionary with stats
        """
        
        return {
            'model': self.model,
            'total_tokens_used': self.total_tokens_used,
            'estimated_cost': self.total_tokens_used * 0.00002,
            'cached_embeddings': len(self.embedding_cache)
        }


# Test code
if __name__ == "__main__":
    print("\n🧪 TESTING EMBEDDING GENERATOR")
    print("=" * 70)
    
    # Create generator
    generator = EmbeddingGenerator()
    
    # Test 1: Single embedding
    print("\n📝 Test 1: Single text embedding")
    test_text = "Machine learning is a subset of artificial intelligence."
    embedding = generator.get_embedding(test_text)
    
    if embedding:
        print(f"✅ Embedding generated")
        print(f"   Dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        print(f"   Type: {type(embedding[0])}")
    else:
        print("❌ Failed to generate embedding")
        exit(1)
    
    # Test 2: Batch embeddings
    print("\n📝 Test 2: Batch embeddings")
    test_texts = [
        "Deep learning uses neural networks.",
        "Natural language processing analyzes text.",
        "Computer vision processes images.",
        "Reinforcement learning learns from rewards.",
        "Supervised learning uses labeled data."
    ]
    
    embeddings = generator.get_embeddings_batch(test_texts, batch_size=5)
    
    if len(embeddings) == len(test_texts):
        print(f"✅ All embeddings generated")
        print(f"   Count: {len(embeddings)}")
        print(f"   Each dimension: {len(embeddings[0])}")
    else:
        print(f"❌ Expected {len(test_texts)} embeddings, got {len(embeddings)}")
        exit(1)
    
    # Test 3: Cache test
    print("\n📝 Test 3: Cache functionality")
    # This should use cache (same text as Test 1)
    cached_embedding = generator.get_embedding(test_text)
    
    if cached_embedding == embedding:
        print("✅ Cache working correctly")
    else:
        print("❌ Cache not working")
    
    # Show stats
    print("\n📊 Statistics:")
    stats = generator.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ EMBEDDING GENERATOR TEST PASSED!")
    print("=" * 70)