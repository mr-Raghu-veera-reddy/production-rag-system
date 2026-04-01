"""
Re-Ranker Module
Re-rank retrieved chunks using LLM for better relevance
"""

import openai
import os
from dotenv import load_dotenv
from typing import List, Dict
import time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class Reranker:
    """
    Re-rank retrieved chunks using LLM-based relevance scoring
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize re-ranker
        
        Args:
            model: OpenAI model to use for scoring
        """
        
        self.model = model
        self.total_tokens_used = 0
        
        print("🎯 Re-ranker initialized")
        print(f"   Model: {model}")
    
    def score_relevance(self, query: str, chunk_text: str) -> float:
        """
        Score how relevant a chunk is to a query
        
        Args:
            query: User's question
            chunk_text: Text chunk to score
            
        Returns:
            Relevance score from 0-10
        """
        
        # Truncate chunk if too long (to save tokens)
        max_chars = 1000
        if len(chunk_text) > max_chars:
            chunk_text = chunk_text[:max_chars] + "..."
        
        prompt = f"""Rate how relevant this text is for answering the question.

Question: {query}

Text: {chunk_text}

On a scale of 0-10:
- 0 = Completely irrelevant
- 5 = Somewhat relevant
- 10 = Highly relevant and directly answers the question

Return ONLY a number from 0 to 10, nothing else."""

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at judging text relevance. Return only a number."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,  # Deterministic scoring
                max_tokens=5
            )
            
            # Extract score
            score_text = response.choices[0].message.content.strip()
            
            # Track tokens
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            
            # Parse score
            try:
                score = float(score_text)
                # Ensure in range
                score = max(0, min(10, score))
            except:
                print(f"   ⚠️ Failed to parse score: '{score_text}', using 5.0")
                score = 5.0
            
            return score
            
        except Exception as e:
            print(f"   ❌ Error scoring chunk: {e}")
            # Fallback to neutral score
            return 5.0
    
    def rerank(
        self, 
        query: str, 
        chunks: List[Dict], 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Re-rank chunks by LLM-scored relevance
        
        Args:
            query: User's question
            chunks: Retrieved chunks from Retriever
            top_k: Number of chunks to keep after re-ranking
            
        Returns:
            Re-ranked chunks (highest scored first)
        """
        
        if len(chunks) <= top_k:
            print(f"🎯 Re-ranking skipped (only {len(chunks)} chunks)")
            return chunks
        
        print(f"\n🎯 Re-ranking {len(chunks)} chunks to top {top_k}...")
        
        scored_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"   Scoring chunk {i}/{len(chunks)}...", end=" ")
            
            # Score this chunk
            score = self.score_relevance(query, chunk['text'])
            
            print(f"{score:.1f}/10")
            
            # Add score to chunk
            chunk_with_score = chunk.copy()
            chunk_with_score['rerank_score'] = score
            scored_chunks.append(chunk_with_score)
            
            # Small delay to avoid rate limits
            time.sleep(0.2)
        
        # Sort by re-rank score (highest first)
        scored_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Take top k
        top_chunks = scored_chunks[:top_k]
        
        print(f"✅ Re-ranked to top {len(top_chunks)} chunks")
        print(f"   Score range: {top_chunks[-1]['rerank_score']:.1f} - {top_chunks[0]['rerank_score']:.1f}")
        
        return top_chunks
    
    def rerank_batch(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Re-rank chunks using batch scoring (faster but more expensive)
        
        Args:
            query: User's question
            chunks: Retrieved chunks
            top_k: Number to keep
            
        Returns:
            Re-ranked chunks
        """
        
        if len(chunks) <= top_k:
            return chunks
        
        print(f"\n🎯 Batch re-ranking {len(chunks)} chunks...")
        
        # Create batch prompt
        chunks_text = "\n\n".join([
            f"[Chunk {i+1}]\n{chunk['text'][:500]}"
            for i, chunk in enumerate(chunks)
        ])
        
        prompt = f"""Rate each chunk's relevance to the question on a scale of 0-10.

Question: {query}

{chunks_text}

Return scores as: 1:score, 2:score, 3:score, etc.
Example: 1:8, 2:3, 3:9"""

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You judge text relevance. Return scores in format: 1:score, 2:score"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=100
            )
            
            # Parse scores
            scores_text = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            
            # Extract scores
            scores = {}
            for pair in scores_text.split(','):
                try:
                    idx, score = pair.split(':')
                    scores[int(idx.strip())] = float(score.strip())
                except:
                    continue
            
            # Add scores to chunks
            scored_chunks = []
            for i, chunk in enumerate(chunks, 1):
                chunk_with_score = chunk.copy()
                chunk_with_score['rerank_score'] = scores.get(i, 5.0)
                scored_chunks.append(chunk_with_score)
            
            # Sort and return top k
            scored_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            print(f"✅ Batch re-ranked ({tokens_used} tokens)")
            
            return scored_chunks[:top_k]
            
        except Exception as e:
            print(f"❌ Batch re-ranking failed: {e}")
            # Fallback to original order
            return chunks[:top_k]
    
    def get_stats(self):
        """Get re-ranker statistics"""
        return {
            'model': self.model,
            'total_tokens_used': self.total_tokens_used,
            'estimated_cost': self.total_tokens_used * 0.0015 / 1000
        }


# Test code
if __name__ == "__main__":
    print("\n🧪 TESTING RE-RANKER")
    print("=" * 80)
    
    # Create some sample chunks
    sample_chunks = [
        {
            'text': 'Machine learning is a subset of artificial intelligence that enables computers to learn from data.',
            'source': 'doc1.pdf',
            'chunk_id': '0',
            'distance': 0.3
        },
        {
            'text': 'Deep learning uses neural networks with multiple layers to process complex patterns.',
            'source': 'doc2.pdf',
            'chunk_id': '1',
            'distance': 0.4
        },
        {
            'text': 'The weather today is sunny with a high of 75 degrees.',
            'source': 'doc3.pdf',
            'chunk_id': '2',
            'distance': 0.5
        },
        {
            'text': 'Neural networks are computational models inspired by biological neural networks in the brain.',
            'source': 'doc4.pdf',
            'chunk_id': '3',
            'distance': 0.35
        },
        {
            'text': 'Supervised learning requires labeled training data to make predictions.',
            'source': 'doc5.pdf',
            'chunk_id': '4',
            'distance': 0.45
        }
    ]
    
    reranker = Reranker()
    
    # Test query
    query = "What is machine learning?"
    
    print(f"\nQuery: '{query}'")
    print(f"Chunks to re-rank: {len(sample_chunks)}")
    
    # Test individual scoring
    print("\n" + "=" * 80)
    print("TEST 1: Individual scoring")
    print("=" * 80)
    
    reranked = reranker.rerank(query, sample_chunks, top_k=3)
    
    print("\n📊 Re-ranked results:")
    for i, chunk in enumerate(reranked, 1):
        print(f"\n[{i}] Score: {chunk['rerank_score']:.1f}/10")
        print(f"    Source: {chunk['source']}")
        print(f"    Text: {chunk['text'][:100]}...")
    
    # Test batch scoring
    print("\n" + "=" * 80)
    print("TEST 2: Batch scoring")
    print("=" * 80)
    
    reranked_batch = reranker.rerank_batch(query, sample_chunks, top_k=3)
    
    print("\n📊 Batch re-ranked results:")
    for i, chunk in enumerate(reranked_batch, 1):
        print(f"\n[{i}] Score: {chunk['rerank_score']:.1f}/10")
        print(f"    Source: {chunk['source']}")
    
    # Show stats
    print("\n" + "=" * 80)
    print("📊 STATISTICS")
    print("=" * 80)
    stats = reranker.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ RE-RANKER TEST PASSED!")
    print("=" * 80)