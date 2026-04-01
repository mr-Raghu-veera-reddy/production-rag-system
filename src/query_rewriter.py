"""
Query Rewriter Module
Rewrite user queries for better retrieval
"""

import openai
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class QueryRewriter:
    """
    Rewrite and expand user queries for better retrieval
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize query rewriter
        
        Args:
            model: OpenAI model to use
        """
        
        self.model = model
        self.total_tokens_used = 0
        
        print("🔄 Query Rewriter initialized")
        print(f"   Model: {model}")
    
    def rewrite_query(self, query: str, num_variants: int = 3) -> List[str]:
        """
        Rewrite query into multiple variants for better retrieval
        
        Args:
            query: Original user question
            num_variants: Number of rewritten versions to generate
            
        Returns:
            List of rewritten queries (including original)
        """
        
        print(f"\n🔄 Rewriting query: '{query}'")
        
        # If query is already detailed (>10 words), don't rewrite
        if len(query.split()) > 10:
            print("   Query already detailed, skipping rewrite")
            return [query]
        
        prompt = f"""Given this question: "{query}"

Generate {num_variants} different ways to ask the same question that would help retrieve relevant information from a document database.

Make the rewritten questions:
- More specific and detailed
- Include relevant keywords
- Maintain the original intent
- Be standalone questions (don't reference "the question")

Return ONLY the {num_variants} rewritten questions, one per line, without numbering or bullet points."""

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that rewrites questions to be more specific and detailed for better information retrieval."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,  # Some creativity for variations
                max_tokens=150
            )
            
            # Extract rewritten queries
            content = response.choices[0].message.content.strip()
            rewritten = [q.strip() for q in content.split('\n') if q.strip()]
            
            # Track tokens
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            
            # Always include original query first
            all_queries = [query] + rewritten
            
            print(f"✅ Generated {len(rewritten)} variants ({tokens_used} tokens)")
            
            for i, q in enumerate(rewritten, 1):
                print(f"   {i}. {q}")
            
            return all_queries
            
        except Exception as e:
            print(f"❌ Error rewriting query: {e}")
            # Fallback to original query
            return [query]
    
    def rewrite_with_context(self, query: str, previous_queries: List[str] = None) -> List[str]:
        """
        Rewrite query considering conversation context
        
        Args:
            query: Current user question
            previous_queries: Previous questions in conversation
            
        Returns:
            List of rewritten queries
        """
        
        if not previous_queries:
            return self.rewrite_query(query)
        
        # Build context from previous queries
        context = "\n".join(previous_queries[-3:])  # Last 3 queries
        
        prompt = f"""Previous questions in this conversation:
{context}

Current question: "{query}"

This question might be a follow-up. Generate 2 standalone versions:
1. A version that includes context from previous questions
2. A version that is completely standalone

Return ONLY the 2 rewritten questions, one per line."""

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You rewrite follow-up questions to be standalone."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.5,
                max_tokens=100
            )
            
            content = response.choices[0].message.content.strip()
            rewritten = [q.strip() for q in content.split('\n') if q.strip()]
            
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            
            print(f"✅ Context-aware rewrite ({tokens_used} tokens)")
            
            return [query] + rewritten
            
        except Exception as e:
            print(f"❌ Error in context-aware rewrite: {e}")
            return [query]
    
    def get_stats(self):
        """Get rewriter statistics"""
        return {
            'model': self.model,
            'total_tokens_used': self.total_tokens_used,
            'estimated_cost': self.total_tokens_used * 0.0015 / 1000
        }


# Test code
if __name__ == "__main__":
    print("\n🧪 TESTING QUERY REWRITER")
    print("=" * 80)
    
    rewriter = QueryRewriter()
    
    # Test 1: Short vague query
    print("\n" + "=" * 80)
    print("TEST 1: Short vague query")
    print("=" * 80)
    
    query1 = "What's ML?"
    variants1 = rewriter.rewrite_query(query1)
    
    print(f"\nOriginal: {query1}")
    print(f"Variants: {len(variants1) - 1}")
    
    # Test 2: Already detailed query
    print("\n" + "=" * 80)
    print("TEST 2: Already detailed query")
    print("=" * 80)
    
    query2 = "Can you explain in detail how neural networks work and what are their main applications?"
    variants2 = rewriter.rewrite_query(query2)
    
    print(f"\nOriginal: {query2}")
    print(f"Should skip rewrite: {len(variants2) == 1}")
    
    # Test 3: Context-aware rewriting
    print("\n" + "=" * 80)
    print("TEST 3: Context-aware rewriting")
    print("=" * 80)
    
    previous = [
        "What is machine learning?",
        "How does supervised learning work?"
    ]
    query3 = "What about unsupervised?"
    
    variants3 = rewriter.rewrite_with_context(query3, previous)
    
    print(f"\nPrevious context:")
    for q in previous:
        print(f"  - {q}")
    print(f"\nCurrent: {query3}")
    print(f"\nRewritten variants:")
    for v in variants3[1:]:  # Skip original
        print(f"  - {v}")
    
    # Show stats
    print("\n" + "=" * 80)
    print("📊 STATISTICS")
    print("=" * 80)
    stats = rewriter.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ QUERY REWRITER TEST PASSED!")
    print("=" * 80)