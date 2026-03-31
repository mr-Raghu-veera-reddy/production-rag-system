"""
QA Generator Module
Generate answers using LLM with retrieved context
"""

import openai
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class QAGenerator:
    """
    Generate answers to questions using LLM with context
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.3):
        """
        Initialize QA generator
        
        Args:
            model: OpenAI model to use
                - gpt-3.5-turbo: Fast, cheap ($0.0015/1K tokens)
                - gpt-4: Best quality, expensive ($0.03/1K tokens)
            temperature: Creativity level (0 = focused, 1 = creative)
        """
        
        self.model = model
        self.temperature = temperature
        self.total_tokens_used = 0
        
        print(f"🤖 QA Generator initialized")
        print(f"   Model: {model}")
        print(f"   Temperature: {temperature}")
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for LLM with context and query
        
        Args:
            query: User's question
            context: Retrieved context chunks
            
        Returns:
            Formatted prompt
        """
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer the question using ONLY the information from the context above
- If the context doesn't contain enough information to answer, say "I don't have enough information in the provided documents to answer this question"
- Cite the sources by mentioning the document names (e.g., "According to document.pdf...")
- Be concise and accurate
- If multiple documents mention the topic, synthesize the information

ANSWER:"""
        
        return prompt
    
    def generate_answer(self, query: str, chunks: List[Dict]) -> Dict:
        """
        Generate answer for query using retrieved chunks
        
        Args:
            query: User's question
            chunks: Retrieved chunks from Retriever
            
        Returns:
            Dictionary with answer and metadata
        """
        
        print(f"\n🤖 Generating answer for: '{query}'")
        print(f"   Using {len(chunks)} context chunks...")
        
        # Combine chunks into context
        context = "\n\n".join([
            f"[Source: {chunk['source']}]\n{chunk['text']}"
            for chunk in chunks
        ])
        
        # Create prompt
        prompt = self.create_prompt(query, context)
        
        try:
            # Call OpenAI API
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context. Always cite your sources."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=500  # Limit response length
            )
            
            # Extract answer
            answer = response.choices[0].message.content
            
            # Track tokens
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            
            # Calculate cost (GPT-3.5-turbo pricing)
            cost = tokens_used * 0.0015 / 1000
            
            print(f"✅ Answer generated ({tokens_used} tokens, ${cost:.4f})")
            
            # Get unique sources
            sources = list(set([chunk['source'] for chunk in chunks]))
            
            result = {
                'query': query,
                'answer': answer,
                'sources': sources,
                'chunks_used': len(chunks),
                'tokens_used': tokens_used,
                'cost': cost,
                'model': self.model
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            
            return {
                'query': query,
                'answer': f"Error generating answer: {str(e)}",
                'sources': [],
                'chunks_used': 0,
                'tokens_used': 0,
                'cost': 0,
                'model': self.model,
                'error': str(e)
            }
    
    def print_answer(self, result: Dict):
        """
        Pretty print the answer result
        
        Args:
            result: Result dictionary from generate_answer
        """
        
        print("\n" + "=" * 80)
        print("💬 ANSWER")
        print("=" * 80)
        print(result['answer'])
        
        print("\n" + "=" * 80)
        print("📚 SOURCES")
        print("=" * 80)
        for source in result['sources']:
            print(f"  • {source}")
        
        print("\n" + "=" * 80)
        print("📊 METADATA")
        print("=" * 80)
        print(f"  Chunks used: {result['chunks_used']}")
        print(f"  Tokens used: {result['tokens_used']}")
        print(f"  Cost: ${result['cost']:.4f}")
        print(f"  Model: {result['model']}")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about QA generation
        
        Returns:
            Dictionary with stats
        """
        
        return {
            'model': self.model,
            'temperature': self.temperature,
            'total_tokens_used': self.total_tokens_used,
            'total_cost': self.total_tokens_used * 0.0015 / 1000
        }


# Test code
if __name__ == "__main__":
    print("\n🧪 TESTING QA GENERATOR")
    print("=" * 80)
    
    # First, get some context chunks
    from retriever import Retriever
    
    retriever = Retriever()
    generator = QAGenerator()
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What are the applications of deep learning?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST QUERY {i}/{len(test_queries)}")
        print('=' * 80)
        
        # Retrieve context
        chunks = retriever.retrieve(query, top_k=3)
        
        # Generate answer
        result = generator.generate_answer(query, chunks)
        
        # Print answer
        generator.print_answer(result)
    
    # Show overall stats
    print("\n" + "=" * 80)
    print("📊 OVERALL STATISTICS")
    print("=" * 80)
    stats = generator.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✅ QA GENERATOR TEST PASSED!")
    print("=" * 80)