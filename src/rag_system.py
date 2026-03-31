"""
Complete RAG System
Combines retrieval and generation into one system
"""

from .retriever import Retriever
from .qa_generator import QAGenerator
from typing import Dict
import time

class RAGSystem:
    """
    Complete Retrieval Augmented Generation system
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        top_k: int = 5,
        temperature: float = 0.3
    ):
        """
        Initialize RAG system
        
        Args:
            model: OpenAI model for generation
            top_k: Number of chunks to retrieve
            temperature: Generation temperature
        """
        
        print("=" * 80)
        print("🚀 INITIALIZING RAG SYSTEM")
        print("=" * 80)
        
        self.retriever = Retriever(top_k=top_k)
        self.generator = QAGenerator(model=model, temperature=temperature)
        self.top_k = top_k
        
        print("\n✅ RAG System ready!")
        print("=" * 80)
    
    def query(self, question: str, top_k: int = None) -> Dict:
        """
        Answer a question using RAG pipeline
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve (optional override)
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        
        start_time = time.time()
        
        print("\n" + "=" * 80)
        print(f"❓ QUESTION: {question}")
        print("=" * 80)
        
        # Step 1: Retrieve relevant chunks
        print("\n[Step 1/2] Retrieving relevant context...")
        chunks = self.retriever.retrieve(question, top_k=top_k)
        
        if not chunks:
            return {
                'query': question,
                'answer': "I couldn't find any relevant information in the documents.",
                'sources': [],
                'chunks_used': 0,
                'tokens_used': 0,
                'cost': 0,
                'latency': 0,
                'error': 'No chunks retrieved'
            }
        
        # Step 2: Generate answer
        print("\n[Step 2/2] Generating answer...")
        result = self.generator.generate_answer(question, chunks)
        
        # Add latency
        end_time = time.time()
        result['latency'] = end_time - start_time
        
        # Add retrieved chunks for reference
        result['retrieved_chunks'] = chunks
        
        return result
    
    def print_result(self, result: Dict, show_chunks: bool = False):
        """
        Pretty print the RAG result
        
        Args:
            result: Result from query()
            show_chunks: Whether to show retrieved chunks
        """
        
        print("\n" + "=" * 80)
        print("💡 ANSWER")
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
        print(f"  Chunks retrieved: {result['chunks_used']}")
        print(f"  Tokens used: {result['tokens_used']}")
        print(f"  Cost: ${result['cost']:.4f}")
        print(f"  Latency: {result['latency']:.2f}s")
        
        if show_chunks and 'retrieved_chunks' in result:
            print("\n" + "=" * 80)
            print("📄 RETRIEVED CHUNKS")
            print("=" * 80)
            for chunk in result['retrieved_chunks']:
                print(f"\n[{chunk['source']} - Chunk {chunk['chunk_id']}]")
                print(f"Distance: {chunk['distance']:.4f}")
                print(f"{chunk['text'][:200]}...")
                print("-" * 80)
    
    def interactive_mode(self):
        """
        Run RAG system in interactive question-answering mode
        """
        
        print("\n" + "=" * 80)
        print("💬 INTERACTIVE RAG SYSTEM")
        print("=" * 80)
        print("Ask questions about your documents!")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 80)
        
        while True:
            # Get question from user
            question = input("\n❓ Your question: ").strip()
            
            # Check for exit
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Skip empty questions
            if not question:
                continue
            
            # Get answer
            result = self.query(question)
            
            # Print result
            self.print_result(result)


# Main execution
if __name__ == "__main__":
    print("\n🧪 TESTING COMPLETE RAG SYSTEM")
    print("=" * 80)
    
    # Create RAG system
    rag = RAGSystem(
        model="gpt-3.5-turbo",
        top_k=5,
        temperature=0.3
    )
    
    # Test queries
    test_queries = [
        "What is machine learning and how does it work?",
        "Explain the difference between supervised and unsupervised learning",
        "What are neural networks?",
        "What are some applications of deep learning?"
    ]
    
    print("\n" + "=" * 80)
    print("🧪 RUNNING TEST QUERIES")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'#' * 80}")
        print(f"# TEST {i}/{len(test_queries)}")
        print('#' * 80)
        
        result = rag.query(query)
        rag.print_result(result, show_chunks=False)
    
    print("\n\n" + "=" * 80)
    print("✅ RAG SYSTEM TEST PASSED!")
    print("=" * 80)
    
    # Offer interactive mode
    print("\nWould you like to try interactive mode? (y/n)")
    choice = input("> ").strip().lower()
    
    if choice == 'y':
        rag.interactive_mode()