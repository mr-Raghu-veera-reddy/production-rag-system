"""
Complete RAG System
Combines retrieval and generation into one system
"""

# from retriever import Retriever
# from qa_generator import QAGenerator
# # from monitoring import Monitor
# from rag_monitor import Monitor
# from typing import Dict
# import time
from src.retriever import Retriever
from src.qa_generator import QAGenerator
from src.rag_monitor import Monitor
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
        temperature: float = 0.3,
        use_advanced_retrieval: bool = False
    ):
        """
        Initialize RAG system
        
        Args:
            model: OpenAI model for generation
            top_k: Number of chunks to retrieve
            temperature: Generation temperature
            use_advanced_retrieval: Use query rewriting + re-ranking
        """
        
        print("=" * 80)
        print("🚀 INITIALIZING RAG SYSTEM")
        print("=" * 80)
        
        # Choose retriever based on mode
        if use_advanced_retrieval:
            from advanced_retriever import AdvancedRetriever
            self.retriever = AdvancedRetriever(
                use_query_rewriting=True,
                use_reranking=True,
                top_k=top_k
            )
            print("✅ Using ADVANCED retrieval (query rewriting + re-ranking)")
        else:
            self.retriever = Retriever(top_k=top_k)
            print("✅ Using BASIC retrieval")
        
        self.generator = QAGenerator(model=model, temperature=temperature)
        self.top_k = top_k
        
        # Add monitoring
        self.monitor = Monitor()
        
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
        
        # Log query with monitoring
        self.monitor.log_query(question, result['answer'], result)
        
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
    
    def show_stats(self, last_n: int = None):
        """
        Show system statistics
        
        Args:
            last_n: Show stats for last N queries only
        """
        self.monitor.print_stats(last_n)
    
    def interactive_mode(self):
        """
        Run RAG system in interactive question-answering mode
        """
        
        print("\n" + "=" * 80)
        print("💬 INTERACTIVE RAG SYSTEM")
        print("=" * 80)
        print("Ask questions about your documents!")
        print("Type 'quit' or 'exit' to stop")
        print("Type 'stats' to see statistics")
        print("=" * 80)
        
        while True:
            # Get question from user
            question = input("\n❓ Your question: ").strip()
            
            # Check for exit
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                # Show final stats
                self.show_stats()
                break
            
            # Check for stats command
            if question.lower() == 'stats':
                self.show_stats()
                continue
            
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
    
    # Test 1: Basic mode
    print("\n" + "=" * 80)
    print("TEST 1: Basic Mode")
    print("=" * 80)
    
    rag_basic = RAGSystem(
        model="gpt-3.5-turbo",
        top_k=5,
        temperature=0.3,
        use_advanced_retrieval=False
    )
    
    result1 = rag_basic.query("What is machine learning?")
    rag_basic.print_result(result1)
    
    # Test 2: Advanced mode
    print("\n" + "=" * 80)
    print("TEST 2: Advanced Mode")
    print("=" * 80)
    
    rag_advanced = RAGSystem(
        model="gpt-3.5-turbo",
        top_k=5,
        temperature=0.3,
        use_advanced_retrieval=True
    )
    
    result2 = rag_advanced.query("What is ML?")  # Short query to test rewriting
    rag_advanced.print_result(result2)
    
    # Show monitoring stats
    print("\n" + "=" * 80)
    print("MONITORING STATISTICS")
    print("=" * 80)
    rag_advanced.show_stats()
    
    print("\n✅ RAG SYSTEM TEST PASSED!")
    print("=" * 80)
    
    # Offer interactive mode
    print("\nWould you like to try interactive mode? (y/n)")
    choice = input("> ").strip().lower()
    
    if choice == 'y':
        # Use basic mode for interactive (faster)
        rag_interactive = RAGSystem(
            model="gpt-3.5-turbo",
            top_k=5,
            temperature=0.3,
            use_advanced_retrieval=False
        )
        rag_interactive.interactive_mode()