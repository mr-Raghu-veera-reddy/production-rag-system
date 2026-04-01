"""
Monitoring Module
Track RAG system performance and costs
"""

import json
import os
from datetime import datetime
from typing import Dict, List
import pandas as pd

class Monitor:
    """
    Monitor RAG system metrics and costs
    """
    
    def __init__(self, log_file: str = "logs/rag_metrics.jsonl"):
        """
        Initialize monitor
        
        Args:
            log_file: Path to log file
        """
        
        self.log_file = log_file
        
        # Create logs directory
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        print(f"📊 Monitor initialized")
        print(f"   Log file: {log_file}")
    
    def log_query(
        self,
        query: str,
        answer: str,
        result: Dict
    ):
        """
        Log a query with all metrics
        
        Args:
            query: User's question
            answer: Generated answer
            result: Full result dictionary from RAG system
        """
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': answer,
            'answer_length': len(answer),
            'sources': result.get('sources', []),
            'num_sources': len(result.get('sources', [])),
            'chunks_used': result.get('chunks_used', 0),
            'tokens_used': result.get('tokens_used', 0),
            'cost': result.get('cost', 0),
            'latency': result.get('latency', 0),
            'model': result.get('model', 'unknown'),
            'error': result.get('error', None)
        }
        
        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_stats(self, last_n: int = None) -> Dict:
        """
        Get aggregate statistics
        
        Args:
            last_n: Only analyze last N queries (None = all)
            
        Returns:
            Dictionary with statistics
        """
        
        if not os.path.exists(self.log_file):
            return {
                'error': 'No logs found'
            }
        
        # Read logs
        with open(self.log_file, 'r') as f:
            logs = [json.loads(line) for line in f]
        
        if not logs:
            return {
                'error': 'No queries logged yet'
            }
        
        # Take last N if specified
        if last_n:
            logs = logs[-last_n:]
        
        # Calculate stats
        total_queries = len(logs)
        successful_queries = len([log for log in logs if not log.get('error')])
        failed_queries = total_queries - successful_queries
        
        total_tokens = sum(log.get('tokens_used', 0) for log in logs)
        total_cost = sum(log.get('cost', 0) for log in logs)
        avg_latency = sum(log.get('latency', 0) for log in logs) / total_queries
        avg_chunks = sum(log.get('chunks_used', 0) for log in logs) / total_queries
        
        return {
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'failed_queries': failed_queries,
            'success_rate': successful_queries / total_queries if total_queries > 0 else 0,
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'avg_tokens_per_query': total_tokens / total_queries if total_queries > 0 else 0,
            'avg_cost_per_query': total_cost / total_queries if total_queries > 0 else 0,
            'avg_latency': avg_latency,
            'avg_chunks_used': avg_chunks
        }
    
    def print_stats(self, last_n: int = None):
        """
        Print statistics in a nice format
        
        Args:
            last_n: Only show stats for last N queries
        """
        
        stats = self.get_stats(last_n)
        
        if 'error' in stats:
            print(f"❌ {stats['error']}")
            return
        
        print("\n" + "=" * 80)
        print("📊 RAG SYSTEM STATISTICS")
        if last_n:
            print(f"(Last {last_n} queries)")
        print("=" * 80)
        
        print(f"\n📈 Usage:")
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  Successful: {stats['successful_queries']}")
        print(f"  Failed: {stats['failed_queries']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        
        print(f"\n💰 Costs:")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Total cost: ${stats['total_cost']:.4f}")
        print(f"  Avg cost per query: ${stats['avg_cost_per_query']:.4f}")
        
        print(f"\n⚡ Performance:")
        print(f"  Avg latency: {stats['avg_latency']:.2f}s")
        print(f"  Avg chunks used: {stats['avg_chunks_used']:.1f}")
        print(f"  Avg tokens per query: {stats['avg_tokens_per_query']:.0f}")
        
        print("=" * 80)
    
    def export_to_csv(self, output_file: str = "logs/rag_metrics.csv"):
        """
        Export logs to CSV for analysis
        
        Args:
            output_file: Path to output CSV
        """
        
        if not os.path.exists(self.log_file):
            print("❌ No logs to export")
            return
        
        # Read logs
        with open(self.log_file, 'r') as f:
            logs = [json.loads(line) for line in f]
        
        # Convert to DataFrame
        df = pd.DataFrame(logs)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        print(f"✅ Exported {len(logs)} queries to {output_file}")
    
    def get_recent_queries(self, n: int = 10) -> List[Dict]:
        """
        Get N most recent queries
        
        Args:
            n: Number of queries to return
            
        Returns:
            List of query dictionaries
        """
        
        if not os.path.exists(self.log_file):
            return []
        
        with open(self.log_file, 'r') as f:
            logs = [json.loads(line) for line in f]
        
        return logs[-n:]
    
    def clear_logs(self):
        """Clear all logs"""
        
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
            print("🗑️ Logs cleared")
        else:
            print("ℹ️ No logs to clear")


# Test code
if __name__ == "__main__":
    print("\n🧪 TESTING MONITOR")
    print("=" * 80)
    
    monitor = Monitor(log_file="logs/test_metrics.jsonl")
    
    # Simulate some queries
    print("\nSimulating queries...")
    
    for i in range(5):
        monitor.log_query(
            query=f"Test query {i+1}",
            answer=f"Test answer {i+1} " * 20,
            result={
                'sources': ['doc1.pdf', 'doc2.pdf'],
                'chunks_used': 5,
                'tokens_used': 250 + i * 10,
                'cost': 0.0005 + i * 0.0001,
                'latency': 3.5 + i * 0.2,
                'model': 'gpt-3.5-turbo'
            }
        )
    
    print("✅ Logged 5 queries")
    
    # Show stats
    monitor.print_stats()
    
    # Show recent queries
    print("\n" + "=" * 80)
    print("📝 RECENT QUERIES")
    print("=" * 80)
    
    recent = monitor.get_recent_queries(n=3)
    for i, query in enumerate(recent, 1):
        print(f"\n[{i}] {query['query']}")
        print(f"    Latency: {query['latency']:.2f}s")
        print(f"    Cost: ${query['cost']:.4f}")
    
    # Export to CSV
    print("\n" + "=" * 80)
    monitor.export_to_csv("logs/test_export.csv")
    
    # Clean up test files
    print("\nCleaning up test files...")
    if os.path.exists("logs/test_metrics.jsonl"):
        os.remove("logs/test_metrics.jsonl")
    if os.path.exists("logs/test_export.csv"):
        os.remove("logs/test_export.csv")
    
    print("\n✅ MONITOR TEST PASSED!")
    print("=" * 80)