"""
RAG System Evaluator
Evaluate RAG system performance with metrics
"""

import json
import sys
import os

# # Add parent directory to path to import from src
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add parent directory and src directory to path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path, "src"))

from src.rag_system import RAGSystem
import openai
from dotenv import load_dotenv
import time
from typing import List, Dict

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class RAGEvaluator:
    """
    Evaluate RAG system performance
    """
    
    def __init__(self, rag_system: RAGSystem):
        """
        Initialize evaluator
        
        Args:
            rag_system: RAG system to evaluate
        """
        
        self.rag_system = rag_system
        self.results = []
        
        print("📊 RAG Evaluator initialized")
    
    def load_test_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load test dataset from JSON file
        
        Args:
            dataset_path: Path to JSON file
            
        Returns:
            List of test cases
        """
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"✅ Loaded {len(dataset)} test cases")
        
        return dataset
    
    def evaluate_answer_quality(
        self, 
        question: str, 
        generated_answer: str, 
        expected_answer: str
    ) -> Dict:
        """
        Evaluate answer quality using GPT-4 as judge
        
        Args:
            question: The question
            generated_answer: Answer from RAG system
            expected_answer: Expected/ideal answer
            
        Returns:
            Dictionary with scores
        """
        
        prompt = f"""You are an expert evaluator. Rate the generated answer compared to the expected answer.

Question: {question}

Expected Answer: {expected_answer}

Generated Answer: {generated_answer}

Rate the generated answer on these criteria (0-10 scale):
1. ACCURACY: Is the information correct?
2. COMPLETENESS: Does it cover the key points?
3. RELEVANCE: Does it answer the question?
4. CLARITY: Is it well-explained?

Return ONLY a JSON object with these scores:
{{"accuracy": X, "completeness": X, "relevance": X, "clarity": X}}"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=100
            )
            
            # Parse JSON response
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            scores = json.loads(content)
            
            # Calculate overall score
            scores['overall'] = sum(scores.values()) / len(scores)
            
            return scores
            
        except Exception as e:
            print(f"   ⚠️ Error evaluating answer: {e}")
            # Return neutral scores
            return {
                'accuracy': 5.0,
                'completeness': 5.0,
                'relevance': 5.0,
                'clarity': 5.0,
                'overall': 5.0
            }
    
    def evaluate_retrieval(
        self,
        retrieved_sources: List[str],
        expected_sources: List[str]
    ) -> Dict:
        """
        Evaluate retrieval quality
        
        Args:
            retrieved_sources: Sources retrieved by system
            expected_sources: Expected sources
            
        Returns:
            Dictionary with precision, recall, F1
        """
        
        retrieved_set = set(retrieved_sources)
        expected_set = set(expected_sources)
        
        # True positives: sources that should be retrieved and were
        tp = len(retrieved_set & expected_set)
        
        # Precision: Of retrieved sources, how many were relevant?
        precision = tp / len(retrieved_set) if len(retrieved_set) > 0 else 0
        
        # Recall: Of relevant sources, how many were retrieved?
        recall = tp / len(expected_set) if len(expected_set) > 0 else 0
        
        # F1 score: Harmonic mean
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'retrieved_count': len(retrieved_set),
            'expected_count': len(expected_set),
            'correct_count': tp
        }
    
    def evaluate_single_query(self, test_case: Dict) -> Dict:
        """
        Evaluate a single test query
        
        Args:
            test_case: Test case dictionary
            
        Returns:
            Evaluation results
        """
        
        question = test_case['question']
        expected_answer = test_case['expected_answer']
        expected_sources = test_case['expected_sources']
        
        print(f"\n   Testing: {question[:60]}...")
        
        # Get answer from RAG system
        start_time = time.time()
        result = self.rag_system.query(question)
        latency = time.time() - start_time
        
        generated_answer = result['answer']
        retrieved_sources = result['sources']
        
        # Evaluate answer quality
        print(f"   Evaluating answer quality...")
        quality_scores = self.evaluate_answer_quality(
            question,
            generated_answer,
            expected_answer
        )
        
        # Evaluate retrieval
        retrieval_scores = self.evaluate_retrieval(
            retrieved_sources,
            expected_sources
        )
        
        # Combine results
        evaluation = {
            'test_id': test_case['id'],
            'question': question,
            'category': test_case['category'],
            'difficulty': test_case['difficulty'],
            'generated_answer': generated_answer,
            'expected_answer': expected_answer,
            'quality_scores': quality_scores,
            'retrieval_scores': retrieval_scores,
            'latency': latency,
            'tokens_used': result['tokens_used'],
            'cost': result['cost'],
            'retrieved_sources': retrieved_sources,
            'expected_sources': expected_sources
        }
        
        print(f"   ✅ Overall score: {quality_scores['overall']:.1f}/10")
        print(f"   ✅ Retrieval F1: {retrieval_scores['f1']:.2%}")
        print(f"   ⏱️  Latency: {latency:.2f}s")
        
        return evaluation
    
    def run_evaluation(self, dataset_path: str) -> Dict:
        """
        Run complete evaluation on test dataset
        
        Args:
            dataset_path: Path to test dataset JSON
            
        Returns:
            Complete evaluation results
        """
        
        print("\n" + "=" * 80)
        print("🧪 RUNNING RAG SYSTEM EVALUATION")
        print("=" * 80)
        
        # Load dataset
        test_cases = self.load_test_dataset(dataset_path)
        
        # Evaluate each test case
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}]")
            
            evaluation = self.evaluate_single_query(test_case)
            results.append(evaluation)
            
            # Small delay to avoid rate limits
            time.sleep(1)
        
        # Calculate aggregate metrics
        aggregate = self._calculate_aggregate_metrics(results)
        
        # Store results
        self.results = results
        
        # Print summary
        self._print_summary(aggregate)
        
        return {
            'individual_results': results,
            'aggregate_metrics': aggregate
        }
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics from all results"""
        
        total_tests = len(results)
        
        # Quality metrics
        avg_accuracy = sum(r['quality_scores']['accuracy'] for r in results) / total_tests
        avg_completeness = sum(r['quality_scores']['completeness'] for r in results) / total_tests
        avg_relevance = sum(r['quality_scores']['relevance'] for r in results) / total_tests
        avg_clarity = sum(r['quality_scores']['clarity'] for r in results) / total_tests
        avg_overall = sum(r['quality_scores']['overall'] for r in results) / total_tests
        
        # Retrieval metrics
        avg_precision = sum(r['retrieval_scores']['precision'] for r in results) / total_tests
        avg_recall = sum(r['retrieval_scores']['recall'] for r in results) / total_tests
        avg_f1 = sum(r['retrieval_scores']['f1'] for r in results) / total_tests
        
        # Performance metrics
        avg_latency = sum(r['latency'] for r in results) / total_tests
        total_cost = sum(r['cost'] for r in results)
        avg_cost = total_cost / total_tests
        total_tokens = sum(r['tokens_used'] for r in results)
        
        # By category
        categories = {}
        for result in results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result['quality_scores']['overall'])
        
        category_avg = {
            cat: sum(scores) / len(scores)
            for cat, scores in categories.items()
        }
        
        # By difficulty
        difficulties = {}
        for result in results:
            diff = result['difficulty']
            if diff not in difficulties:
                difficulties[diff] = []
            difficulties[diff].append(result['quality_scores']['overall'])
        
        difficulty_avg = {
            diff: sum(scores) / len(scores)
            for diff, scores in difficulties.items()
        }
        
        return {
            'total_tests': total_tests,
            'quality_metrics': {
                'accuracy': avg_accuracy,
                'completeness': avg_completeness,
                'relevance': avg_relevance,
                'clarity': avg_clarity,
                'overall': avg_overall
            },
            'retrieval_metrics': {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1
            },
            'performance_metrics': {
                'avg_latency': avg_latency,
                'total_cost': total_cost,
                'avg_cost': avg_cost,
                'total_tokens': total_tokens
            },
            'by_category': category_avg,
            'by_difficulty': difficulty_avg
        }
    
    def _print_summary(self, aggregate: Dict):
        """Print evaluation summary"""
        
        print("\n" + "=" * 80)
        print("📊 EVALUATION SUMMARY")
        print("=" * 80)
        
        print(f"\n📈 Answer Quality (0-10 scale):")
        print(f"  Accuracy:     {aggregate['quality_metrics']['accuracy']:.2f}/10")
        print(f"  Completeness: {aggregate['quality_metrics']['completeness']:.2f}/10")
        print(f"  Relevance:    {aggregate['quality_metrics']['relevance']:.2f}/10")
        print(f"  Clarity:      {aggregate['quality_metrics']['clarity']:.2f}/10")
        print(f"  Overall:      {aggregate['quality_metrics']['overall']:.2f}/10")
        
        print(f"\n🎯 Retrieval Accuracy:")
        print(f"  Precision: {aggregate['retrieval_metrics']['precision']:.2%}")
        print(f"  Recall:    {aggregate['retrieval_metrics']['recall']:.2%}")
        print(f"  F1 Score:  {aggregate['retrieval_metrics']['f1']:.2%}")
        
        print(f"\n⚡ Performance:")
        print(f"  Avg Latency:  {aggregate['performance_metrics']['avg_latency']:.2f}s")
        print(f"  Total Cost:   ${aggregate['performance_metrics']['total_cost']:.4f}")
        print(f"  Avg Cost:     ${aggregate['performance_metrics']['avg_cost']:.4f}")
        print(f"  Total Tokens: {aggregate['performance_metrics']['total_tokens']:,}")
        
        print(f"\n📂 By Category:")
        for cat, score in aggregate['by_category'].items():
            print(f"  {cat:15s}: {score:.2f}/10")
        
        print(f"\n📊 By Difficulty:")
        for diff, score in aggregate['by_difficulty'].items():
            print(f"  {diff:15s}: {score:.2f}/10")
        
        print("=" * 80)
    
    def save_results(self, output_file: str):
        """
        Save evaluation results to JSON file
        
        Args:
            output_file: Path to output file
        """
        
        if not self.results:
            print("❌ No results to save. Run evaluation first.")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to {output_file}")


# Test
if __name__ == "__main__":
    print("\n🧪 TESTING EVALUATOR")
    print("=" * 80)
    
    # Create RAG system
    from src.rag_system import RAGSystem
    
    rag = RAGSystem(
        model="gpt-3.5-turbo",
        top_k=5,
        use_advanced_retrieval=False
    )
    
    # Create evaluator
    evaluator = RAGEvaluator(rag)
    
    # Run evaluation
    results = evaluator.run_evaluation("evaluation/test_dataset.json")
    
    # Save results
    evaluator.save_results("evaluation/results.json")
    
    print("\n✅ EVALUATOR TEST COMPLETE!")