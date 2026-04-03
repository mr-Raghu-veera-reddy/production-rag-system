"""
Compare Basic vs Advanced RAG modes
"""

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path, "src"))

from src.rag_system import RAGSystem
from evaluator import RAGEvaluator
import json

print("\n" + "=" * 80)
print("🆚 COMPARING BASIC VS ADVANCED MODES")
print("=" * 80)

# Test Basic Mode
print("\n" + "─" * 80)
print("📊 EVALUATING BASIC MODE")
print("─" * 80)

rag_basic = RAGSystem(
    model="gpt-3.5-turbo",
    top_k=5,
    use_advanced_retrieval=False
)

evaluator_basic = RAGEvaluator(rag_basic)
results_basic = evaluator_basic.run_evaluation("evaluation/test_dataset.json")
evaluator_basic.save_results("evaluation/results_basic.json")

# Test Advanced Mode
print("\n" + "─" * 80)
print("📊 EVALUATING ADVANCED MODE")
print("─" * 80)

rag_advanced = RAGSystem(
    model="gpt-3.5-turbo",
    top_k=5,
    use_advanced_retrieval=True
)

evaluator_advanced = RAGEvaluator(rag_advanced)
results_advanced = evaluator_advanced.run_evaluation("evaluation/test_dataset.json")
evaluator_advanced.save_results("evaluation/results_advanced.json")

# Compare results
print("\n" + "=" * 80)
print("📊 COMPARISON SUMMARY")
print("=" * 80)

basic_metrics = results_basic['aggregate_metrics']
advanced_metrics = results_advanced['aggregate_metrics']

print("\n📈 Answer Quality:")
print(f"{'Metric':<15} {'Basic':<10} {'Advanced':<10} {'Difference':<10}")
print("-" * 50)

for metric in ['accuracy', 'completeness', 'relevance', 'clarity', 'overall']:
    basic_val = basic_metrics['quality_metrics'][metric]
    advanced_val = advanced_metrics['quality_metrics'][metric]
    diff = advanced_val - basic_val
    
    sign = "+" if diff > 0 else ""
    print(f"{metric:<15} {basic_val:<10.2f} {advanced_val:<10.2f} {sign}{diff:<10.2f}")

print("\n🎯 Retrieval Metrics:")
print(f"{'Metric':<15} {'Basic':<10} {'Advanced':<10} {'Difference':<10}")
print("-" * 50)

for metric in ['precision', 'recall', 'f1']:
    basic_val = basic_metrics['retrieval_metrics'][metric]
    advanced_val = advanced_metrics['retrieval_metrics'][metric]
    diff = advanced_val - basic_val
    
    sign = "+" if diff > 0 else ""
    print(f"{metric:<15} {basic_val*100:<10.1f}% {advanced_val*100:<10.1f}% {sign}{diff*100:<10.1f}%")

print("\n⚡ Performance:")
print(f"{'Metric':<15} {'Basic':<15} {'Advanced':<15} {'Difference':<15}")
print("-" * 60)

lat_basic = basic_metrics['performance_metrics']['avg_latency']
lat_advanced = advanced_metrics['performance_metrics']['avg_latency']
lat_diff = lat_advanced - lat_basic

cost_basic = basic_metrics['performance_metrics']['avg_cost']
cost_advanced = advanced_metrics['performance_metrics']['avg_cost']
cost_diff = cost_advanced - cost_basic

print(f"{'Latency':<15} {lat_basic:<15.2f}s {lat_advanced:<15.2f}s {lat_diff:+15.2f}s")
print(f"{'Cost':<15} ${cost_basic:<14.4f} ${cost_advanced:<14.4f} ${cost_diff:+14.4f}")

# Determine winner
quality_winner = "Advanced" if advanced_metrics['quality_metrics']['overall'] > basic_metrics['quality_metrics']['overall'] else "Basic"
retrieval_winner = "Advanced" if advanced_metrics['retrieval_metrics']['f1'] > basic_metrics['retrieval_metrics']['f1'] else "Basic"

print("\n" + "=" * 80)
print("🏆 WINNER")
print("=" * 80)
print(f"Quality:   {quality_winner}")
print(f"Retrieval: {retrieval_winner}")
print(f"Speed:     Basic (faster by {lat_diff:.2f}s)")
print(f"Cost:      Basic (cheaper by ${cost_diff:.4f})")

# Overall recommendation
if advanced_metrics['quality_metrics']['overall'] > basic_metrics['quality_metrics']['overall'] + 0.5:
    print("\n✅ RECOMMENDATION: Use Advanced mode (significantly better quality)")
elif lat_diff > 2.0:
    print("\n✅ RECOMMENDATION: Use Basic mode (similar quality, much faster)")
else:
    print("\n✅ RECOMMENDATION: Advanced mode for important queries, Basic for speed")

print("=" * 80)