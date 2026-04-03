"""
Generate evaluation report
"""

import json
import sys
import os
from datetime import datetime

print("\n" + "=" * 80)
print("📄 GENERATING EVALUATION REPORT")
print("=" * 80)

# Load results
with open('evaluation/results_basic.json', 'r') as f:
    results_basic = json.load(f)

with open('evaluation/results_advanced.json', 'r') as f:
    results_advanced = json.load(f)

# Calculate metrics
def calculate_metrics(results):
    total = len(results)
    
    quality = {
        'accuracy': sum(r['quality_scores']['accuracy'] for r in results) / total,
        'completeness': sum(r['quality_scores']['completeness'] for r in results) / total,
        'relevance': sum(r['quality_scores']['relevance'] for r in results) / total,
        'clarity': sum(r['quality_scores']['clarity'] for r in results) / total,
        'overall': sum(r['quality_scores']['overall'] for r in results) / total
    }
    
    retrieval = {
        'precision': sum(r['retrieval_scores']['precision'] for r in results) / total,
        'recall': sum(r['retrieval_scores']['recall'] for r in results) / total,
        'f1': sum(r['retrieval_scores']['f1'] for r in results) / total
    }
    
    performance = {
        'avg_latency': sum(r['latency'] for r in results) / total,
        'total_cost': sum(r['cost'] for r in results),
        'avg_cost': sum(r['cost'] for r in results) / total
    }
    
    return quality, retrieval, performance

basic_quality, basic_retrieval, basic_perf = calculate_metrics(results_basic)
adv_quality, adv_retrieval, adv_perf = calculate_metrics(results_advanced)

# Helper functions MUST be defined before they are used in the f-string
def _get_best_result(results):
    best = max(results, key=lambda x: x['quality_scores']['overall'])
    return f"""
**Question:** {best['question']}
**Score:** {best['quality_scores']['overall']:.1f}/10
**Latency:** {best['latency']:.2f}s
"""

def _get_most_improved(basic, advanced):
    improvements = []
    for b, a in zip(basic, advanced):
        diff = a['quality_scores']['overall'] - b['quality_scores']['overall']
        improvements.append((diff, a))
    
    best_improvement = max(improvements, key=lambda x: x[0])
    diff, result = best_improvement
    
    return f"""
**Question:** {result['question']}
**Improvement:** +{diff:.1f} points
**Basic:** {result['quality_scores']['overall']-diff:.1f}/10
**Advanced:** {result['quality_scores']['overall']:.1f}/10
"""

# Generate markdown report
report = f"""# RAG System Evaluation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Total Test Cases:** {len(results_basic)}

---

## Executive Summary

This report evaluates the performance of a production-grade RAG (Retrieval Augmented Generation) system across **{len(results_basic)} test queries** in two modes: Basic and Advanced.

### Key Findings

- **Answer Quality:** Advanced mode achieves **{adv_quality['overall']:.2f}/10** vs Basic **{basic_quality['overall']:.2f}/10**
- **Retrieval Accuracy:** Advanced mode F1 score of **{adv_retrieval['f1']:.1%}** vs Basic **{basic_retrieval['f1']:.1%}**
- **Performance:** Basic mode is faster (**{basic_perf['avg_latency']:.2f}s** vs **{adv_perf['avg_latency']:.2f}s**)
- **Cost:** Basic mode is cheaper (**${basic_perf['avg_cost']:.4f}** vs **${adv_perf['avg_cost']:.4f}** per query)

**Recommendation:** Use **Advanced mode** for production use cases where quality matters, **Basic mode** for high-volume scenarios requiring speed.

---

## Answer Quality Metrics

| Metric | Basic Mode | Advanced Mode | Improvement |
|--------|------------|---------------|-------------|
| **Accuracy** | {basic_quality['accuracy']:.2f}/10 | {adv_quality['accuracy']:.2f}/10 | {(adv_quality['accuracy']-basic_quality['accuracy']):.2f} |
| **Completeness** | {basic_quality['completeness']:.2f}/10 | {adv_quality['completeness']:.2f}/10 | {(adv_quality['completeness']-basic_quality['completeness']):.2f} |
| **Relevance** | {basic_quality['relevance']:.2f}/10 | {adv_quality['relevance']:.2f}/10 | {(adv_quality['relevance']-basic_quality['relevance']):.2f} |
| **Clarity** | {basic_quality['clarity']:.2f}/10 | {adv_quality['clarity']:.2f}/10 | {(adv_quality['clarity']-basic_quality['clarity']):.2f} |
| **Overall** | **{basic_quality['overall']:.2f}/10** | **{adv_quality['overall']:.2f}/10** | **{(adv_quality['overall']-basic_quality['overall']):.2f}** |

---

## Retrieval Performance

| Metric | Basic Mode | Advanced Mode | Improvement |
|--------|------------|---------------|-------------|
| **Precision** | {basic_retrieval['precision']:.1%} | {adv_retrieval['precision']:.1%} | {(adv_retrieval['precision']-basic_retrieval['precision'])*100:+.1f}% |
| **Recall** | {basic_retrieval['recall']:.1%} | {adv_retrieval['recall']:.1%} | {(adv_retrieval['recall']-basic_retrieval['recall'])*100:+.1f}% |
| **F1 Score** | **{basic_retrieval['f1']:.1%}** | **{adv_retrieval['f1']:.1%}** | **{(adv_retrieval['f1']-basic_retrieval['f1'])*100:+.1f}%** |

---

## Performance & Cost

| Metric | Basic Mode | Advanced Mode | Difference |
|--------|------------|---------------|------------|
| **Avg Latency** | {basic_perf['avg_latency']:.2f}s | {adv_perf['avg_latency']:.2f}s | {(adv_perf['avg_latency']-basic_perf['avg_latency']):+.2f}s |
| **Avg Cost/Query** | ${basic_perf['avg_cost']:.4f} | ${adv_perf['avg_cost']:.4f} | ${(adv_perf['avg_cost']-basic_perf['avg_cost']):+.4f} |
| **Total Cost ({len(results_basic)} queries)** | ${basic_perf['total_cost']:.4f} | ${adv_perf['total_cost']:.4f} | ${(adv_perf['total_cost']-basic_perf['total_cost']):+.4f} |

---

## Mode Comparison Analysis

### Advanced Mode Advantages
- **+{(adv_quality['overall']-basic_quality['overall']):.2f} points** better answer quality
- **+{(adv_retrieval['f1']-basic_retrieval['f1'])*100:.1f}%** better retrieval accuracy
- Query rewriting improves vague questions
- Re-ranking ensures best chunks are used

### Basic Mode Advantages
- **{basic_perf['avg_latency']:.2f}s faster** response time
- **${(adv_perf['avg_cost']-basic_perf['avg_cost']):.4f} cheaper** per query
- Simpler architecture, easier to debug
- Better for high-volume scenarios

---

## Recommendations

### For Production Use:
- **Default:** Advanced mode (better quality justifies the cost)
- **High-volume:** Basic mode for queries >1000/day
- **Critical queries:** Always use Advanced mode
- **User-facing:** Advanced mode (better UX)

### For Development:
- **Testing:** Basic mode (faster iteration)
- **Evaluation:** Both modes (compare results)

---

## Sample Results

### Best Performing Query
{_get_best_result(results_advanced)}

### Most Improved by Advanced Mode
{_get_most_improved(results_basic, results_advanced)}

---

## Conclusion

The RAG system demonstrates **strong performance** across both modes:
- **Answer quality:** {adv_quality['overall']:.1f}/10 (Advanced) shows production-ready accuracy
- **Retrieval accuracy:** {adv_retrieval['f1']:.1%} F1 score indicates reliable source selection
- **Performance:** Sub-{adv_perf['avg_latency']:.0f}s latency suitable for real-time applications

**Advanced mode is recommended** for production deployment where quality is prioritized over speed.

---

*Generated by RAG System Evaluation Framework*
"""

# Save report
with open('evaluation/EVALUATION_REPORT.md', 'w', encoding='utf-8') as f:
    f.write(report)

print("✅ Report generated: evaluation/EVALUATION_REPORT.md")

# Also create a summary
summary = f"""
RAG SYSTEM EVALUATION SUMMARY
{'=' * 60}

Answer Quality:     {adv_quality['overall']:.2f}/10 (Advanced) | {basic_quality['overall']:.2f}/10 (Basic)
Retrieval F1:       {adv_retrieval['f1']:.1%} (Advanced) | {basic_retrieval['f1']:.1%} (Basic)
Avg Latency:        {adv_perf['avg_latency']:.2f}s (Advanced) | {basic_perf['avg_latency']:.2f}s (Basic)
Avg Cost:           ${adv_perf['avg_cost']:.4f} (Advanced) | ${basic_perf['avg_cost']:.4f} (Basic)

Recommendation: Advanced mode for production
{'=' * 60}
"""

with open('evaluation/SUMMARY.txt', 'w') as f:
    f.write(summary)

print("✅ Summary generated: evaluation/SUMMARY.txt")
print("\n" + summary)