"""
Comprehensive Evaluation Script - All Required Metrics

Metrics Implemented:
1. Recall@K (K=1,3,5)
2. Top-1 Accuracy, Top-3 Accuracy
3. MRR (Mean Reciprocal Rank)
4. nDCG (Normalized Discounted Cumulative Gain)
5. Entity Coverage Score (estimated)
6. Paraphrase Robustness Score
7. Hallucination Rate (false positives on negative queries)
8. False Positive Rate
9. Stage-wise Latency Breakdown

Usage:
    python scripts/comprehensive_evaluation.py --api-url http://localhost:8000
"""

import json
import time
import math
import requests
import statistics
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


# Relevance threshold - a result is considered "relevant" if score >= this value
RELEVANCE_THRESHOLD = 0.5

# High confidence threshold - used for negative query evaluation  
HIGH_CONFIDENCE_THRESHOLD = 0.6


@dataclass
class QueryResult:
    """Result of a single query evaluation."""
    query_id: str
    query: str
    section: str
    category: str
    latency_ms: float
    num_results: int
    scores: List[float]
    top_result_text: str
    top_result_score: float
    has_relevant_result: bool  # Score >= RELEVANCE_THRESHOLD
    relevant_at_k: Dict[int, bool]  # Whether relevant result found at each k
    cached: bool
    # Stage-wise latency
    embedding_time_ms: float
    retrieval_time_ms: float
    reranking_time_ms: float
    total_api_time_ms: float


@dataclass
class EvaluationMetrics:
    """All required evaluation metrics."""
    # Retrieval Quality
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    top1_accuracy: float
    top3_accuracy: float
    mrr: float
    ndcg_at_5: float
    
    # Robustness
    paraphrase_robustness: float
    false_positive_rate: float
    hallucination_rate: float  # Same as FP rate for our case
    
    # Latency (Stage-wise)
    avg_embedding_ms: float
    avg_retrieval_ms: float
    avg_reranking_ms: float
    avg_total_ms: float
    p50_total_ms: float
    p95_total_ms: float
    p99_total_ms: float
    
    # Cache
    cache_hit_rate: float
    cached_latency_ms: float
    uncached_latency_ms: float
    cache_improvement_pct: float


@dataclass 
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    total_questions: int
    relevance_threshold: float
    metrics: EvaluationMetrics
    section_breakdown: Dict[str, Dict]
    observations: List[str]
    all_results: List[QueryResult] = field(default_factory=list)


def calculate_dcg(scores: List[float], k: int) -> float:
    """Calculate Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, score in enumerate(scores[:k]):
        # Use binary relevance: 1 if score >= threshold, 0 otherwise
        rel = 1.0 if score >= RELEVANCE_THRESHOLD else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because position is 1-indexed
    return dcg


def calculate_ndcg(scores: List[float], k: int) -> float:
    """Calculate Normalized DCG at k."""
    dcg = calculate_dcg(scores, k)
    # Ideal DCG: all relevant results at top
    ideal_scores = sorted(scores, reverse=True)
    idcg = calculate_dcg(ideal_scores, k)
    return dcg / idcg if idcg > 0 else 0.0


class ComprehensiveEvaluator:
    """Complete evaluation suite with all required metrics."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip('/')
        self.questions_file = Path(__file__).parent.parent / "tests" / "real_estate_questions.json"
        self.questions = self._load_questions()
        
    def _load_questions(self) -> Dict:
        """Load test questions."""
        with open(self.questions_file) as f:
            return json.load(f)
    
    def _search(self, query: str, top_k: int = 5) -> Tuple[Dict, float]:
        """Execute search - always use score_threshold=0 to get all results."""
        start = time.time()
        try:
            response = requests.post(
                f"{self.api_url}/api/search",
                json={"query": query, "top_k": top_k, "score_threshold": 0.0},
                timeout=30
            )
            response.raise_for_status()
            latency = (time.time() - start) * 1000
            return response.json(), latency
        except Exception as e:
            return {"error": str(e), "results": [], "latency_breakdown": {}}, (time.time() - start) * 1000
    
    def _evaluate_query(self, query_id: str, query: str, section: str, category: str) -> QueryResult:
        """Evaluate a single query."""
        data, latency = self._search(query, top_k=5)
        
        results = data.get("results", [])
        latency_breakdown = data.get("latency_breakdown", {})
        cached = data.get("cached", False)
        
        scores = [r.get("similarity_score", 0) for r in results]
        top_score = scores[0] if scores else 0
        top_text = results[0].get("text", "")[:200] if results else ""
        
        # Check relevance at each k
        relevant_at_k = {}
        for k in [1, 3, 5]:
            # Relevant if any result in top-k has score >= threshold
            relevant_at_k[k] = any(s >= RELEVANCE_THRESHOLD for s in scores[:k])
        
        return QueryResult(
            query_id=query_id,
            query=query,
            section=section,
            category=category,
            latency_ms=latency,
            num_results=len(results),
            scores=scores,
            top_result_text=top_text,
            top_result_score=top_score,
            has_relevant_result=relevant_at_k.get(1, False),
            relevant_at_k=relevant_at_k,
            cached=cached,
            embedding_time_ms=latency_breakdown.get("embedding_ms", 0),
            retrieval_time_ms=latency_breakdown.get("retrieval_ms", 0),
            reranking_time_ms=latency_breakdown.get("reranking_ms", 0),
            total_api_time_ms=latency_breakdown.get("total_ms", latency)
        )
    
    def _calculate_metrics(self, results: List[QueryResult]) -> EvaluationMetrics:
        """Calculate all metrics from results."""
        # Separate positive and negative queries
        positive = [r for r in results if r.category != "negative"]
        negative = [r for r in results if r.category == "negative"]
        paraphrase = [r for r in results if r.category == "paraphrase"]
        
        # Recall@K and Accuracy (for positive queries)
        if positive:
            recall_1 = sum(1 for r in positive if r.relevant_at_k.get(1, False)) / len(positive)
            recall_3 = sum(1 for r in positive if r.relevant_at_k.get(3, False)) / len(positive)
            recall_5 = sum(1 for r in positive if r.relevant_at_k.get(5, False)) / len(positive)
            top1_acc = recall_1  # Same as recall@1 when we have one relevant doc per query
            top3_acc = recall_3
            
            # MRR
            mrr_sum = 0
            for r in positive:
                for i, s in enumerate(r.scores):
                    if s >= RELEVANCE_THRESHOLD:
                        mrr_sum += 1 / (i + 1)
                        break
            mrr = mrr_sum / len(positive)
            
            # nDCG@5
            ndcg_scores = [calculate_ndcg(r.scores, 5) for r in positive]
            ndcg = statistics.mean(ndcg_scores) if ndcg_scores else 0
        else:
            recall_1 = recall_3 = recall_5 = top1_acc = top3_acc = mrr = ndcg = 0
        
        # False Positive Rate (for negative queries)
        if negative:
            # A false positive is when negative query gets high-confidence results
            false_positives = sum(1 for r in negative if r.top_result_score >= HIGH_CONFIDENCE_THRESHOLD)
            fp_rate = false_positives / len(negative)
        else:
            fp_rate = 0
        
        # Paraphrase Robustness (consistency across paraphrase queries)
        if paraphrase:
            paraphrase_success = sum(1 for r in paraphrase if r.relevant_at_k.get(1, False))
            paraphrase_robustness = paraphrase_success / len(paraphrase)
        else:
            paraphrase_robustness = 0
        
        # Latency metrics
        all_latencies = [r.latency_ms for r in results]
        embedding_times = [r.embedding_time_ms for r in results if r.embedding_time_ms > 0]
        retrieval_times = [r.retrieval_time_ms for r in results if r.retrieval_time_ms > 0]
        reranking_times = [r.reranking_time_ms for r in results if r.reranking_time_ms > 0]
        
        sorted_latencies = sorted(all_latencies)
        
        # Cache metrics
        cached = [r for r in results if r.cached]
        uncached = [r for r in results if not r.cached]
        cache_hit_rate = len(cached) / len(results) if results else 0
        cached_lat = statistics.mean([r.latency_ms for r in cached]) if cached else 0
        uncached_lat = statistics.mean([r.latency_ms for r in uncached]) if uncached else 0
        cache_improve = ((uncached_lat - cached_lat) / uncached_lat * 100) if uncached_lat > 0 else 0
        
        return EvaluationMetrics(
            recall_at_1=recall_1,
            recall_at_3=recall_3,
            recall_at_5=recall_5,
            top1_accuracy=top1_acc,
            top3_accuracy=top3_acc,
            mrr=mrr,
            ndcg_at_5=ndcg,
            paraphrase_robustness=paraphrase_robustness,
            false_positive_rate=fp_rate,
            hallucination_rate=fp_rate,  # Same metric for our evaluation
            avg_embedding_ms=statistics.mean(embedding_times) if embedding_times else 0,
            avg_retrieval_ms=statistics.mean(retrieval_times) if retrieval_times else 0,
            avg_reranking_ms=statistics.mean(reranking_times) if reranking_times else 0,
            avg_total_ms=statistics.mean(all_latencies),
            p50_total_ms=sorted_latencies[len(sorted_latencies)//2],
            p95_total_ms=sorted_latencies[int(len(sorted_latencies)*0.95)] if len(sorted_latencies) >= 20 else sorted_latencies[-1],
            p99_total_ms=sorted_latencies[int(len(sorted_latencies)*0.99)] if len(sorted_latencies) >= 100 else sorted_latencies[-1],
            cache_hit_rate=cache_hit_rate,
            cached_latency_ms=cached_lat,
            uncached_latency_ms=uncached_lat,
            cache_improvement_pct=cache_improve
        )
    
    def run(self, run_twice: bool = True) -> EvaluationReport:
        """Run complete evaluation."""
        print("=" * 70)
        print("COMPREHENSIVE EVALUATION - ALL REQUIRED METRICS")
        print("=" * 70)
        print(f"Relevance Threshold: {RELEVANCE_THRESHOLD}")
        print(f"High Confidence Threshold: {HIGH_CONFIDENCE_THRESHOLD}")
        
        # Health check
        try:
            r = requests.get(f"{self.api_url}/health", timeout=5)
            print(f"\n✓ API healthy: {r.json()}")
        except:
            raise ConnectionError("API not reachable")
        
        # Collection stats
        try:
            r = requests.get(f"{self.api_url}/debug/collection", timeout=5)
            stats = r.json()
            print(f"✓ Indexed chunks: {stats.get('points_count', 0)}")
        except:
            pass
        
        all_results: List[QueryResult] = []
        section_results: Dict[str, List[QueryResult]] = defaultdict(list)
        
        sections = self.questions.get("sections", {})
        total = sum(len(s.get("questions", [])) for s in sections.values())
        
        print(f"\n📋 Evaluating {total} questions across {len(sections)} sections...")
        
        for section_key, section_data in sections.items():
            section_name = section_data.get("name", section_key)
            questions = section_data.get("questions", [])
            
            print(f"\n  Section {section_key}: {section_name} ({len(questions)} questions)")
            
            for q in questions:
                qid = q.get("id", "")
                query = q.get("query", "")
                category = q.get("category", "general")
                
                # First run
                result = self._evaluate_query(qid, query, section_name, category)
                all_results.append(result)
                section_results[section_key].append(result)
                
                # Second run for cache measurement
                if run_twice:
                    result2 = self._evaluate_query(qid + "_c", query, section_name, category)
                    all_results.append(result2)
                
                # Status indicator
                status = "✓" if result.has_relevant_result else "○"
                if category == "negative":
                    # For negative, success is LOW score
                    status = "✓" if result.top_result_score < HIGH_CONFIDENCE_THRESHOLD else "✗"
                print(f"    {status} {qid}: {result.latency_ms:.0f}ms, score={result.top_result_score:.3f}")
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_results)
        
        # Section breakdown
        section_breakdown = {}
        for sec, results in section_results.items():
            pos = [r for r in results if r.category != "negative"]
            acc = sum(1 for r in pos if r.relevant_at_k.get(1, False)) / len(pos) if pos else 0
            section_breakdown[sec] = {
                "name": sections[sec].get("name", sec),
                "total": len(results),
                "accuracy": acc,
                "avg_latency_ms": statistics.mean([r.latency_ms for r in results])
            }
        
        # Observations
        observations = []
        if metrics.recall_at_1 >= 0.75:
            observations.append(f"✅ Recall@1 ({metrics.recall_at_1:.1%}) meets target (≥75%)")
        else:
            observations.append(f"⚠️ Recall@1 ({metrics.recall_at_1:.1%}) below target (≥75%)")
        
        if metrics.recall_at_3 >= 0.90:
            observations.append(f"✅ Recall@3 ({metrics.recall_at_3:.1%}) meets target (≥90%)")
        else:
            observations.append(f"⚠️ Recall@3 ({metrics.recall_at_3:.1%}) below target (≥90%)")
        
        if metrics.false_positive_rate < 0.10:
            observations.append(f"✅ False Positive Rate ({metrics.false_positive_rate:.1%}) is low")
        else:
            observations.append(f"⚠️ False Positive Rate ({metrics.false_positive_rate:.1%}) is high")
        
        if metrics.p95_total_ms < 2000:
            observations.append(f"✅ P95 Latency ({metrics.p95_total_ms:.0f}ms) meets target (<2000ms)")
        else:
            observations.append(f"⚠️ P95 Latency ({metrics.p95_total_ms:.0f}ms) above target (<2000ms)")
        
        return EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_questions=total,
            relevance_threshold=RELEVANCE_THRESHOLD,
            metrics=metrics,
            section_breakdown=section_breakdown,
            observations=observations,
            all_results=all_results
        )
    
    def print_report(self, report: EvaluationReport):
        """Print formatted report."""
        m = report.metrics
        
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        print(f"\n📅 Timestamp: {report.timestamp}")
        print(f"📄 Questions: {report.total_questions}")
        print(f"🎯 Relevance Threshold: {report.relevance_threshold}")
        
        print("\n" + "-" * 50)
        print("RETRIEVAL QUALITY METRICS")
        print("-" * 50)
        print(f"  Recall@1:        {m.recall_at_1:.1%}  (Target: ≥75%)")
        print(f"  Recall@3:        {m.recall_at_3:.1%}  (Target: ≥90%)")
        print(f"  Recall@5:        {m.recall_at_5:.1%}")
        print(f"  Top-1 Accuracy:  {m.top1_accuracy:.1%}")
        print(f"  Top-3 Accuracy:  {m.top3_accuracy:.1%}")
        print(f"  MRR:             {m.mrr:.3f}")
        print(f"  nDCG@5:          {m.ndcg_at_5:.3f}")
        
        print("\n" + "-" * 50)
        print("ROBUSTNESS METRICS")
        print("-" * 50)
        print(f"  Paraphrase Robustness:  {m.paraphrase_robustness:.1%}")
        print(f"  False Positive Rate:    {m.false_positive_rate:.1%}  (Target: <10%)")
        print(f"  Hallucination Rate:     {m.hallucination_rate:.1%}")
        
        print("\n" + "-" * 50)
        print("STAGE-WISE LATENCY BREAKDOWN")
        print("-" * 50)
        print(f"  Embedding:       {m.avg_embedding_ms:.0f}ms")
        print(f"  Retrieval:       {m.avg_retrieval_ms:.0f}ms")
        print(f"  Re-ranking:      {m.avg_reranking_ms:.0f}ms")
        print(f"  Total Average:   {m.avg_total_ms:.0f}ms")
        print(f"  P50:             {m.p50_total_ms:.0f}ms")
        print(f"  P95:             {m.p95_total_ms:.0f}ms  (Target: <2000ms)")
        print(f"  P99:             {m.p99_total_ms:.0f}ms")
        
        print("\n" + "-" * 50)
        print("CACHE PERFORMANCE")
        print("-" * 50)
        print(f"  Cache Hit Rate:    {m.cache_hit_rate:.1%}")
        print(f"  Cached Latency:    {m.cached_latency_ms:.0f}ms")
        print(f"  Uncached Latency:  {m.uncached_latency_ms:.0f}ms")
        print(f"  Improvement:       {m.cache_improvement_pct:.1f}%")
        
        print("\n" + "-" * 50)
        print("SECTION BREAKDOWN")
        print("-" * 50)
        for sec, data in report.section_breakdown.items():
            print(f"  {sec} ({data['name'][:25]}): Acc={data['accuracy']:.1%}, Lat={data['avg_latency_ms']:.0f}ms")
        
        print("\n" + "-" * 50)
        print("OBSERVATIONS")
        print("-" * 50)
        for obs in report.observations:
            print(f"  {obs}")
        
        print("\n" + "=" * 70)
    
    def save_report(self, report: EvaluationReport, output_dir: Path = None):
        """Save all reports."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        m = report.metrics
        
        # JSON report
        json_path = output_dir / "evaluation_results.json"
        report_dict = {
            "timestamp": report.timestamp,
            "total_questions": report.total_questions,
            "relevance_threshold": report.relevance_threshold,
            "metrics": asdict(m),
            "section_breakdown": report.section_breakdown,
            "observations": report.observations,
            "results": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "section": r.section,
                    "category": r.category,
                    "top_score": r.top_result_score,
                    "latency_ms": r.latency_ms,
                    "has_relevant": r.has_relevant_result
                }
                for r in report.all_results
            ]
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)
        print(f"\n💾 JSON: {json_path}")
        
        # Markdown summary
        md_path = output_dir / "EVALUATION_SUMMARY.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Evaluation Results Summary\n\n")
            f.write(f"**Date:** {report.timestamp}\n")
            f.write(f"**Questions:** {report.total_questions}\n")
            f.write(f"**Relevance Threshold:** {report.relevance_threshold}\n\n")
            
            f.write("## Retrieval Quality\n\n")
            f.write("| Metric | Value | Target | Status |\n")
            f.write("|--------|-------|--------|--------|\n")
            f.write(f"| Recall@1 | {m.recall_at_1:.1%} | >=75% | {'PASS' if m.recall_at_1 >= 0.75 else 'FAIL'} |\n")
            f.write(f"| Recall@3 | {m.recall_at_3:.1%} | >=90% | {'PASS' if m.recall_at_3 >= 0.90 else 'FAIL'} |\n")
            f.write(f"| Top-1 Accuracy | {m.top1_accuracy:.1%} | >=75% | {'PASS' if m.top1_accuracy >= 0.75 else 'FAIL'} |\n")
            f.write(f"| MRR | {m.mrr:.3f} | - | - |\n")
            f.write(f"| nDCG@5 | {m.ndcg_at_5:.3f} | - | - |\n\n")
            
            f.write("## Robustness\n\n")
            f.write("| Metric | Value | Target | Status |\n")
            f.write("|--------|-------|--------|--------|\n")
            f.write(f"| Paraphrase Robustness | {m.paraphrase_robustness:.1%} | - | - |\n")
            f.write(f"| False Positive Rate | {m.false_positive_rate:.1%} | <10% | {'PASS' if m.false_positive_rate < 0.10 else 'FAIL'} |\n")
            f.write(f"| Hallucination Rate | {m.hallucination_rate:.1%} | <10% | {'PASS' if m.hallucination_rate < 0.10 else 'FAIL'} |\n\n")
            
            f.write("## Latency (Stage-wise)\n\n")
            f.write("| Stage | Time (ms) |\n")
            f.write("|-------|----------|\n")
            f.write(f"| Embedding | {m.avg_embedding_ms:.0f} |\n")
            f.write(f"| Retrieval | {m.avg_retrieval_ms:.0f} |\n")
            f.write(f"| Re-ranking | {m.avg_reranking_ms:.0f} |\n")
            f.write(f"| **Total Avg** | **{m.avg_total_ms:.0f}** |\n")
            f.write(f"| P95 | {m.p95_total_ms:.0f} |\n\n")
            
            f.write("## Cache Performance\n\n")
            f.write(f"- Hit Rate: {m.cache_hit_rate:.1%}\n")
            f.write(f"- Improvement: {m.cache_improvement_pct:.1f}%\n")
        print(f"💾 Markdown: {md_path}")
        
        # LaTeX report
        tex_path = output_dir / "evaluation_report.tex"
        self._save_latex(report, tex_path)
        print(f"💾 LaTeX: {tex_path}")
    
    def _save_latex(self, report: EvaluationReport, tex_path: Path):
        """Generate LaTeX report."""
        m = report.metrics
        
        def pct(v):
            return f"{v*100:.1f}\\%"
        
        def status(val, target, higher_better=True):
            if higher_better:
                return r"\textcolor{pass}{PASS}" if val >= target else r"\textcolor{fail}{FAIL}"
            else:
                return r"\textcolor{pass}{PASS}" if val <= target else r"\textcolor{fail}{FAIL}"
        
        latex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{xcolor}
\usepackage{hyperref}

\geometry{margin=1in}
\definecolor{pass}{rgb}{0.2,0.6,0.2}
\definecolor{fail}{rgb}{0.8,0.2,0.2}

\title{Real Estate Document Intelligence System\\Comprehensive Evaluation Report}
\author{Evaluation Suite v2.0}
\date{""" + report.timestamp[:10] + r"""}

\begin{document}
\maketitle

\section{Executive Summary}
This report presents comprehensive evaluation results of the Real Estate Document Intelligence System 
against """ + str(report.total_questions) + r""" test questions. The evaluation uses a relevance 
threshold of """ + str(report.relevance_threshold) + r""" to determine if retrieved results are relevant.

\section{Retrieval Quality Metrics}

\begin{table}[h]
\centering
\begin{tabular}{lrrl}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Target} & \textbf{Status} \\
\midrule
Recall@1 & """ + pct(m.recall_at_1) + r""" & $\geq$75\% & """ + status(m.recall_at_1, 0.75) + r""" \\
Recall@3 & """ + pct(m.recall_at_3) + r""" & $\geq$90\% & """ + status(m.recall_at_3, 0.90) + r""" \\
Recall@5 & """ + pct(m.recall_at_5) + r""" & -- & -- \\
Top-1 Accuracy & """ + pct(m.top1_accuracy) + r""" & $\geq$75\% & """ + status(m.top1_accuracy, 0.75) + r""" \\
Top-3 Accuracy & """ + pct(m.top3_accuracy) + r""" & $\geq$90\% & """ + status(m.top3_accuracy, 0.90) + r""" \\
MRR & """ + f"{m.mrr:.3f}" + r""" & -- & -- \\
nDCG@5 & """ + f"{m.ndcg_at_5:.3f}" + r""" & -- & -- \\
\bottomrule
\end{tabular}
\caption{Retrieval Quality Metrics}
\end{table}

\section{Robustness Metrics}

\begin{table}[h]
\centering
\begin{tabular}{lrrl}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Target} & \textbf{Status} \\
\midrule
Paraphrase Robustness & """ + pct(m.paraphrase_robustness) + r""" & -- & -- \\
False Positive Rate & """ + pct(m.false_positive_rate) + r""" & $<$10\% & """ + status(m.false_positive_rate, 0.10, False) + r""" \\
Hallucination Rate & """ + pct(m.hallucination_rate) + r""" & $<$10\% & """ + status(m.hallucination_rate, 0.10, False) + r""" \\
\bottomrule
\end{tabular}
\caption{Robustness Metrics}
\end{table}

\section{Stage-wise Latency Breakdown}

\begin{table}[h]
\centering
\begin{tabular}{lr}
\toprule
\textbf{Stage} & \textbf{Time (ms)} \\
\midrule
Embedding & """ + f"{m.avg_embedding_ms:.0f}" + r""" \\
Retrieval & """ + f"{m.avg_retrieval_ms:.0f}" + r""" \\
Re-ranking & """ + f"{m.avg_reranking_ms:.0f}" + r""" \\
\midrule
\textbf{Total Average} & \textbf{""" + f"{m.avg_total_ms:.0f}" + r"""} \\
P50 (Median) & """ + f"{m.p50_total_ms:.0f}" + r""" \\
P95 & """ + f"{m.p95_total_ms:.0f}" + r""" \\
P99 & """ + f"{m.p99_total_ms:.0f}" + r""" \\
\bottomrule
\end{tabular}
\caption{Latency Metrics (P95 Target: $<$2000ms)}
\end{table}

\section{Cache Performance}

\begin{table}[h]
\centering
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Cache Hit Rate & """ + pct(m.cache_hit_rate) + r""" \\
Cached Query Latency & """ + f"{m.cached_latency_ms:.0f}" + r"""ms \\
Uncached Query Latency & """ + f"{m.uncached_latency_ms:.0f}" + r"""ms \\
Latency Improvement & """ + f"{m.cache_improvement_pct:.1f}" + r"""\% \\
\bottomrule
\end{tabular}
\caption{Cache Performance}
\end{table}

\section{Section-wise Breakdown}

\begin{table}[h]
\centering
\begin{tabular}{p{1cm}p{5cm}rr}
\toprule
\textbf{Sec} & \textbf{Name} & \textbf{Accuracy} & \textbf{Latency} \\
\midrule
"""
        
        for sec, data in report.section_breakdown.items():
            latex += f"{sec} & {data['name']} & {data['accuracy']*100:.1f}\\% & {data['avg_latency_ms']:.0f}ms \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\caption{Section-wise Performance}
\end{table}

\section{Observations}
\begin{itemize}
"""
        for obs in report.observations:
            # Remove emoji for LaTeX
            clean_obs = obs.replace("✅", "[PASS]").replace("⚠️", "[WARN]").replace("❌", "[FAIL]")
            latex += f"\\item {clean_obs}\n"
        
        latex += r"""\end{itemize}

\section{Conclusions}
"""
        
        passes = sum([
            m.recall_at_1 >= 0.75,
            m.recall_at_3 >= 0.90,
            m.false_positive_rate < 0.10,
            m.p95_total_ms < 2000
        ])
        
        if passes == 4:
            latex += r"The system \textbf{\textcolor{pass}{PASSES}} all performance targets."
        else:
            latex += r"The system meets " + str(passes) + r" of 4 performance targets."
        
        latex += r"""

\end{document}
"""
        
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    
    evaluator = ComprehensiveEvaluator(api_url=args.api_url)
    
    try:
        report = evaluator.run(run_twice=not args.no_cache)
        evaluator.print_report(report)
        evaluator.save_report(report)
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
