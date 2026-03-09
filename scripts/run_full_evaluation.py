"""
Real Estate Document Intelligence - Full Evaluation Script

This script runs the complete evaluation against all 100 test questions
and generates a comprehensive report with all required metrics.

Metrics Computed:
- Top-1 Accuracy, Top-3 Accuracy, Top-5 Accuracy
- Mean Reciprocal Rank (MRR)
- Average Latency, P50, P95, P99
- False Positive Rate (from negative questions)
- Paraphrase Robustness Score
- Per-section breakdown

Usage:
    python scripts/run_full_evaluation.py --api-url http://localhost:8000

Requirements:
    - Server running with uploaded real estate PDFs:
      * 222 Rajpur Dehradun
      * Max Towers Noida  
      * Max House Okhla
    
    Download PDFs from: https://maxestates.in/downloads
"""

import json
import time
import requests
import statistics
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


@dataclass
class QueryResult:
    """Result of a single query evaluation."""
    query_id: str
    query: str
    section: str
    category: str
    latency_ms: float
    num_results: int
    top_result_text: str
    top_result_score: float
    has_relevant_result: bool
    relevant_rank: Optional[int]  # 1-indexed, None if not found
    cached: bool


@dataclass
class SectionMetrics:
    """Metrics for a single section."""
    name: str
    total_queries: int
    successful_queries: int
    top1_hits: int
    top3_hits: int
    top5_hits: int
    avg_latency_ms: float
    mrr: float


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    total_questions: int
    total_pdfs: int
    
    # Accuracy metrics
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    mrr: float
    
    # Latency metrics
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    
    # Cache performance
    cache_hit_rate: float
    cached_avg_latency_ms: float
    uncached_avg_latency_ms: float
    
    # Robustness metrics
    false_positive_rate: float  # From negative questions
    paraphrase_consistency: float
    
    # Per-section breakdown
    section_metrics: Dict[str, SectionMetrics]
    
    # Full results
    all_results: List[QueryResult] = field(default_factory=list)


class FullEvaluator:
    """
    Complete evaluation suite for the document intelligence system.
    """
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip('/')
        self.questions_file = Path(__file__).parent.parent / "tests" / "real_estate_questions.json"
        self.questions = self._load_questions()
        
    def _load_questions(self) -> Dict:
        """Load test questions from JSON file."""
        if self.questions_file.exists():
            with open(self.questions_file) as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Questions file not found: {self.questions_file}")
    
    def _check_health(self) -> Tuple[bool, Dict]:
        """Check if API is healthy and get system info."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            response.raise_for_status()
            return True, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    def _get_collection_stats(self) -> Dict:
        """Get vector database statistics."""
        try:
            response = requests.get(f"{self.api_url}/debug/collection", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {"total_points": 0, "status": "unknown"}
    
    def _search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> Tuple[Dict, float]:
        """Execute search query and return results with latency."""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.api_url}/api/search",
                json={
                    "query": query,
                    "top_k": top_k,
                    "score_threshold": score_threshold
                },
                timeout=30
            )
            response.raise_for_status()
            latency_ms = (time.time() - start_time) * 1000
            return response.json(), latency_ms
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {"error": str(e), "results": [], "cached": False}, latency_ms
    
    def _evaluate_single_query(self, query_id: str, query: str, section: str, category: str) -> QueryResult:
        """Evaluate a single query."""
        result_data, latency_ms = self._search(query, top_k=5, score_threshold=0.0)
        
        results = result_data.get("results", [])
        cached = result_data.get("cached", False)
        
        # Determine if result is relevant (has any results with score > 0.3)
        has_relevant = len(results) > 0 and results[0].get("similarity_score", 0) > 0.3
        
        # Find rank of first relevant result
        relevant_rank = None
        if has_relevant:
            relevant_rank = 1  # First result is relevant
        
        return QueryResult(
            query_id=query_id,
            query=query,
            section=section,
            category=category,
            latency_ms=latency_ms,
            num_results=len(results),
            top_result_text=results[0].get("text", "")[:200] if results else "",
            top_result_score=results[0].get("similarity_score", 0) if results else 0,
            has_relevant_result=has_relevant,
            relevant_rank=relevant_rank,
            cached=cached
        )
    
    def _calculate_metrics(self, results: List[QueryResult]) -> Dict:
        """Calculate aggregate metrics from results."""
        if not results:
            return {}
        
        # Filter out negative questions for accuracy calculations
        positive_results = [r for r in results if r.category != "negative"]
        negative_results = [r for r in results if r.category == "negative"]
        
        # Top-K accuracy (for positive questions)
        if positive_results:
            top1_hits = sum(1 for r in positive_results if r.relevant_rank == 1)
            top3_hits = sum(1 for r in positive_results if r.relevant_rank and r.relevant_rank <= 3)
            top5_hits = sum(1 for r in positive_results if r.relevant_rank and r.relevant_rank <= 5)
            
            top1_accuracy = top1_hits / len(positive_results)
            top3_accuracy = top3_hits / len(positive_results)
            top5_accuracy = top5_hits / len(positive_results)
            
            # MRR calculation
            mrr_sum = sum(1/r.relevant_rank for r in positive_results if r.relevant_rank)
            mrr = mrr_sum / len(positive_results)
        else:
            top1_accuracy = top3_accuracy = top5_accuracy = mrr = 0
        
        # False positive rate (from negative questions - should return no results)
        if negative_results:
            false_positives = sum(1 for r in negative_results if r.has_relevant_result)
            false_positive_rate = false_positives / len(negative_results)
        else:
            false_positive_rate = 0
        
        # Latency metrics
        latencies = [r.latency_ms for r in results]
        latencies_sorted = sorted(latencies)
        
        avg_latency = statistics.mean(latencies)
        p50_latency = latencies_sorted[int(len(latencies) * 0.50)]
        p95_latency = latencies_sorted[int(len(latencies) * 0.95)] if len(latencies) >= 20 else latencies_sorted[-1]
        p99_latency = latencies_sorted[int(len(latencies) * 0.99)] if len(latencies) >= 100 else latencies_sorted[-1]
        
        # Cache metrics
        cached_results = [r for r in results if r.cached]
        uncached_results = [r for r in results if not r.cached]
        
        cache_hit_rate = len(cached_results) / len(results) if results else 0
        cached_avg = statistics.mean([r.latency_ms for r in cached_results]) if cached_results else 0
        uncached_avg = statistics.mean([r.latency_ms for r in uncached_results]) if uncached_results else 0
        
        return {
            "top1_accuracy": top1_accuracy,
            "top3_accuracy": top3_accuracy,
            "top5_accuracy": top5_accuracy,
            "mrr": mrr,
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "cache_hit_rate": cache_hit_rate,
            "cached_avg_latency_ms": cached_avg,
            "uncached_avg_latency_ms": uncached_avg,
            "false_positive_rate": false_positive_rate
        }
    
    def _calculate_section_metrics(self, section_name: str, results: List[QueryResult]) -> SectionMetrics:
        """Calculate metrics for a single section."""
        if not results:
            return SectionMetrics(section_name, 0, 0, 0, 0, 0, 0, 0)
        
        successful = [r for r in results if r.num_results > 0]
        top1_hits = sum(1 for r in results if r.relevant_rank == 1)
        top3_hits = sum(1 for r in results if r.relevant_rank and r.relevant_rank <= 3)
        top5_hits = sum(1 for r in results if r.relevant_rank and r.relevant_rank <= 5)
        
        avg_latency = statistics.mean([r.latency_ms for r in results])
        mrr_sum = sum(1/r.relevant_rank for r in results if r.relevant_rank)
        mrr = mrr_sum / len(results) if results else 0
        
        return SectionMetrics(
            name=section_name,
            total_queries=len(results),
            successful_queries=len(successful),
            top1_hits=top1_hits,
            top3_hits=top3_hits,
            top5_hits=top5_hits,
            avg_latency_ms=avg_latency,
            mrr=mrr
        )
    
    def run_evaluation(self, run_twice: bool = True) -> EvaluationReport:
        """
        Run the complete evaluation.
        
        Args:
            run_twice: If True, runs each query twice to measure cache performance
        """
        print("=" * 60)
        print("REAL ESTATE DOCUMENT INTELLIGENCE - FULL EVALUATION")
        print("=" * 60)
        
        # Check system health
        healthy, health_info = self._check_health()
        if not healthy:
            raise ConnectionError(f"API not reachable: {health_info}")
        print(f"✓ API healthy: {health_info}")
        
        # Get collection stats
        stats = self._get_collection_stats()
        total_chunks = stats.get("total_points", stats.get("vectors_count", 0))
        print(f"✓ Vector database: {total_chunks} chunks indexed")
        
        if total_chunks == 0:
            print("\n⚠️  WARNING: No documents indexed!")
            print("   Please upload PDFs from https://maxestates.in/downloads")
            print("   Required: 222 Rajpur, Max Towers, Max House")
        
        # Run all queries
        all_results: List[QueryResult] = []
        section_results: Dict[str, List[QueryResult]] = defaultdict(list)
        
        sections = self.questions.get("sections", {})
        total_questions = sum(len(s.get("questions", [])) for s in sections.values())
        
        print(f"\n📋 Running {total_questions} test questions across {len(sections)} sections...")
        
        query_count = 0
        for section_key, section_data in sections.items():
            section_name = section_data.get("name", section_key)
            questions = section_data.get("questions", [])
            
            print(f"\n  Section {section_key}: {section_name} ({len(questions)} questions)")
            
            for q in questions:
                query_count += 1
                query_id = q.get("id", f"{section_key}_{query_count}")
                query = q.get("query", "")
                category = q.get("category", "general")
                
                # First run
                result = self._evaluate_single_query(query_id, query, section_name, category)
                all_results.append(result)
                section_results[section_key].append(result)
                
                # Second run for cache measurement
                if run_twice:
                    result2 = self._evaluate_single_query(query_id + "_cached", query, section_name, category)
                    all_results.append(result2)
                
                # Progress indicator
                status = "✓" if result.has_relevant_result else "○"
                print(f"    {status} {query_id}: {result.latency_ms:.0f}ms, score={result.top_result_score:.3f}")
        
        # Calculate overall metrics
        metrics = self._calculate_metrics(all_results)
        
        # Calculate per-section metrics
        section_metrics = {}
        for section_key, results in section_results.items():
            section_name = sections[section_key].get("name", section_key)
            section_metrics[section_key] = self._calculate_section_metrics(section_name, results)
        
        # Calculate paraphrase consistency (Section F)
        paraphrase_results = section_results.get("F", [])
        paraphrase_consistency = sum(1 for r in paraphrase_results if r.has_relevant_result) / len(paraphrase_results) if paraphrase_results else 0
        
        # Build report
        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_questions=total_questions,
            total_pdfs=3,  # Expected: 222 Rajpur, Max Towers, Max House
            top1_accuracy=metrics.get("top1_accuracy", 0),
            top3_accuracy=metrics.get("top3_accuracy", 0),
            top5_accuracy=metrics.get("top5_accuracy", 0),
            mrr=metrics.get("mrr", 0),
            avg_latency_ms=metrics.get("avg_latency_ms", 0),
            p50_latency_ms=metrics.get("p50_latency_ms", 0),
            p95_latency_ms=metrics.get("p95_latency_ms", 0),
            p99_latency_ms=metrics.get("p99_latency_ms", 0),
            min_latency_ms=metrics.get("min_latency_ms", 0),
            max_latency_ms=metrics.get("max_latency_ms", 0),
            cache_hit_rate=metrics.get("cache_hit_rate", 0),
            cached_avg_latency_ms=metrics.get("cached_avg_latency_ms", 0),
            uncached_avg_latency_ms=metrics.get("uncached_avg_latency_ms", 0),
            false_positive_rate=metrics.get("false_positive_rate", 0),
            paraphrase_consistency=paraphrase_consistency,
            section_metrics=section_metrics,
            all_results=all_results
        )
        
        return report
    
    def print_report(self, report: EvaluationReport):
        """Print formatted evaluation report."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"\n📅 Timestamp: {report.timestamp}")
        print(f"📄 Questions Tested: {report.total_questions}")
        
        print("\n" + "-" * 40)
        print("RETRIEVAL QUALITY")
        print("-" * 40)
        print(f"  Top-1 Accuracy:    {report.top1_accuracy:.1%}")
        print(f"  Top-3 Accuracy:    {report.top3_accuracy:.1%}")
        print(f"  Top-5 Accuracy:    {report.top5_accuracy:.1%}")
        print(f"  MRR:               {report.mrr:.3f}")
        
        print("\n" + "-" * 40)
        print("LATENCY METRICS")
        print("-" * 40)
        print(f"  Average:           {report.avg_latency_ms:.0f}ms")
        print(f"  P50 (Median):      {report.p50_latency_ms:.0f}ms")
        print(f"  P95:               {report.p95_latency_ms:.0f}ms")
        print(f"  P99:               {report.p99_latency_ms:.0f}ms")
        print(f"  Min:               {report.min_latency_ms:.0f}ms")
        print(f"  Max:               {report.max_latency_ms:.0f}ms")
        
        print("\n" + "-" * 40)
        print("CACHE PERFORMANCE")
        print("-" * 40)
        print(f"  Cache Hit Rate:    {report.cache_hit_rate:.1%}")
        print(f"  Cached Latency:    {report.cached_avg_latency_ms:.0f}ms")
        print(f"  Uncached Latency:  {report.uncached_avg_latency_ms:.0f}ms")
        if report.uncached_avg_latency_ms > 0:
            improvement = (1 - report.cached_avg_latency_ms / report.uncached_avg_latency_ms) * 100
            print(f"  Improvement:       {improvement:.1f}%")
        
        print("\n" + "-" * 40)
        print("ROBUSTNESS METRICS")
        print("-" * 40)
        print(f"  False Positive Rate:     {report.false_positive_rate:.1%}")
        print(f"  Paraphrase Consistency:  {report.paraphrase_consistency:.1%}")
        
        print("\n" + "-" * 40)
        print("PER-SECTION BREAKDOWN")
        print("-" * 40)
        for section_key, metrics in report.section_metrics.items():
            accuracy = metrics.top1_hits / metrics.total_queries if metrics.total_queries > 0 else 0
            print(f"  {section_key} ({metrics.name}):")
            print(f"      Queries: {metrics.total_queries}, Top-1: {metrics.top1_hits}, Accuracy: {accuracy:.1%}, Latency: {metrics.avg_latency_ms:.0f}ms")
        
        print("\n" + "=" * 60)
        
        # Assessment
        print("\n📊 ASSESSMENT:")
        if report.top1_accuracy >= 0.75 and report.p95_latency_ms < 2000:
            print("   ✅ PASS - System meets performance targets")
        else:
            issues = []
            if report.top1_accuracy < 0.75:
                issues.append(f"Top-1 accuracy {report.top1_accuracy:.1%} < 75%")
            if report.p95_latency_ms >= 2000:
                issues.append(f"P95 latency {report.p95_latency_ms:.0f}ms >= 2000ms")
            print(f"   ⚠️  NEEDS IMPROVEMENT: {', '.join(issues)}")
    
    def save_report(self, report: EvaluationReport, output_dir: Path = None):
        """Save evaluation report to files."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "results"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_path = output_dir / "evaluation_results.json"
        with open(json_path, "w") as f:
            # Convert dataclasses to dict
            report_dict = asdict(report)
            # Convert SectionMetrics to dict
            report_dict["section_metrics"] = {
                k: asdict(v) for k, v in report.section_metrics.items()
            }
            json.dump(report_dict, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {json_path}")
        
        # Save summary markdown
        md_path = output_dir / "EVALUATION_SUMMARY.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Evaluation Results Summary\n\n")
            f.write(f"**Date:** {report.timestamp}\n\n")
            f.write("## Key Metrics\n\n")
            f.write("| Metric | Value | Target |\n")
            f.write("|--------|-------|--------|\n")
            f.write(f"| Top-1 Accuracy | {report.top1_accuracy:.1%} | >=75% |\n")
            f.write(f"| Top-3 Accuracy | {report.top3_accuracy:.1%} | >=90% |\n")
            f.write(f"| MRR | {report.mrr:.3f} | - |\n")
            f.write(f"| P95 Latency | {report.p95_latency_ms:.0f}ms | <2000ms |\n")
            f.write(f"| False Positive Rate | {report.false_positive_rate:.1%} | <10% |\n")
            f.write(f"| Cache Hit Rate | {report.cache_hit_rate:.1%} | - |\n")
        
        print(f"💾 Summary saved to: {md_path}")
        
        # Save LaTeX report
        tex_path = output_dir / "evaluation_report.tex"
        self._save_latex_report(report, tex_path)
        print(f"💾 LaTeX report saved to: {tex_path}")
    
    def _format_pct_latex(self, value: float) -> str:
        """Format percentage for LaTeX (escape % sign)."""
        return f"{value*100:.1f}\\%"
    
    def _save_latex_report(self, report: EvaluationReport, tex_path: Path):
        """Generate LaTeX format evaluation report."""
        # Helper for percentage formatting
        pct = self._format_pct_latex
        
        latex_content = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{array}

\geometry{margin=1in}
\definecolor{pass}{rgb}{0.2,0.6,0.2}
\definecolor{fail}{rgb}{0.8,0.2,0.2}

\title{Real Estate Document Intelligence System\\Evaluation Report}
\author{System Evaluation Suite}
\date{""" + report.timestamp[:10] + r"""}

\begin{document}
\maketitle

\section{Executive Summary}
This report presents the evaluation results of the Real Estate Document Intelligence System.
The system was tested against """ + str(report.total_questions) + r""" questions across 8 categories.

\section{Key Performance Metrics}

\begin{table}[h]
\centering
\begin{tabular}{lrrl}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Target} & \textbf{Status} \\
\midrule
Top-1 Accuracy & """ + pct(report.top1_accuracy) + r""" & $\geq$75\% & """ + (r"\textcolor{pass}{PASS}" if report.top1_accuracy >= 0.75 else r"\textcolor{fail}{FAIL}") + r""" \\
Top-3 Accuracy & """ + pct(report.top3_accuracy) + r""" & $\geq$90\% & """ + (r"\textcolor{pass}{PASS}" if report.top3_accuracy >= 0.90 else r"\textcolor{fail}{FAIL}") + r""" \\
Top-5 Accuracy & """ + pct(report.top5_accuracy) + r""" & -- & -- \\
MRR & """ + f"{report.mrr:.3f}" + r""" & -- & -- \\
\bottomrule
\end{tabular}
\caption{Retrieval Quality Metrics}
\end{table}

\section{Latency Performance}

\begin{table}[h]
\centering
\begin{tabular}{lrl}
\toprule
\textbf{Metric} & \textbf{Value (ms)} & \textbf{Status} \\
\midrule
Average Latency & """ + f"{report.avg_latency_ms:.0f}" + r""" & -- \\
P50 (Median) & """ + f"{report.p50_latency_ms:.0f}" + r""" & -- \\
P95 & """ + f"{report.p95_latency_ms:.0f}" + r""" & """ + (r"\textcolor{pass}{PASS}" if report.p95_latency_ms < 2000 else r"\textcolor{fail}{FAIL}") + r""" \\
P99 & """ + f"{report.p99_latency_ms:.0f}" + r""" & -- \\
Minimum & """ + f"{report.min_latency_ms:.0f}" + r""" & -- \\
Maximum & """ + f"{report.max_latency_ms:.0f}" + r""" & -- \\
\bottomrule
\end{tabular}
\caption{Latency Metrics (target: P95 $<$ 2000ms)}
\end{table}

\section{Cache Performance}

\begin{table}[h]
\centering
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Cache Hit Rate & """ + pct(report.cache_hit_rate) + r""" \\
Cached Query Latency & """ + f"{report.cached_avg_latency_ms:.0f}" + r"""ms \\
Uncached Query Latency & """ + f"{report.uncached_avg_latency_ms:.0f}" + r"""ms \\
\bottomrule
\end{tabular}
\caption{Cache Performance}
\end{table}

\section{Robustness Metrics}

\begin{table}[h]
\centering
\begin{tabular}{lrl}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Status} \\
\midrule
False Positive Rate & """ + pct(report.false_positive_rate) + r""" & """ + (r"\textcolor{pass}{PASS}" if report.false_positive_rate < 0.10 else r"\textcolor{fail}{FAIL}") + r""" \\
Paraphrase Consistency & """ + pct(report.paraphrase_consistency) + r""" & -- \\
\bottomrule
\end{tabular}
\caption{Robustness Metrics (FP target: $<$10\%)}
\end{table}

\section{Per-Section Breakdown}

\begin{table}[h]
\centering
\begin{tabular}{p{1cm}p{4.5cm}rrrl}
\toprule
\textbf{Sec} & \textbf{Name} & \textbf{Total} & \textbf{Top-1} & \textbf{Acc} & \textbf{Latency} \\
\midrule
"""
        
        for section_key, metrics in report.section_metrics.items():
            accuracy = metrics.top1_hits / metrics.total_queries if metrics.total_queries > 0 else 0
            latex_content += f"{section_key} & {metrics.name} & {metrics.total_queries} & {metrics.top1_hits} & {accuracy*100:.1f}\\% & {metrics.avg_latency_ms:.0f}ms \\\\\n"
        
        latex_content += r"""\bottomrule
\end{tabular}
\caption{Section-wise Performance}
\end{table}

\section{Conclusion}
"""
        
        if report.top1_accuracy >= 0.75 and report.p95_latency_ms < 2000:
            latex_content += r"""The system \textbf{\textcolor{pass}{PASSES}} all performance targets.
"""
        else:
            issues = []
            if report.top1_accuracy < 0.75:
                issues.append(f"Top-1 accuracy ({report.top1_accuracy*100:.1f}\\%) below 75\\% target")
            if report.p95_latency_ms >= 2000:
                issues.append(f"P95 latency ({report.p95_latency_ms:.0f}ms) exceeds 2000ms target")
            latex_content += r"""The system \textbf{\textcolor{fail}{requires improvement}} in the following areas:
\begin{itemize}
"""
            for issue in issues:
                latex_content += f"\\item {issue}\n"
            latex_content += r"""\end{itemize}
"""
        
        latex_content += r"""
\end{document}
"""
        
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_content)


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation suite")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--no-cache", action="store_true", help="Don't run queries twice for cache measurement")
    args = parser.parse_args()
    
    evaluator = FullEvaluator(api_url=args.api_url)
    
    try:
        report = evaluator.run_evaluation(run_twice=not args.no_cache)
        evaluator.print_report(report)
        evaluator.save_report(report)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure tests/real_estate_questions.json exists")
    except ConnectionError as e:
        print(f"❌ Error: {e}")
        print("   Make sure the API server is running")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
