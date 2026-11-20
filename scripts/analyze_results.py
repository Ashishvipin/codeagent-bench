#!/usr/bin/env python3
"""
analyze_results.py — Analyze benchmark results with statistical rigor.

Reads JSONL result files produced by run_benchmark.py and generates
detailed statistical analysis including bootstrap CIs, McNemar's tests
for paired comparisons, and difficulty calibration curves.

Usage:
    python scripts/analyze_results.py --results results/run_001.jsonl
    python scripts/analyze_results.py --compare results/run_model_a.jsonl results/run_model_b.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codeagent_bench.scorer import TaskResult
from codeagent_bench.statistics import StatisticalAnalyzer
from codeagent_bench.utils import load_jsonl, setup_logging

console = Console()


def _load_results(path: str) -> tuple[dict, list[TaskResult]]:
    """Load a results JSONL file into metadata + TaskResults."""
    records = load_jsonl(path)
    metadata = {}
    results = []

    for rec in records:
        if rec.get("record_type") == "run_metadata":
            metadata = rec
        elif rec.get("record_type") == "task_result":
            results.append(
                TaskResult(
                    task_id=rec["task_id"],
                    category=rec["category"],
                    difficulty=rec["difficulty"],
                    agent_output="",
                    score=rec["score"],
                    grade=rec["grade"],
                    execution_time_s=rec.get("execution_time_s", 0),
                )
            )

    return metadata, results


@click.command()
@click.option(
    "--results",
    required=False,
    type=click.Path(exists=True),
    help="Path to a single results JSONL file.",
)
@click.option(
    "--compare",
    nargs=2,
    type=click.Path(exists=True),
    default=None,
    help="Two result files for paired comparison (McNemar's test).",
)
@click.option("--report", default=None, type=click.Path(), help="Output path for analysis report.")
@click.option("--bootstrap-n", default=10000, type=int, help="Bootstrap iterations.")
@click.option("--confidence", default=0.95, type=float, help="Confidence level.")
@click.option("--log-level", default="INFO", help="Logging level.")
def main(
    results: str | None,
    compare: tuple[str, str] | None,
    report: str | None,
    bootstrap_n: int,
    confidence: float,
    log_level: str,
) -> None:
    """Analyze CodeAgent-Bench results with statistical rigor."""
    setup_logging(log_level)

    analyzer = StatisticalAnalyzer(
        n_bootstrap=bootstrap_n,
        confidence=confidence,
    )

    if compare:
        _run_comparison(analyzer, compare[0], compare[1], report)
    elif results:
        _run_single_analysis(analyzer, results, report)
    else:
        console.print("[red]Provide --results or --compare[/]")
        sys.exit(1)


def _run_single_analysis(
    analyzer: StatisticalAnalyzer,
    results_path: str,
    report_path: str | None,
) -> None:
    """Analyze a single benchmark run."""
    metadata, results = _load_results(results_path)

    console.print(f"\n[bold blue]Analysis: {metadata.get('run_id', 'unknown')}[/]\n")

    # Overall CI
    score_ci = analyzer.score_ci(results)
    pass_ci = analyzer.pass_rate_ci(results)

    table = Table(title="Statistical Summary (Bootstrap CIs)")
    table.add_column("Metric", style="bold")
    table.add_column("Estimate", justify="right")
    table.add_column(f"{analyzer.confidence*100:.0f}% CI", justify="right")

    table.add_row(
        "Overall Score",
        f"{score_ci.estimate:.4f}",
        f"[{score_ci.lower:.4f}, {score_ci.upper:.4f}]",
    )
    table.add_row(
        "Pass Rate",
        f"{pass_ci.estimate:.4f}",
        f"[{pass_ci.lower:.4f}, {pass_ci.upper:.4f}]",
    )
    console.print(table)

    # Category CIs
    cat_table = Table(title="Per-Category Analysis")
    cat_table.add_column("Category")
    cat_table.add_column("n", justify="right")
    cat_table.add_column("Score", justify="right")
    cat_table.add_column(f"{analyzer.confidence*100:.0f}% CI", justify="right")

    for cat, ci in sorted(analyzer.category_cis(results).items()):
        n = sum(1 for r in results if r.category == cat)
        cat_table.add_row(
            cat,
            str(n),
            f"{ci.estimate:.4f}",
            f"[{ci.lower:.4f}, {ci.upper:.4f}]",
        )
    console.print(cat_table)

    # Difficulty calibration
    cal = analyzer.difficulty_calibration(results)
    cal_table = Table(title="Difficulty Calibration")
    cal_table.add_column("Difficulty")
    cal_table.add_column("n", justify="right")
    cal_table.add_column("Mean Score", justify="right")
    cal_table.add_column("Pass Rate", justify="right")

    for diff in ["easy", "medium", "hard"]:
        info = cal.get(diff, {})
        cal_table.add_row(
            diff.capitalize(),
            str(int(info.get("count", 0))),
            f"{info.get('mean_score', 0):.4f}",
            f"{info.get('pass_rate', 0):.4f}",
        )
    console.print(cal_table)

    # Save report
    if report_path:
        summary = analyzer.full_summary(results)
        summary["run_metadata"] = metadata
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        console.print(f"\n[green]Analysis saved to {report_path}[/]")


def _run_comparison(
    analyzer: StatisticalAnalyzer,
    path_a: str,
    path_b: str,
    report_path: str | None,
) -> None:
    """Compare two benchmark runs using McNemar's test."""
    meta_a, results_a = _load_results(path_a)
    meta_b, results_b = _load_results(path_b)

    model_a = meta_a.get("model", "Model A")
    model_b = meta_b.get("model", "Model B")

    console.print(f"\n[bold blue]Paired Comparison: {model_a} vs {model_b}[/]\n")

    comparison = analyzer.mcnemar_test(
        results_a, results_b, model_a=model_a, model_b=model_b
    )

    table = Table(title="McNemar's Test")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row(f"{model_a} Pass Rate", f"{comparison.model_a_score:.4f}")
    table.add_row(f"{model_b} Pass Rate", f"{comparison.model_b_score:.4f}")
    table.add_row("χ² Statistic", f"{comparison.statistic:.4f}")
    table.add_row("p-value", f"{comparison.p_value:.6f}")
    table.add_row(
        "Significant?",
        "[green]Yes[/]" if comparison.significant else "[red]No[/]",
    )
    table.add_row("Effect Size (Cohen's g)", f"{comparison.effect_size:.4f}")
    table.add_row("Common Tasks", str(comparison.n_tasks))

    console.print(table)

    if comparison.significant:
        better = model_a if comparison.model_a_score > comparison.model_b_score else model_b
        console.print(
            f"\n[bold green]✅ {better} is significantly better "
            f"(p={comparison.p_value:.4f})[/]"
        )
    else:
        console.print(
            f"\n[yellow]No significant difference "
            f"(p={comparison.p_value:.4f})[/]"
        )

    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(
            json.dumps(comparison.to_dict(), indent=2), encoding="utf-8"
        )
        console.print(f"\n[green]Comparison saved to {report_path}[/]")


if __name__ == "__main__":
    main()
