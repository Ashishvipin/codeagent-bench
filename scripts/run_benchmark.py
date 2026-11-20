#!/usr/bin/env python3
"""
run_benchmark.py — CLI entry point for running CodeAgent-Bench evaluations.

Usage:
    python scripts/run_benchmark.py --tasks benchmarks/tasks/ --agent mock --output results/run.jsonl
    python scripts/run_benchmark.py --tasks benchmarks/tasks/error_handling.jsonl --agent completions --model my-model
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codeagent_bench.agent_wrapper import create_agent
from codeagent_bench.contamination import ContaminationDetector
from codeagent_bench.reporter import ReportGenerator
from codeagent_bench.runner import BenchmarkRunner
from codeagent_bench.scorer import Scorer
from codeagent_bench.statistics import StatisticalAnalyzer
from codeagent_bench.task_loader import TaskLoader
from codeagent_bench.utils import setup_logging

console = Console()


@click.command()
@click.option(
    "--tasks",
    required=True,
    type=click.Path(exists=True),
    help="Path to JSONL task file or directory of task files.",
)
@click.option(
    "--agent",
    default="mock",
    type=click.Choice(["mock", "completions", "http"]),
    help="Agent type to evaluate.",
)
@click.option("--model", default=None, help="Model name for the target agent.")
@click.option("--output", default=None, type=click.Path(), help="Output JSONL path for results.")
@click.option(
    "--report",
    default=None,
    type=click.Path(),
    help="Output path for Markdown report.",
)
@click.option(
    "--category",
    multiple=True,
    help="Filter by failure category (can specify multiple).",
)
@click.option(
    "--difficulty",
    multiple=True,
    type=click.Choice(["easy", "medium", "hard"]),
    help="Filter by difficulty level.",
)
@click.option("--run-id", default=None, help="Custom run identifier.")
@click.option(
    "--contamination-check / --no-contamination-check",
    default=True,
    help="Run contamination checks on tasks.",
)
@click.option("--log-level", default="INFO", help="Logging level.")
def main(
    tasks: str,
    agent: str,
    model: str | None,
    output: str | None,
    report: str | None,
    category: tuple[str, ...],
    difficulty: tuple[str, ...],
    run_id: str | None,
    contamination_check: bool,
    log_level: str,
) -> None:
    """Run CodeAgent-Bench evaluation against a coding agent."""
    setup_logging(log_level)

    console.print("\n[bold blue]╔══════════════════════════════════════╗[/]")
    console.print("[bold blue]║       CodeAgent-Bench  v0.1.0        ║[/]")
    console.print("[bold blue]╚══════════════════════════════════════╝[/]\n")

    # Build agent kwargs
    agent_kwargs = {}
    if model:
        agent_kwargs["model"] = model

    # Create runner
    runner = BenchmarkRunner(
        scorer=Scorer(strategy="weighted_test_pass"),
        task_loader=TaskLoader(strict=False),
    )

    # Run benchmark
    console.print(f"[bold]Agent:[/] {agent} ({model or 'default'})")
    console.print(f"[bold]Tasks:[/] {tasks}")
    console.print()

    bench_run = runner.run(
        tasks_path=tasks,
        output_path=output,
        categories=list(category) if category else None,
        difficulties=list(difficulty) if difficulty else None,
        agent_type=agent,
        agent_kwargs=agent_kwargs,
        run_id=run_id,
    )

    if not bench_run.results:
        console.print("[yellow]No results produced. Check task files and filters.[/]")
        sys.exit(1)

    # Print results table
    _print_results_table(bench_run)

    # Contamination check
    contamination_report = None
    if contamination_check:
        console.print("\n[bold]Running contamination checks...[/]")
        loader = TaskLoader(strict=False)
        tasks_path_obj = Path(tasks)
        if tasks_path_obj.is_dir():
            task_list = loader.load_directory(tasks_path_obj)
        else:
            task_list = loader.load_file(tasks_path_obj)

        detector = ContaminationDetector()
        contamination_report = detector.run_full_check(task_list)

        if contamination_report.summary.get("clean"):
            console.print("[green]✅ No contamination detected[/]")
        else:
            console.print("[yellow]⚠️  Potential contamination found — see report[/]")

    # Generate report
    if report:
        reporter = ReportGenerator()
        reporter.generate_markdown(bench_run, contamination_report, output_path=report)
        console.print(f"\n[green]Report saved to {report}[/]")

    # Also save JSON report alongside markdown
    if report:
        json_path = Path(report).with_suffix(".json")
        reporter = ReportGenerator()
        reporter.generate_json(bench_run, contamination_report, output_path=json_path)

    console.print("\n[bold green]Benchmark run complete.[/]\n")


def _print_results_table(bench_run) -> None:
    """Print a summary results table using Rich."""
    table = Table(title=f"Results: {bench_run.run_id}", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Agent", bench_run.model)
    table.add_row("Tasks", str(bench_run.num_tasks))
    table.add_row("Overall Score", f"{bench_run.overall_score:.4f}")
    table.add_row("Pass Rate", f"{bench_run.pass_rate:.4f}")
    table.add_row("Duration", f"{bench_run.total_time_s:.1f}s")

    # Grade counts
    grades = {"pass": 0, "partial": 0, "fail": 0}
    for r in bench_run.results:
        grades[r.grade] = grades.get(r.grade, 0) + 1
    table.add_row("✅ Pass", str(grades["pass"]))
    table.add_row("🟡 Partial", str(grades["partial"]))
    table.add_row("❌ Fail", str(grades["fail"]))

    console.print(table)

    # Category breakdown
    from codeagent_bench.scorer import Scorer

    cat_table = Table(title="By Failure Category")
    cat_table.add_column("Category")
    cat_table.add_column("Tasks", justify="right")
    cat_table.add_column("Mean Score", justify="right")
    cat_table.add_column("Pass Rate", justify="right")

    for cat, info in sorted(Scorer.aggregate_by_category(bench_run.results).items()):
        cat_table.add_row(
            cat,
            str(int(info["count"])),
            f"{info['mean_score']:.4f}",
            f"{info['pass_rate']:.4f}",
        )
    console.print(cat_table)


if __name__ == "__main__":
    main()
