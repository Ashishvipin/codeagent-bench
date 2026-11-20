#!/usr/bin/env python3
"""
validate_tasks.py — Validate benchmark task files for schema compliance,
semantic correctness, weight consistency, and contamination risks.

Usage:
    python scripts/validate_tasks.py --tasks benchmarks/tasks/
    python scripts/validate_tasks.py --tasks benchmarks/tasks/error_handling.jsonl --strict
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codeagent_bench.contamination import ContaminationDetector
from codeagent_bench.task_loader import TaskLoader
from codeagent_bench.utils import setup_logging

console = Console()


@click.command()
@click.option(
    "--tasks",
    required=True,
    type=click.Path(exists=True),
    help="Path to JSONL task file or directory.",
)
@click.option(
    "--strict / --no-strict",
    default=False,
    help="Fail on first validation error.",
)
@click.option(
    "--check-duplicates / --no-check-duplicates",
    default=True,
    help="Check for internal task duplicates.",
)
@click.option("--log-level", default="INFO", help="Logging level.")
def main(
    tasks: str,
    strict: bool,
    check_duplicates: bool,
    log_level: str,
) -> None:
    """Validate CodeAgent-Bench task files."""
    setup_logging(log_level)

    console.print("\n[bold blue]Task Validator[/]\n")

    loader = TaskLoader(strict=strict)
    tasks_path = Path(tasks)

    # Load tasks (validation happens during loading)
    try:
        if tasks_path.is_dir():
            task_list = loader.load_directory(tasks_path)
        else:
            task_list = loader.load_file(tasks_path)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Error: {exc}[/]")
        sys.exit(1)

    if not task_list:
        console.print("[yellow]No valid tasks found.[/]")
        sys.exit(1)

    # Print summary
    summary = loader.summary(task_list)

    table = Table(title="Task Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Valid Tasks", str(summary["total_tasks"]))

    for cat, count in sorted(summary["by_category"].items()):
        table.add_row(f"  {cat}", str(count))

    for diff, count in sorted(summary["by_difficulty"].items()):
        table.add_row(f"  [{diff}]", str(count))

    console.print(table)

    # Check for duplicate task IDs
    ids = [t.task_id for t in task_list]
    if len(ids) != len(set(ids)):
        from collections import Counter

        dupes = [id_ for id_, count in Counter(ids).items() if count > 1]
        console.print(f"[red]❌ Duplicate task IDs: {dupes}[/]")
        sys.exit(1)

    # Weight validation per task
    weight_issues = []
    for task in task_list:
        total_w = task.total_weight
        if not (0.99 <= total_w <= 1.01):
            weight_issues.append(
                f"  {task.task_id}: weights sum to {total_w:.3f}"
            )

    if weight_issues:
        console.print("[yellow]⚠️  Weight issues:[/]")
        for issue in weight_issues:
            console.print(f"  {issue}")
    else:
        console.print("[green]✅ All task weights sum to ~1.0[/]")

    # Contamination check
    if check_duplicates:
        detector = ContaminationDetector(ngram_size=8, jaccard_threshold=0.15)
        pairs = detector.check_internal_duplicates(task_list)
        if pairs:
            console.print(f"[yellow]⚠️  {len(pairs)} similar task pairs found:[/]")
            for pair in pairs:
                console.print(
                    f"  {pair['task_a']} ↔ {pair['task_b']} "
                    f"(Jaccard={pair['jaccard_similarity']:.3f})"
                )
        else:
            console.print("[green]✅ No internal duplicates detected[/]")

    console.print("\n[bold green]Validation complete.[/]\n")


if __name__ == "__main__":
    main()
