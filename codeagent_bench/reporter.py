"""
Reporter: Generate structured benchmark reports in Markdown and JSON.

Produces human-readable reports with per-category breakdowns, difficulty
curves, statistical summaries, and contamination audit results.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codeagent_bench.contamination import ContaminationReport
from codeagent_bench.runner import BenchmarkRun
from codeagent_bench.scorer import Scorer, TaskResult
from codeagent_bench.statistics import StatisticalAnalyzer

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate benchmark evaluation reports.

    Parameters
    ----------
    analyzer : StatisticalAnalyzer | None
        Pre-configured statistical analyzer.
    """

    def __init__(self, analyzer: StatisticalAnalyzer | None = None) -> None:
        self.analyzer = analyzer or StatisticalAnalyzer()

    def generate_markdown(
        self,
        bench_run: BenchmarkRun,
        contamination: ContaminationReport | None = None,
        output_path: str | Path | None = None,
    ) -> str:
        """Generate a full Markdown report."""
        lines: list[str] = []

        # Header
        lines.append(f"# CodeAgent-Bench Report: {bench_run.run_id}")
        lines.append("")
        lines.append(f"**Agent:** {bench_run.model}  ")
        lines.append(f"**Tasks:** {bench_run.num_tasks}  ")
        lines.append(f"**Date:** {bench_run.start_time}  ")
        lines.append(f"**Duration:** {bench_run.total_time_s:.1f}s  ")
        lines.append("")

        # Overall results with CI
        lines.append("## Overall Results")
        lines.append("")

        score_ci = self.analyzer.score_ci(bench_run.results)
        pass_ci = self.analyzer.pass_rate_ci(bench_run.results)

        lines.append("| Metric | Value | 95% CI |")
        lines.append("|---|---|---|")
        lines.append(
            f"| **Overall Score** | {score_ci.estimate:.4f} | "
            f"[{score_ci.lower:.4f}, {score_ci.upper:.4f}] |"
        )
        lines.append(
            f"| **Pass Rate** | {pass_ci.estimate:.4f} | "
            f"[{pass_ci.lower:.4f}, {pass_ci.upper:.4f}] |"
        )
        lines.append("")

        # Grade distribution
        grade_counts = {"pass": 0, "partial": 0, "fail": 0}
        for r in bench_run.results:
            grade_counts[r.grade] = grade_counts.get(r.grade, 0) + 1

        lines.append("### Grade Distribution")
        lines.append("")
        lines.append("| Grade | Count | Percentage |")
        lines.append("|---|---|---|")
        for grade, count in grade_counts.items():
            pct = count / bench_run.num_tasks * 100 if bench_run.num_tasks else 0
            emoji = {"pass": "✅", "partial": "🟡", "fail": "❌"}.get(grade, "")
            lines.append(f"| {emoji} {grade.capitalize()} | {count} | {pct:.1f}% |")
        lines.append("")

        # Per-category breakdown
        lines.append("## Results by Failure Category")
        lines.append("")

        cat_cis = self.analyzer.category_cis(bench_run.results)
        cat_agg = Scorer.aggregate_by_category(bench_run.results)

        lines.append("| Category | Tasks | Mean Score | 95% CI | Pass Rate |")
        lines.append("|---|---|---|---|---|")
        for cat in sorted(cat_agg.keys()):
            info = cat_agg[cat]
            ci = cat_cis.get(cat)
            ci_str = f"[{ci.lower:.3f}, {ci.upper:.3f}]" if ci else "N/A"
            lines.append(
                f"| {cat} | {info['count']:.0f} | "
                f"{info['mean_score']:.4f} | {ci_str} | "
                f"{info['pass_rate']:.4f} |"
            )
        lines.append("")

        # Difficulty calibration
        lines.append("## Difficulty Calibration")
        lines.append("")

        diff_cal = self.analyzer.difficulty_calibration(bench_run.results)
        lines.append("| Difficulty | Tasks | Mean Score | Pass Rate |")
        lines.append("|---|---|---|---|")
        for diff in ["easy", "medium", "hard"]:
            info = diff_cal.get(diff, {"count": 0, "mean_score": 0, "pass_rate": 0})
            lines.append(
                f"| {diff.capitalize()} | {info['count']:.0f} | "
                f"{info['mean_score']:.4f} | {info['pass_rate']:.4f} |"
            )
        lines.append("")

        # Monotonicity check
        scores = [diff_cal.get(d, {}).get("mean_score", 0) for d in ["easy", "medium", "hard"]]
        if scores[0] >= scores[1] >= scores[2]:
            lines.append(
                "> ✅ **Calibration check passed:** Easy > Medium > Hard "
                "(difficulty labels are well-calibrated)"
            )
        else:
            lines.append(
                "> ⚠️ **Calibration check failed:** Difficulty labels may need "
                "recalibration — scores do not decrease monotonically"
            )
        lines.append("")

        # Per-task details
        lines.append("## Per-Task Results")
        lines.append("")
        lines.append("| Task ID | Category | Difficulty | Score | Grade | Tests Passed |")
        lines.append("|---|---|---|---|---|---|")
        for r in sorted(bench_run.results, key=lambda x: x.task_id):
            grade_emoji = {"pass": "✅", "partial": "🟡", "fail": "❌"}.get(r.grade, "")
            lines.append(
                f"| {r.task_id} | {r.category} | {r.difficulty} | "
                f"{r.score:.3f} | {grade_emoji} {r.grade} | "
                f"{r.num_passed}/{r.num_total} |"
            )
        lines.append("")

        # Contamination report
        if contamination:
            lines.append("## Contamination Audit")
            lines.append("")
            if contamination.summary.get("clean"):
                lines.append("> ✅ **No contamination detected**")
            else:
                lines.append("> ⚠️ **Potential contamination detected**")
                if contamination.duplicate_pairs:
                    lines.append(
                        f"\n- **Internal duplicates:** "
                        f"{len(contamination.duplicate_pairs)} pairs"
                    )
                if contamination.flagged_tasks:
                    lines.append(
                        f"- **External overlaps:** "
                        f"{len(contamination.flagged_tasks)} tasks"
                    )
                if contamination.canary_results:
                    lines.append(
                        f"- **Canary leaks:** "
                        f"{len(contamination.canary_results)} tasks 🚨"
                    )
            lines.append("")

        # Footer
        lines.append("---")
        lines.append(
            f"*Generated by CodeAgent-Bench v0.1.0 at "
            f"{datetime.now(timezone.utc).isoformat()}*"
        )

        report = "\n".join(lines)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(report, encoding="utf-8")
            logger.info("Report written to %s", output_path)

        return report

    def generate_json(
        self,
        bench_run: BenchmarkRun,
        contamination: ContaminationReport | None = None,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON report."""
        stats = self.analyzer.full_summary(bench_run.results)

        report = {
            "benchmark": "codeagent-bench",
            "version": "0.1.0",
            "run": bench_run.to_dict(),
            "statistics": stats,
            "per_category": Scorer.aggregate_by_category(bench_run.results),
            "per_difficulty": Scorer.aggregate_by_difficulty(bench_run.results),
            "per_task": [r.to_dict() for r in bench_run.results],
        }

        if contamination:
            report["contamination"] = contamination.to_dict()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(
                json.dumps(report, indent=2),
                encoding="utf-8",
            )
            logger.info("JSON report written to %s", output_path)

        return report
