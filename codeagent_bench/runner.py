"""
Benchmark Runner: Orchestrate end-to-end benchmark evaluation runs.

Loads tasks, invokes agents, scores outputs, and produces structured
result files in JSONL format for downstream analysis.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from codeagent_bench.agent_wrapper import BaseAgent, create_agent
from codeagent_bench.scorer import Scorer, TaskResult
from codeagent_bench.task_loader import Task, TaskLoader

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRun:
    """Metadata and results for a complete benchmark run."""

    run_id: str
    agent_type: str
    model: str
    num_tasks: int
    results: list[TaskResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    total_time_s: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.grade == "pass") / len(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent_type": self.agent_type,
            "model": self.model,
            "num_tasks": self.num_tasks,
            "overall_score": round(self.overall_score, 4),
            "pass_rate": round(self.pass_rate, 4),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_time_s": round(self.total_time_s, 2),
            "config": self.config,
        }


class BenchmarkRunner:
    """Orchestrate loading, running, and scoring of benchmark evaluations.

    Parameters
    ----------
    agent : BaseAgent | None
        Pre-configured agent to use.  If *None*, provide ``agent_type``
        and ``agent_kwargs`` to auto-create one.
    scorer : Scorer | None
        Pre-configured scorer.  Defaults to weighted-test-pass scoring.
    task_loader : TaskLoader | None
        Pre-configured task loader.  Defaults to strict validation.
    """

    def __init__(
        self,
        agent: BaseAgent | None = None,
        scorer: Scorer | None = None,
        task_loader: TaskLoader | None = None,
    ) -> None:
        self.agent = agent
        self.scorer = scorer or Scorer()
        self.task_loader = task_loader or TaskLoader(strict=False)

    def run(
        self,
        tasks_path: str | Path,
        output_path: str | Path | None = None,
        categories: list[str] | None = None,
        difficulties: list[str] | None = None,
        task_ids: list[str] | None = None,
        agent_type: str = "mock",
        agent_kwargs: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> BenchmarkRun:
        """Execute a full benchmark run.

        Parameters
        ----------
        tasks_path : Path
            Path to a JSONL file or directory of JSONL task files.
        output_path : Path, optional
            Where to write the JSONL results.
        categories : list[str], optional
            Filter to specific failure categories.
        difficulties : list[str], optional
            Filter to specific difficulty levels.
        task_ids : list[str], optional
            Filter to specific task IDs.
        agent_type : str
            Agent type name (used if ``self.agent`` is None).
        agent_kwargs : dict
            Additional kwargs for agent creation.
        run_id : str, optional
            Custom run identifier.

        Returns
        -------
        BenchmarkRun
            Complete run metadata and results.
        """
        from datetime import datetime, timezone

        # Generate run ID
        if run_id is None:
            run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # Resolve agent
        agent = self.agent or create_agent(agent_type, **(agent_kwargs or {}))

        logger.info("Starting benchmark run %s with agent %s", run_id, agent.model)

        # Load tasks
        tasks_path = Path(tasks_path)
        if tasks_path.is_dir():
            tasks = self.task_loader.load_directory(tasks_path)
        else:
            tasks = self.task_loader.load_file(tasks_path)

        # Apply filters
        if categories:
            tasks = self.task_loader.filter_by_category(tasks, categories)
        if difficulties:
            tasks = self.task_loader.filter_by_difficulty(tasks, difficulties)
        if task_ids:
            tasks = self.task_loader.filter_by_ids(tasks, task_ids)

        if not tasks:
            logger.warning("No tasks matched the given filters")
            return BenchmarkRun(
                run_id=run_id,
                agent_type=agent_type,
                model=agent.model,
                num_tasks=0,
            )

        logger.info("Running %d tasks", len(tasks))

        # Execute benchmark
        bench_run = BenchmarkRun(
            run_id=run_id,
            agent_type=agent_type,
            model=agent.model,
            num_tasks=len(tasks),
            start_time=datetime.now(timezone.utc).isoformat(),
        )

        start = time.monotonic()
        results: list[TaskResult] = []

        for task in tqdm(tasks, desc="Evaluating", unit="task"):
            # Get agent output
            response = agent.generate(task)

            # Score the output
            task_result = self.scorer.score_task(task, response.code_output)
            task_result.execution_time_s = response.latency_s
            results.append(task_result)

            logger.info(
                "Task %s: score=%.3f grade=%s (%d/%d tests)",
                task.task_id,
                task_result.score,
                task_result.grade,
                task_result.num_passed,
                task_result.num_total,
            )

        elapsed = time.monotonic() - start
        bench_run.results = results
        bench_run.end_time = datetime.now(timezone.utc).isoformat()
        bench_run.total_time_s = elapsed

        # Write results
        if output_path:
            self._write_results(bench_run, Path(output_path))

        # Log summary
        logger.info(
            "Run %s complete: %d tasks, overall_score=%.3f, pass_rate=%.3f, time=%.1fs",
            run_id,
            len(results),
            bench_run.overall_score,
            bench_run.pass_rate,
            elapsed,
        )

        return bench_run

    @staticmethod
    def _write_results(bench_run: BenchmarkRun, output_path: Path) -> None:
        """Write benchmark results to a JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as fh:
            # Write run metadata as first line
            meta = bench_run.to_dict()
            meta["record_type"] = "run_metadata"
            fh.write(json.dumps(meta) + "\n")

            # Write individual task results
            for result in bench_run.results:
                record = result.to_dict()
                record["record_type"] = "task_result"
                record["run_id"] = bench_run.run_id
                fh.write(json.dumps(record) + "\n")

        logger.info("Results written to %s", output_path)
