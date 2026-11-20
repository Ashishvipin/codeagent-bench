"""
Scorer: Objective grading of coding agent outputs against test suites.

Supports weighted test pass, pass@k (unbiased estimator), and partial
credit scoring with per-task and per-category aggregation.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeagent_bench.task_loader import Task

logger = logging.getLogger(__name__)


@dataclass
class TestCaseResult:
    """Result of a single test case execution."""

    index: int
    passed: bool
    weight: float
    expected: Any = None
    actual: Any = None
    error: str | None = None
    description: str = ""


@dataclass
class TaskResult:
    """Aggregated result for a single task."""

    task_id: str
    category: str
    difficulty: str
    agent_output: str
    test_results: list[TestCaseResult] = field(default_factory=list)
    score: float = 0.0
    grade: str = "fail"  # "pass", "partial", "fail"
    execution_time_s: float = 0.0
    error: str | None = None

    @property
    def num_passed(self) -> int:
        return sum(1 for r in self.test_results if r.passed)

    @property
    def num_total(self) -> int:
        return len(self.test_results)

    @property
    def pass_rate(self) -> float:
        if not self.test_results:
            return 0.0
        return self.num_passed / self.num_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "score": round(self.score, 4),
            "grade": self.grade,
            "num_passed": self.num_passed,
            "num_total": self.num_total,
            "pass_rate": round(self.pass_rate, 4),
            "execution_time_s": round(self.execution_time_s, 4),
            "error": self.error,
            "test_results": [
                {
                    "index": r.index,
                    "passed": r.passed,
                    "weight": r.weight,
                    "error": r.error,
                    "description": r.description,
                }
                for r in self.test_results
            ],
        }


class Scorer:
    """Score agent outputs against task test suites.

    Parameters
    ----------
    strategy : str
        Scoring strategy: ``"weighted_test_pass"``, ``"pass_at_k"``, or
        ``"partial_credit"``.
    partial_credit_threshold : float
        Minimum weighted score needed to earn a ``"partial"`` grade
        (only used with ``"partial_credit"`` and ``"weighted_test_pass"``).
    timeout : int
        Maximum seconds to allow for test execution per task.
    """

    STRATEGIES = {"weighted_test_pass", "pass_at_k", "partial_credit"}

    def __init__(
        self,
        strategy: str = "weighted_test_pass",
        partial_credit_threshold: float = 0.5,
        timeout: int = 30,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from {self.STRATEGIES}"
            )
        self.strategy = strategy
        self.partial_credit_threshold = partial_credit_threshold
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_task(self, task: Task, agent_output: str) -> TaskResult:
        """Score a single task given the agent's code output.

        Returns a fully populated :class:`TaskResult`.
        """
        result = TaskResult(
            task_id=task.task_id,
            category=task.category,
            difficulty=task.difficulty,
            agent_output=agent_output,
        )

        if not agent_output or not agent_output.strip():
            result.error = "Empty agent output"
            result.grade = "fail"
            result.score = 0.0
            return result

        # Execute each test case
        for idx, test_case in enumerate(task.test_suite):
            tc_result = self._run_test_case(
                agent_code=agent_output,
                test_case=test_case,
                index=idx,
                setup=task.setup,
            )
            result.test_results.append(tc_result)

        # Compute score and grade
        result.score = self._compute_score(result.test_results)
        result.grade = self._assign_grade(result.score)
        return result

    def score_batch(
        self,
        tasks: list[Task],
        agent_outputs: dict[str, str],
    ) -> list[TaskResult]:
        """Score a batch of tasks.

        Parameters
        ----------
        tasks : list[Task]
            Tasks to evaluate.
        agent_outputs : dict[str, str]
            Mapping of ``task_id`` → agent code output.
        """
        results: list[TaskResult] = []
        for task in tasks:
            output = agent_outputs.get(task.task_id, "")
            result = self.score_task(task, output)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # pass@k (unbiased estimator from Chen et al., 2021)
    # ------------------------------------------------------------------

    @staticmethod
    def pass_at_k(n: int, c: int, k: int) -> float:
        """Compute the unbiased pass@k estimator.

        Parameters
        ----------
        n : int
            Total number of samples generated for the task.
        c : int
            Number of samples that passed all tests.
        k : int
            The k in pass@k.

        Returns
        -------
        float
            Estimated pass@k probability.
        """
        if n - c < k:
            return 1.0
        return 1.0 - math.prod(
            (n - c - i) / (n - i) for i in range(k)
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_by_category(
        results: list[TaskResult],
    ) -> dict[str, dict[str, float]]:
        """Aggregate scores by failure category."""
        from collections import defaultdict

        cat_scores: dict[str, list[float]] = defaultdict(list)
        cat_pass: dict[str, list[bool]] = defaultdict(list)

        for r in results:
            cat_scores[r.category].append(r.score)
            cat_pass[r.category].append(r.grade == "pass")

        summary = {}
        for cat, scores in cat_scores.items():
            n = len(scores)
            summary[cat] = {
                "mean_score": sum(scores) / n if n else 0.0,
                "pass_rate": sum(cat_pass[cat]) / n if n else 0.0,
                "count": n,
            }
        return summary

    @staticmethod
    def aggregate_by_difficulty(
        results: list[TaskResult],
    ) -> dict[str, dict[str, float]]:
        """Aggregate scores by difficulty level."""
        from collections import defaultdict

        diff_scores: dict[str, list[float]] = defaultdict(list)
        diff_pass: dict[str, list[bool]] = defaultdict(list)

        for r in results:
            diff_scores[r.difficulty].append(r.score)
            diff_pass[r.difficulty].append(r.grade == "pass")

        summary = {}
        for diff, scores in diff_scores.items():
            n = len(scores)
            summary[diff] = {
                "mean_score": sum(scores) / n if n else 0.0,
                "pass_rate": sum(diff_pass[diff]) / n if n else 0.0,
                "count": n,
            }
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_test_case(
        self,
        agent_code: str,
        test_case: dict[str, Any],
        index: int,
        setup: dict[str, Any],
    ) -> TestCaseResult:
        """Execute a single test case by running the agent's code in a
        subprocess with the test inputs, then comparing against expected
        output or expected exception."""

        tc_result = TestCaseResult(
            index=index,
            passed=False,
            weight=test_case.get("weight", 0.0),
            expected=test_case.get("expected", test_case.get("expected_exception")),
            description=test_case.get("description", ""),
        )

        expected_exception = test_case.get("expected_exception")
        expected_value = test_case.get("expected")
        test_input = test_case.get("input", {})

        # Build the test script
        test_script = self._build_test_script(
            agent_code=agent_code,
            test_input=test_input,
            expected_value=expected_value,
            expected_exception=expected_exception,
            setup=setup,
        )

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
            ) as tmp:
                tmp.write(test_script)
                tmp_path = tmp.name

            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir(),
            )

            if proc.returncode == 0:
                tc_result.passed = True
                tc_result.actual = proc.stdout.strip()
            else:
                tc_result.passed = False
                tc_result.error = proc.stderr.strip()[:500]
                tc_result.actual = proc.stdout.strip()

        except subprocess.TimeoutExpired:
            tc_result.error = f"Timeout after {self.timeout}s"
        except Exception as exc:
            tc_result.error = str(exc)[:500]
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return tc_result

    @staticmethod
    def _build_test_script(
        agent_code: str,
        test_input: dict[str, Any],
        expected_value: Any,
        expected_exception: str | None,
        setup: dict[str, Any],
    ) -> str:
        """Build a self-contained Python test script."""
        setup_code = setup.get("setup_code", "") if setup else ""

        # Create any setup files
        file_setup_lines = []
        if setup and "files" in setup:
            for fname, content in setup["files"].items():
                escaped = content.replace("\\", "\\\\").replace("'", "\\'")
                file_setup_lines.append(
                    f"import pathlib; "
                    f"pathlib.Path('{fname}').write_text('{escaped}')"
                )

        file_setup = "\n".join(file_setup_lines)

        if expected_exception:
            return textwrap.dedent(f"""\
                import sys, json
                {file_setup}
                {setup_code}

                # --- Agent code ---
                {agent_code}
                # --- End agent code ---

                test_input = {json.dumps(test_input)}
                try:
                    # Attempt to find and call the main function
                    import inspect
                    funcs = [
                        obj for name, obj in list(locals().items())
                        if callable(obj) and not name.startswith('_')
                        and name not in ('json', 'sys', 'inspect', 'pathlib')
                    ]
                    if funcs:
                        result = funcs[-1](**test_input)
                        print(f"ERROR: Expected {expected_exception} but got result: {{result}}", file=sys.stderr)
                        sys.exit(1)
                except {expected_exception}:
                    print("PASS")
                    sys.exit(0)
                except Exception as e:
                    print(f"ERROR: Expected {expected_exception} but got {{type(e).__name__}}: {{e}}", file=sys.stderr)
                    sys.exit(1)
            """)
        else:
            return textwrap.dedent(f"""\
                import sys, json
                {file_setup}
                {setup_code}

                # --- Agent code ---
                {agent_code}
                # --- End agent code ---

                test_input = {json.dumps(test_input)}
                expected = {json.dumps(expected_value)}

                try:
                    import inspect
                    funcs = [
                        obj for name, obj in list(locals().items())
                        if callable(obj) and not name.startswith('_')
                        and name not in ('json', 'sys', 'inspect', 'pathlib')
                    ]
                    if funcs:
                        result = funcs[-1](**test_input)
                        if result == expected:
                            print("PASS")
                            sys.exit(0)
                        else:
                            print(f"FAIL: expected {{expected}}, got {{result}}", file=sys.stderr)
                            sys.exit(1)
                    else:
                        print("ERROR: No callable function found in agent output", file=sys.stderr)
                        sys.exit(1)
                except Exception as e:
                    print(f"ERROR: {{type(e).__name__}}: {{e}}", file=sys.stderr)
                    sys.exit(1)
            """)

    def _compute_score(self, test_results: list[TestCaseResult]) -> float:
        """Compute the weighted score from individual test results."""
        if not test_results:
            return 0.0
        total_weight = sum(r.weight for r in test_results)
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(r.weight for r in test_results if r.passed)
        return weighted_sum / total_weight

    def _assign_grade(self, score: float) -> str:
        """Assign pass / partial / fail grade."""
        if score >= 1.0 - 1e-9:
            return "pass"
        elif score >= self.partial_credit_threshold:
            return "partial"
        return "fail"
