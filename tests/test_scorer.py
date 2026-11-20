"""Tests for Scorer: grading logic, pass@k, and aggregation."""

from __future__ import annotations

import pytest

from codeagent_bench.scorer import Scorer, TaskResult, TestCaseResult
from codeagent_bench.task_loader import Task


class TestPassAtK:
    """Test the unbiased pass@k estimator."""

    def test_all_pass(self):
        # If all n samples pass, pass@k should be 1.0
        assert Scorer.pass_at_k(n=10, c=10, k=1) == 1.0

    def test_none_pass(self):
        # If no samples pass, pass@k = 0.0
        assert Scorer.pass_at_k(n=10, c=0, k=1) == 0.0

    def test_one_of_ten(self):
        # 1 correct out of 10, pass@1
        result = Scorer.pass_at_k(n=10, c=1, k=1)
        assert 0.09 < result < 0.11  # Should be 0.1

    def test_pass_at_k_increases_with_k(self):
        p1 = Scorer.pass_at_k(n=10, c=3, k=1)
        p5 = Scorer.pass_at_k(n=10, c=3, k=5)
        assert p5 > p1  # Higher k → higher probability

    def test_pass_at_k_edge_case(self):
        # When n - c < k, should return 1.0
        assert Scorer.pass_at_k(n=5, c=4, k=2) == 1.0


class TestScoring:
    """Test the scoring logic."""

    def test_empty_output_scores_zero(self):
        task = Task(
            "ERR_001", "incomplete_error_handling", "easy",
            "Test Task", "D" * 20, "P" * 20,
            [{"input": {}, "expected": 42, "weight": 1.0}],
            "weighted_test_pass",
        )
        scorer = Scorer()
        result = scorer.score_task(task, "")
        assert result.score == 0.0
        assert result.grade == "fail"

    def test_grade_assignment(self):
        scorer = Scorer(partial_credit_threshold=0.5)

        # Full pass
        assert scorer._assign_grade(1.0) == "pass"

        # Partial
        assert scorer._assign_grade(0.7) == "partial"

        # Fail
        assert scorer._assign_grade(0.3) == "fail"

    def test_compute_score_all_pass(self):
        scorer = Scorer()
        results = [
            TestCaseResult(0, True, 0.5),
            TestCaseResult(1, True, 0.5),
        ]
        assert scorer._compute_score(results) == 1.0

    def test_compute_score_partial(self):
        scorer = Scorer()
        results = [
            TestCaseResult(0, True, 0.6),
            TestCaseResult(1, False, 0.4),
        ]
        score = scorer._compute_score(results)
        assert abs(score - 0.6) < 0.01

    def test_compute_score_empty(self):
        scorer = Scorer()
        assert scorer._compute_score([]) == 0.0


class TestAggregation:
    """Test score aggregation by category and difficulty."""

    def _make_results(self) -> list[TaskResult]:
        return [
            TaskResult("ERR_001", "incomplete_error_handling", "easy", "", score=0.8, grade="partial"),
            TaskResult("ERR_002", "incomplete_error_handling", "medium", "", score=1.0, grade="pass"),
            TaskResult("API_001", "incorrect_api_usage", "easy", "", score=0.0, grade="fail"),
            TaskResult("LOGIC_001", "logic_errors", "hard", "", score=0.5, grade="partial"),
        ]

    def test_aggregate_by_category(self):
        results = self._make_results()
        agg = Scorer.aggregate_by_category(results)

        assert "incomplete_error_handling" in agg
        assert agg["incomplete_error_handling"]["count"] == 2
        assert abs(agg["incomplete_error_handling"]["mean_score"] - 0.9) < 0.01

    def test_aggregate_by_difficulty(self):
        results = self._make_results()
        agg = Scorer.aggregate_by_difficulty(results)

        assert "easy" in agg
        assert agg["easy"]["count"] == 2
        assert abs(agg["easy"]["mean_score"] - 0.4) < 0.01

    def test_empty_results(self):
        agg = Scorer.aggregate_by_category([])
        assert agg == {}


class TestTaskResult:
    def test_pass_rate(self):
        result = TaskResult("T001", "cat", "easy", "code")
        result.test_results = [
            TestCaseResult(0, True, 0.5),
            TestCaseResult(1, False, 0.5),
        ]
        assert result.pass_rate == 0.5

    def test_to_dict(self):
        result = TaskResult("T001", "cat", "easy", "code", score=0.75, grade="partial")
        d = result.to_dict()
        assert d["task_id"] == "T001"
        assert d["score"] == 0.75
        assert d["grade"] == "partial"
