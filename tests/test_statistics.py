"""Tests for StatisticalAnalyzer: bootstrap CI, McNemar's, calibration."""

from __future__ import annotations

import numpy as np
import pytest

from codeagent_bench.scorer import TaskResult
from codeagent_bench.statistics import StatisticalAnalyzer


class TestBootstrapCI:
    def test_ci_contains_true_mean(self):
        """Bootstrap CI should contain the sample mean."""
        analyzer = StatisticalAnalyzer(n_bootstrap=5000, seed=42)
        scores = [0.8, 0.9, 0.7, 0.85, 0.95, 0.75, 0.6, 0.9, 0.8, 0.85]
        ci = analyzer.bootstrap_ci(scores)
        assert ci.lower <= ci.estimate <= ci.upper

    def test_ci_narrower_with_more_data(self):
        """More data should produce narrower CI."""
        analyzer = StatisticalAnalyzer(n_bootstrap=5000, seed=42)
        rng = np.random.default_rng(42)

        small = rng.uniform(0.5, 1.0, size=10).tolist()
        large = rng.uniform(0.5, 1.0, size=100).tolist()

        ci_small = analyzer.bootstrap_ci(small)
        ci_large = analyzer.bootstrap_ci(large)

        width_small = ci_small.upper - ci_small.lower
        width_large = ci_large.upper - ci_large.lower
        assert width_large < width_small

    def test_empty_scores(self):
        analyzer = StatisticalAnalyzer()
        ci = analyzer.bootstrap_ci([])
        assert ci.estimate == 0.0

    def test_single_score(self):
        analyzer = StatisticalAnalyzer(n_bootstrap=1000, seed=42)
        ci = analyzer.bootstrap_ci([0.75])
        assert ci.estimate == 0.75
        assert ci.lower == 0.75
        assert ci.upper == 0.75

    def test_median_statistic(self):
        analyzer = StatisticalAnalyzer(n_bootstrap=5000, seed=42)
        scores = [0.1, 0.5, 0.5, 0.5, 0.9]
        ci = analyzer.bootstrap_ci(scores, statistic="median")
        assert ci.estimate == 0.5


class TestMcNemarsTest:
    def _make_results(self, task_ids: list[str], grades: list[str]) -> list[TaskResult]:
        return [
            TaskResult(tid, "cat", "easy", "", grade=g)
            for tid, g in zip(task_ids, grades)
        ]

    def test_no_difference(self):
        """Same results → not significant."""
        analyzer = StatisticalAnalyzer()
        ids = [f"T{i:03d}" for i in range(20)]
        grades = ["pass"] * 10 + ["fail"] * 10

        results_a = self._make_results(ids, grades)
        results_b = self._make_results(ids, grades)

        comp = analyzer.mcnemar_test(results_a, results_b)
        assert not comp.significant
        assert comp.p_value == 1.0

    def test_significant_difference(self):
        """Clear difference should be significant (with enough data)."""
        analyzer = StatisticalAnalyzer(significance=0.05)
        ids = [f"T{i:03d}" for i in range(50)]

        # Model A passes most, Model B fails most
        grades_a = ["pass"] * 40 + ["fail"] * 10
        grades_b = ["fail"] * 40 + ["pass"] * 10

        results_a = self._make_results(ids, grades_a)
        results_b = self._make_results(ids, grades_b)

        comp = analyzer.mcnemar_test(results_a, results_b)
        assert comp.significant
        assert comp.p_value < 0.05

    def test_effect_size_range(self):
        analyzer = StatisticalAnalyzer()
        ids = [f"T{i:03d}" for i in range(20)]
        results_a = self._make_results(ids, ["pass"] * 15 + ["fail"] * 5)
        results_b = self._make_results(ids, ["pass"] * 5 + ["fail"] * 15)

        comp = analyzer.mcnemar_test(results_a, results_b)
        assert 0 <= comp.effect_size <= 0.5


class TestDifficultyCalibration:
    def test_calibration_structure(self):
        results = [
            TaskResult("T1", "cat", "easy", "", score=0.9, grade="pass"),
            TaskResult("T2", "cat", "medium", "", score=0.5, grade="partial"),
            TaskResult("T3", "cat", "hard", "", score=0.1, grade="fail"),
        ]
        cal = StatisticalAnalyzer.difficulty_calibration(results)
        assert "easy" in cal
        assert "medium" in cal
        assert "hard" in cal
        assert cal["easy"]["mean_score"] > cal["hard"]["mean_score"]

    def test_empty_results(self):
        cal = StatisticalAnalyzer.difficulty_calibration([])
        assert cal["easy"]["count"] == 0


class TestCohensKappa:
    def test_perfect_agreement(self):
        ratings = ["pass", "fail", "partial", "pass", "fail"]
        kappa = StatisticalAnalyzer.cohens_kappa(ratings, ratings)
        assert abs(kappa - 1.0) < 0.01

    def test_no_agreement(self):
        """Random opposite ratings should give low/negative kappa."""
        a = ["pass", "pass", "pass", "fail", "fail", "fail"]
        b = ["fail", "fail", "fail", "pass", "pass", "pass"]
        kappa = StatisticalAnalyzer.cohens_kappa(a, b)
        assert kappa < 0.1

    def test_empty_ratings(self):
        kappa = StatisticalAnalyzer.cohens_kappa([], [])
        assert kappa == 0.0


class TestFullSummary:
    def test_summary_keys(self):
        analyzer = StatisticalAnalyzer(n_bootstrap=100, seed=42)
        results = [
            TaskResult("T1", "incomplete_error_handling", "easy", "", score=0.8, grade="partial"),
            TaskResult("T2", "logic_errors", "hard", "", score=0.3, grade="fail"),
        ]
        summary = analyzer.full_summary(results)
        assert "n_tasks" in summary
        assert "overall_score" in summary
        assert "pass_rate" in summary
        assert "by_category" in summary
        assert "difficulty_calibration" in summary
