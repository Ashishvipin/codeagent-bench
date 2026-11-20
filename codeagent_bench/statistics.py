"""
Statistics: Rigorous statistical analysis for benchmark results.

Implements bootstrap confidence intervals, McNemar's paired comparison
test, difficulty calibration, and inter-annotator agreement metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from codeagent_bench.scorer import TaskResult

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """A confidence interval with point estimate."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    n_bootstrap: int

    def __str__(self) -> str:
        return (
            f"{self.estimate:.4f} "
            f"[{self.lower:.4f}, {self.upper:.4f}] "
            f"({self.confidence*100:.0f}% CI, n_boot={self.n_bootstrap})"
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "estimate": round(self.estimate, 4),
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "confidence": self.confidence,
            "n_bootstrap": self.n_bootstrap,
        }


@dataclass
class PairedComparisonResult:
    """Result of a paired model comparison (McNemar's test)."""

    model_a: str
    model_b: str
    model_a_score: float
    model_b_score: float
    statistic: float
    p_value: float
    significant: bool
    n_tasks: int
    effect_size: float  # Cohen's g

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "model_a_score": round(self.model_a_score, 4),
            "model_b_score": round(self.model_b_score, 4),
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "significant": self.significant,
            "n_tasks": self.n_tasks,
            "effect_size": round(self.effect_size, 4),
        }


class StatisticalAnalyzer:
    """Statistical analysis engine for benchmark results.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap resamples for confidence intervals.
    confidence : float
        Confidence level (default: 0.95 for 95% CI).
    significance : float
        Significance level for hypothesis tests (default: 0.05).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_bootstrap: int = 10_000,
        confidence: float = 0.95,
        significance: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.significance = significance
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Bootstrap confidence intervals
    # ------------------------------------------------------------------

    def bootstrap_ci(
        self,
        scores: list[float] | np.ndarray,
        statistic: str = "mean",
    ) -> ConfidenceInterval:
        """Compute a bootstrap confidence interval for the given statistic.

        Parameters
        ----------
        scores : array-like
            Individual task scores.
        statistic : str
            Which statistic to compute: ``"mean"`` or ``"median"``.
        """
        scores = np.asarray(scores, dtype=float)
        n = len(scores)

        if n == 0:
            return ConfidenceInterval(0.0, 0.0, 0.0, self.confidence, 0)

        stat_fn = np.mean if statistic == "mean" else np.median
        point_estimate = float(stat_fn(scores))

        # Generate bootstrap samples
        boot_indices = self.rng.choice(n, size=(self.n_bootstrap, n), replace=True)
        boot_stats = np.array([float(stat_fn(scores[idx])) for idx in boot_indices])

        # Percentile method CI
        alpha = 1.0 - self.confidence
        lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

        return ConfidenceInterval(
            estimate=point_estimate,
            lower=lower,
            upper=upper,
            confidence=self.confidence,
            n_bootstrap=self.n_bootstrap,
        )

    def score_ci(self, results: list[TaskResult]) -> ConfidenceInterval:
        """Compute bootstrap CI for overall benchmark score."""
        scores = [r.score for r in results]
        return self.bootstrap_ci(scores)

    def pass_rate_ci(self, results: list[TaskResult]) -> ConfidenceInterval:
        """Compute bootstrap CI for pass rate."""
        passes = [1.0 if r.grade == "pass" else 0.0 for r in results]
        return self.bootstrap_ci(passes)

    def category_cis(
        self,
        results: list[TaskResult],
    ) -> dict[str, ConfidenceInterval]:
        """Compute per-category bootstrap CIs."""
        from collections import defaultdict

        cat_scores: dict[str, list[float]] = defaultdict(list)
        for r in results:
            cat_scores[r.category].append(r.score)
        return {
            cat: self.bootstrap_ci(scores)
            for cat, scores in cat_scores.items()
        }

    # ------------------------------------------------------------------
    # Paired comparison: McNemar's test
    # ------------------------------------------------------------------

    def mcnemar_test(
        self,
        results_a: list[TaskResult],
        results_b: list[TaskResult],
        model_a: str = "Model A",
        model_b: str = "Model B",
    ) -> PairedComparisonResult:
        """McNemar's test for paired model comparison.

        Compares whether two models have significantly different pass
        rates on the same set of tasks.

        Parameters
        ----------
        results_a, results_b : list[TaskResult]
            Results from two models on the **same** tasks (matched by task_id).
        """
        # Build pass/fail lookup by task_id
        pass_a = {r.task_id: r.grade == "pass" for r in results_a}
        pass_b = {r.task_id: r.grade == "pass" for r in results_b}

        # Find common tasks
        common = set(pass_a.keys()) & set(pass_b.keys())
        if len(common) < 10:
            logger.warning(
                "Only %d common tasks — McNemar's test may be unreliable",
                len(common),
            )

        # Build 2×2 contingency table
        #              B pass  B fail
        #  A pass        a       b
        #  A fail        c       d
        b = sum(1 for t in common if pass_a[t] and not pass_b[t])
        c = sum(1 for t in common if not pass_a[t] and pass_b[t])

        # McNemar's test (with continuity correction)
        if b + c == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            statistic = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = float(sp_stats.chi2.sf(statistic, df=1))

        # Effect size: Cohen's g
        if b + c > 0:
            effect_size = abs(b / (b + c) - 0.5)
        else:
            effect_size = 0.0

        score_a = sum(pass_a[t] for t in common) / len(common) if common else 0.0
        score_b = sum(pass_b[t] for t in common) / len(common) if common else 0.0

        return PairedComparisonResult(
            model_a=model_a,
            model_b=model_b,
            model_a_score=score_a,
            model_b_score=score_b,
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.significance,
            n_tasks=len(common),
            effect_size=effect_size,
        )

    # ------------------------------------------------------------------
    # Difficulty calibration
    # ------------------------------------------------------------------

    @staticmethod
    def difficulty_calibration(
        results: list[TaskResult],
    ) -> dict[str, dict[str, float]]:
        """Compute pass rate and mean score per difficulty level.

        Useful for validating that 'easy' tasks are indeed easier
        than 'hard' tasks — if not, the difficulty labels need recalibration.
        """
        from collections import defaultdict

        diff_scores: dict[str, list[float]] = defaultdict(list)
        diff_pass: dict[str, list[bool]] = defaultdict(list)

        for r in results:
            diff_scores[r.difficulty].append(r.score)
            diff_pass[r.difficulty].append(r.grade == "pass")

        calibration: dict[str, dict[str, float]] = {}
        for diff in ["easy", "medium", "hard"]:
            scores = diff_scores.get(diff, [])
            passes = diff_pass.get(diff, [])
            n = len(scores)
            calibration[diff] = {
                "count": n,
                "mean_score": sum(scores) / n if n else 0.0,
                "pass_rate": sum(passes) / n if n else 0.0,
            }

        return calibration

    # ------------------------------------------------------------------
    # Inter-annotator agreement (Cohen's kappa)
    # ------------------------------------------------------------------

    @staticmethod
    def cohens_kappa(
        ratings_a: list[str],
        ratings_b: list[str],
    ) -> float:
        """Compute Cohen's kappa for two raters.

        Parameters
        ----------
        ratings_a, ratings_b : list[str]
            Categorical ratings from two annotators.
        """
        assert len(ratings_a) == len(ratings_b), "Rating lists must be same length"
        n = len(ratings_a)
        if n == 0:
            return 0.0

        categories = sorted(set(ratings_a) | set(ratings_b))
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        k = len(categories)

        # Build confusion matrix
        matrix = np.zeros((k, k), dtype=int)
        for a, b in zip(ratings_a, ratings_b):
            matrix[cat_to_idx[a], cat_to_idx[b]] += 1

        # Observed agreement
        p_o = np.trace(matrix) / n

        # Expected agreement
        row_sums = matrix.sum(axis=1) / n
        col_sums = matrix.sum(axis=0) / n
        p_e = float(np.sum(row_sums * col_sums))

        if abs(1 - p_e) < 1e-10:
            return 1.0  # Perfect agreement

        return float((p_o - p_e) / (1 - p_e))

    # ------------------------------------------------------------------
    # Full statistical summary
    # ------------------------------------------------------------------

    def full_summary(
        self,
        results: list[TaskResult],
    ) -> dict[str, Any]:
        """Compute a comprehensive statistical summary."""
        return {
            "n_tasks": len(results),
            "overall_score": self.score_ci(results).to_dict(),
            "pass_rate": self.pass_rate_ci(results).to_dict(),
            "by_category": {
                cat: ci.to_dict()
                for cat, ci in self.category_cis(results).items()
            },
            "difficulty_calibration": self.difficulty_calibration(results),
        }
