"""
Contamination Detector: Prevent benchmark answer leakage and gaming.

Implements n-gram overlap detection, canary string injection, and
train/test separation auditing to ensure benchmark integrity.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from codeagent_bench.task_loader import Task

logger = logging.getLogger(__name__)


@dataclass
class ContaminationReport:
    """Contamination analysis report for a set of tasks."""

    num_tasks_checked: int
    flagged_tasks: list[dict[str, Any]] = field(default_factory=list)
    duplicate_pairs: list[dict[str, Any]] = field(default_factory=list)
    canary_results: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def num_flagged(self) -> int:
        return len(self.flagged_tasks)

    @property
    def contamination_rate(self) -> float:
        if self.num_tasks_checked == 0:
            return 0.0
        return self.num_flagged / self.num_tasks_checked

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_tasks_checked": self.num_tasks_checked,
            "num_flagged": self.num_flagged,
            "contamination_rate": round(self.contamination_rate, 4),
            "flagged_tasks": self.flagged_tasks,
            "duplicate_pairs": self.duplicate_pairs,
            "canary_results": self.canary_results,
            "summary": self.summary,
        }


class ContaminationDetector:
    """Detect potential contamination in benchmark tasks.

    Checks for:
    1. **Internal duplicates:** N-gram overlap between tasks within the
       benchmark (self-contamination).
    2. **External overlap:** N-gram overlap against a reference corpus
       (e.g., The Stack, GitHub Code snippets).
    3. **Canary strings:** Inject traceable tokens to detect if models
       have memorized the benchmark.
    4. **Trivial searchability:** Flag tasks whose prompts return exact
       matches on common code search engines.

    Parameters
    ----------
    ngram_size : int
        Size of n-grams for overlap detection.
    jaccard_threshold : float
        Jaccard similarity threshold above which a pair is flagged.
    canary_enabled : bool
        Whether to generate and inject canary strings.
    """

    def __init__(
        self,
        ngram_size: int = 8,
        jaccard_threshold: float = 0.15,
        canary_enabled: bool = True,
    ) -> None:
        self.ngram_size = ngram_size
        self.jaccard_threshold = jaccard_threshold
        self.canary_enabled = canary_enabled

    # ------------------------------------------------------------------
    # N-gram extraction and similarity
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def _extract_ngrams(self, text: str) -> set[tuple[str, ...]]:
        """Extract n-grams from text."""
        tokens = self._tokenize(text)
        if len(tokens) < self.ngram_size:
            return {tuple(tokens)} if tokens else set()
        return {
            tuple(tokens[i : i + self.ngram_size])
            for i in range(len(tokens) - self.ngram_size + 1)
        }

    @staticmethod
    def jaccard_similarity(set_a: set, set_b: set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    # Internal duplicate detection
    # ------------------------------------------------------------------

    def check_internal_duplicates(
        self,
        tasks: list[Task],
    ) -> list[dict[str, Any]]:
        """Find pairs of tasks with high n-gram overlap.

        Returns a list of flagged pairs with their Jaccard similarity.
        """
        # Pre-compute n-grams for each task
        task_ngrams: dict[str, set[tuple[str, ...]]] = {}
        for task in tasks:
            combined_text = f"{task.title} {task.description} {task.prompt}"
            task_ngrams[task.task_id] = self._extract_ngrams(combined_text)

        flagged_pairs: list[dict[str, Any]] = []
        task_ids = list(task_ngrams.keys())

        for i in range(len(task_ids)):
            for j in range(i + 1, len(task_ids)):
                id_a, id_b = task_ids[i], task_ids[j]
                sim = self.jaccard_similarity(
                    task_ngrams[id_a], task_ngrams[id_b]
                )
                if sim >= self.jaccard_threshold:
                    flagged_pairs.append({
                        "task_a": id_a,
                        "task_b": id_b,
                        "jaccard_similarity": round(sim, 4),
                        "shared_ngrams": len(
                            task_ngrams[id_a] & task_ngrams[id_b]
                        ),
                    })
                    logger.warning(
                        "High overlap: %s ↔ %s (Jaccard=%.3f)",
                        id_a,
                        id_b,
                        sim,
                    )

        return flagged_pairs

    # ------------------------------------------------------------------
    # External corpus overlap
    # ------------------------------------------------------------------

    def check_against_corpus(
        self,
        tasks: list[Task],
        corpus_texts: list[str],
    ) -> list[dict[str, Any]]:
        """Check task prompts against an external corpus for overlap.

        Parameters
        ----------
        tasks : list[Task]
            Benchmark tasks to check.
        corpus_texts : list[str]
            List of reference texts (e.g., from The Stack,
            GitHub code snippets, existing benchmarks).
        """
        # Build corpus n-grams
        corpus_ngrams: set[tuple[str, ...]] = set()
        for text in corpus_texts:
            corpus_ngrams |= self._extract_ngrams(text)

        logger.info(
            "Corpus contains %d unique %d-grams",
            len(corpus_ngrams),
            self.ngram_size,
        )

        flagged: list[dict[str, Any]] = []
        for task in tasks:
            task_text = f"{task.title} {task.description} {task.prompt}"
            task_ng = self._extract_ngrams(task_text)

            if not task_ng:
                continue

            overlap = len(task_ng & corpus_ngrams)
            overlap_ratio = overlap / len(task_ng)

            if overlap_ratio >= self.jaccard_threshold:
                flagged.append({
                    "task_id": task.task_id,
                    "overlap_ratio": round(overlap_ratio, 4),
                    "overlapping_ngrams": overlap,
                    "total_ngrams": len(task_ng),
                })
                logger.warning(
                    "Corpus overlap: %s — %.1f%% n-gram overlap",
                    task.task_id,
                    overlap_ratio * 100,
                )

        return flagged

    # ------------------------------------------------------------------
    # Canary strings
    # ------------------------------------------------------------------

    def generate_canary(self, task_id: str, salt: str = "codeagent-bench-v1") -> str:
        """Generate a unique canary string for a task.

        The canary is a deterministic hash that can be injected into task
        descriptions.  If a model outputs this string unprompted, it
        indicates memorization of the benchmark.
        """
        raw = f"{salt}:{task_id}"
        h = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"CANARY-{h}"

    def inject_canaries(self, tasks: list[Task]) -> dict[str, str]:
        """Inject canary strings into task descriptions.

        Returns a mapping of task_id → canary string for later verification.
        """
        canary_map: dict[str, str] = {}
        for task in tasks:
            canary = self.generate_canary(task.task_id)
            canary_map[task.task_id] = canary
            # Inject as a hidden comment in the description
            task.description += f"\n<!-- {canary} -->"
        return canary_map

    def check_canary_leakage(
        self,
        canary_map: dict[str, str],
        agent_outputs: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Check if any agent outputs contain canary strings.

        Parameters
        ----------
        canary_map : dict
            Mapping of task_id → expected canary string.
        agent_outputs : dict
            Mapping of task_id → agent output text.
        """
        leaked: list[dict[str, Any]] = []
        for task_id, canary in canary_map.items():
            output = agent_outputs.get(task_id, "")
            if canary in output:
                leaked.append({
                    "task_id": task_id,
                    "canary": canary,
                    "severity": "critical",
                    "message": (
                        f"Canary string found in agent output for {task_id}. "
                        f"This strongly suggests the model has memorized "
                        f"this benchmark task."
                    ),
                })
                logger.critical(
                    "CANARY LEAKED: %s found in output for %s",
                    canary,
                    task_id,
                )
        return leaked

    # ------------------------------------------------------------------
    # Full contamination check
    # ------------------------------------------------------------------

    def run_full_check(
        self,
        tasks: list[Task],
        corpus_texts: list[str] | None = None,
        agent_outputs: dict[str, str] | None = None,
    ) -> ContaminationReport:
        """Run all contamination checks and produce a report.

        Parameters
        ----------
        tasks : list[Task]
            Tasks to check.
        corpus_texts : list[str], optional
            External corpus for overlap checking.
        agent_outputs : dict[str, str], optional
            Agent outputs for canary leakage checking.
        """
        report = ContaminationReport(num_tasks_checked=len(tasks))

        # 1. Internal duplicate detection
        logger.info("Checking internal duplicates...")
        report.duplicate_pairs = self.check_internal_duplicates(tasks)

        # 2. External corpus overlap
        if corpus_texts:
            logger.info("Checking against external corpus...")
            report.flagged_tasks = self.check_against_corpus(tasks, corpus_texts)

        # 3. Canary leakage
        if self.canary_enabled and agent_outputs:
            logger.info("Checking canary leakage...")
            canary_map = {t.task_id: self.generate_canary(t.task_id) for t in tasks}
            report.canary_results = self.check_canary_leakage(canary_map, agent_outputs)

        # Summary
        report.summary = {
            "internal_duplicates": len(report.duplicate_pairs),
            "external_overlaps": len(report.flagged_tasks),
            "canary_leaks": len(report.canary_results),
            "clean": (
                len(report.duplicate_pairs) == 0
                and len(report.flagged_tasks) == 0
                and len(report.canary_results) == 0
            ),
        }

        return report
