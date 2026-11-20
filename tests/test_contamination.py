"""Tests for ContaminationDetector: n-gram overlap, canaries, and full audit."""

from __future__ import annotations

import pytest

from codeagent_bench.contamination import ContaminationDetector
from codeagent_bench.task_loader import Task


def _make_task(task_id: str, title: str, description: str, prompt: str) -> Task:
    return Task(
        task_id=task_id,
        category="incomplete_error_handling",
        difficulty="easy",
        title=title,
        description=description,
        prompt=prompt,
        test_suite=[{"input": {}, "expected": 1, "weight": 1.0}],
        grading="weighted_test_pass",
    )


class TestNgramExtraction:
    def test_basic_ngrams(self):
        detector = ContaminationDetector(ngram_size=3)
        ngrams = detector._extract_ngrams("the quick brown fox jumps")
        assert len(ngrams) == 3  # 5 tokens - 3 + 1 = 3 trigrams

    def test_short_text(self):
        detector = ContaminationDetector(ngram_size=8)
        ngrams = detector._extract_ngrams("hello world")
        assert len(ngrams) == 1  # Too short for 8-grams, returns whole text

    def test_empty_text(self):
        detector = ContaminationDetector()
        ngrams = detector._extract_ngrams("")
        assert len(ngrams) == 0


class TestJaccardSimilarity:
    def test_identical_sets(self):
        s = {("a", "b"), ("c", "d")}
        assert ContaminationDetector.jaccard_similarity(s, s) == 1.0

    def test_disjoint_sets(self):
        a = {("a", "b")}
        b = {("c", "d")}
        assert ContaminationDetector.jaccard_similarity(a, b) == 0.0

    def test_empty_sets(self):
        assert ContaminationDetector.jaccard_similarity(set(), set()) == 0.0

    def test_partial_overlap(self):
        a = {("a",), ("b",), ("c",)}
        b = {("b",), ("c",), ("d",)}
        sim = ContaminationDetector.jaccard_similarity(a, b)
        assert abs(sim - 0.5) < 0.01  # 2 shared / 4 total


class TestInternalDuplicates:
    def test_no_duplicates(self):
        detector = ContaminationDetector(ngram_size=3, jaccard_threshold=0.3)
        tasks = [
            _make_task("ERR_001", "Task about file reading",
                       "This task tests file reading with error handling and edge cases",
                       "Write a function to read files safely"),
            _make_task("ERR_002", "Task about network requests",
                       "This task tests HTTP client resilience and timeout handling behavior",
                       "Write a function to make HTTP requests"),
        ]
        pairs = detector.check_internal_duplicates(tasks)
        assert len(pairs) == 0

    def test_detects_duplicates(self):
        detector = ContaminationDetector(ngram_size=3, jaccard_threshold=0.3)
        tasks = [
            _make_task("ERR_001", "Read file and handle errors",
                       "Write a function that reads a file and handles errors properly",
                       "Write a function that reads a file and handles errors properly"),
            _make_task("ERR_002", "Read file and handle errors too",
                       "Write a function that reads a file and handles errors properly here",
                       "Write a function that reads a file and handles errors properly here"),
        ]
        pairs = detector.check_internal_duplicates(tasks)
        assert len(pairs) >= 1
        assert pairs[0]["jaccard_similarity"] > 0.3


class TestCanaryStrings:
    def test_canary_generation(self):
        detector = ContaminationDetector()
        canary = detector.generate_canary("ERR_001")
        assert canary.startswith("CANARY-")
        assert len(canary) == 23  # "CANARY-" + 16 hex chars

    def test_canary_deterministic(self):
        detector = ContaminationDetector()
        c1 = detector.generate_canary("ERR_001")
        c2 = detector.generate_canary("ERR_001")
        assert c1 == c2

    def test_canary_unique_per_task(self):
        detector = ContaminationDetector()
        c1 = detector.generate_canary("ERR_001")
        c2 = detector.generate_canary("ERR_002")
        assert c1 != c2

    def test_canary_leakage_detection(self):
        detector = ContaminationDetector()
        canary_map = {"ERR_001": "CANARY-abc123"}
        outputs = {"ERR_001": "Here is my code CANARY-abc123 done"}
        leaked = detector.check_canary_leakage(canary_map, outputs)
        assert len(leaked) == 1
        assert leaked[0]["severity"] == "critical"

    def test_no_leakage(self):
        detector = ContaminationDetector()
        canary_map = {"ERR_001": "CANARY-abc123"}
        outputs = {"ERR_001": "Here is clean code without any canary"}
        leaked = detector.check_canary_leakage(canary_map, outputs)
        assert len(leaked) == 0


class TestCorpusOverlap:
    def test_no_overlap(self):
        detector = ContaminationDetector(ngram_size=3, jaccard_threshold=0.3)
        tasks = [
            _make_task("ERR_001", "Unique benchmark task",
                       "This is a completely original task about quantum computing evaluation",
                       "Write a quantum circuit simulator function"),
        ]
        corpus = ["The cat sat on the mat and looked around the room"]
        flagged = detector.check_against_corpus(tasks, corpus)
        assert len(flagged) == 0


class TestFullCheck:
    def test_clean_report(self):
        detector = ContaminationDetector(ngram_size=3, jaccard_threshold=0.5)
        tasks = [
            _make_task("ERR_001", "Task one about errors",
                       "First task description about handling exceptions in code",
                       "Write a safe file reader"),
            _make_task("ERR_002", "Task two about networks",
                       "Second task description about network request handling",
                       "Write an HTTP client with retries"),
        ]
        report = detector.run_full_check(tasks)
        assert report.num_tasks_checked == 2
        assert "clean" in report.summary

    def test_report_to_dict(self):
        detector = ContaminationDetector()
        tasks = [_make_task("ERR_001", "Test", "D" * 50, "P" * 50)]
        report = detector.run_full_check(tasks)
        d = report.to_dict()
        assert "num_tasks_checked" in d
        assert "contamination_rate" in d
