"""Tests for TaskLoader: schema validation, loading, and filtering."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from codeagent_bench.task_loader import TASK_SCHEMA, Task, TaskLoader


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

VALID_TASK = {
    "task_id": "ERR_001",
    "category": "incomplete_error_handling",
    "difficulty": "easy",
    "title": "File reader missing error handler",
    "description": "Agent produces a file reading function that works but crashes on missing files.",
    "prompt": "Write a function read_file_lines(filepath: str) -> list[str] that reads a file.",
    "test_suite": [
        {"input": {"filepath": "test.txt"}, "expected": ["hello"], "weight": 0.5},
        {"input": {"filepath": "missing.txt"}, "expected": [], "weight": 0.5},
    ],
    "grading": "weighted_test_pass",
    "metadata": {
        "source": "observed_agent_failure",
        "author": "test",
        "created_at": "2026-01-01",
    },
}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------
# Schema validation tests
# ------------------------------------------------------------------


class TestTaskValidation:
    def test_valid_task_passes(self):
        loader = TaskLoader(strict=True)
        errors = loader.validate(VALID_TASK)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_missing_required_field(self):
        loader = TaskLoader()
        invalid = {k: v for k, v in VALID_TASK.items() if k != "task_id"}
        errors = loader.validate(invalid)
        assert any("task_id" in e for e in errors)

    def test_invalid_category(self):
        loader = TaskLoader()
        invalid = {**VALID_TASK, "category": "nonexistent_category"}
        errors = loader.validate(invalid)
        assert len(errors) > 0

    def test_invalid_difficulty(self):
        loader = TaskLoader()
        invalid = {**VALID_TASK, "difficulty": "impossible"}
        errors = loader.validate(invalid)
        assert len(errors) > 0

    def test_weight_sum_validation(self):
        loader = TaskLoader()
        invalid = {
            **VALID_TASK,
            "test_suite": [
                {"input": {}, "expected": 1, "weight": 0.3},
                {"input": {}, "expected": 2, "weight": 0.3},
            ],
        }
        errors = loader.validate(invalid)
        assert any("weights sum" in e for e in errors)

    def test_task_id_prefix_mismatch(self):
        loader = TaskLoader()
        invalid = {**VALID_TASK, "task_id": "API_001"}  # ERR category but API prefix
        errors = loader.validate(invalid)
        assert any("prefix" in e for e in errors)

    def test_test_case_missing_expected(self):
        loader = TaskLoader()
        invalid = {
            **VALID_TASK,
            "test_suite": [
                {"input": {}, "weight": 1.0},  # No expected or expected_exception
            ],
        }
        errors = loader.validate(invalid)
        assert any("expected" in e.lower() for e in errors)


# ------------------------------------------------------------------
# Loading tests
# ------------------------------------------------------------------


class TestTaskLoading:
    def test_load_single_file(self, tmp_path: Path):
        path = tmp_path / "tasks.jsonl"
        _write_jsonl(path, [VALID_TASK])
        loader = TaskLoader()
        tasks = loader.load_file(path)
        assert len(tasks) == 1
        assert tasks[0].task_id == "ERR_001"

    def test_load_directory(self, tmp_path: Path):
        _write_jsonl(tmp_path / "a.jsonl", [VALID_TASK])
        task2 = {**VALID_TASK, "task_id": "ERR_002"}
        _write_jsonl(tmp_path / "b.jsonl", [task2])
        loader = TaskLoader()
        tasks = loader.load_directory(tmp_path)
        assert len(tasks) == 2

    def test_skip_invalid_in_non_strict(self, tmp_path: Path):
        path = tmp_path / "tasks.jsonl"
        invalid = {**VALID_TASK, "category": "bad"}
        _write_jsonl(path, [VALID_TASK, invalid])
        loader = TaskLoader(strict=False)
        tasks = loader.load_file(path)
        assert len(tasks) == 1

    def test_strict_raises_on_invalid(self, tmp_path: Path):
        path = tmp_path / "tasks.jsonl"
        invalid = {**VALID_TASK, "category": "bad"}
        _write_jsonl(path, [invalid])
        loader = TaskLoader(strict=True)
        with pytest.raises(ValueError):
            loader.load_file(path)

    def test_file_not_found(self):
        loader = TaskLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/path.jsonl")


# ------------------------------------------------------------------
# Filtering tests
# ------------------------------------------------------------------


class TestTaskFiltering:
    def _make_tasks(self) -> list[Task]:
        return [
            Task("ERR_001", "incomplete_error_handling", "easy", "t1", "d" * 20, "p" * 20, [{"weight": 1.0, "expected": 1, "input": {}}], "weighted_test_pass"),
            Task("API_001", "incorrect_api_usage", "medium", "t2", "d" * 20, "p" * 20, [{"weight": 1.0, "expected": 1, "input": {}}], "weighted_test_pass"),
            Task("LOGIC_001", "logic_errors", "hard", "t3", "d" * 20, "p" * 20, [{"weight": 1.0, "expected": 1, "input": {}}], "weighted_test_pass"),
        ]

    def test_filter_by_category(self):
        tasks = self._make_tasks()
        result = TaskLoader.filter_by_category(tasks, ["incomplete_error_handling"])
        assert len(result) == 1
        assert result[0].task_id == "ERR_001"

    def test_filter_by_difficulty(self):
        tasks = self._make_tasks()
        result = TaskLoader.filter_by_difficulty(tasks, ["hard"])
        assert len(result) == 1
        assert result[0].task_id == "LOGIC_001"

    def test_filter_by_ids(self):
        tasks = self._make_tasks()
        result = TaskLoader.filter_by_ids(tasks, ["API_001", "LOGIC_001"])
        assert len(result) == 2

    def test_summary(self):
        tasks = self._make_tasks()
        s = TaskLoader.summary(tasks)
        assert s["total_tasks"] == 3
        assert s["by_difficulty"]["easy"] == 1


# ------------------------------------------------------------------
# Task dataclass tests
# ------------------------------------------------------------------


class TestTaskDataclass:
    def test_category_code(self):
        task = Task("ERR_001", "incomplete_error_handling", "easy", "t", "d" * 20, "p" * 20, [], "weighted_test_pass")
        assert task.category_code == "ERR"

    def test_num_tests(self):
        task = Task("ERR_001", "incomplete_error_handling", "easy", "t", "d" * 20, "p" * 20,
                     [{"weight": 0.5}, {"weight": 0.5}], "weighted_test_pass")
        assert task.num_tests == 2

    def test_total_weight(self):
        task = Task("ERR_001", "incomplete_error_handling", "easy", "t", "d" * 20, "p" * 20,
                     [{"weight": 0.6}, {"weight": 0.4}], "weighted_test_pass")
        assert abs(task.total_weight - 1.0) < 0.01

    def test_to_dict_roundtrip(self):
        task = Task("ERR_001", "incomplete_error_handling", "easy", "Title",
                     "D" * 20, "P" * 20, [{"weight": 1.0}], "weighted_test_pass")
        d = task.to_dict()
        assert d["task_id"] == "ERR_001"
        assert d["category"] == "incomplete_error_handling"
