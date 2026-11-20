"""
Task Loader: Load, validate, and filter JSONL benchmark tasks.

Handles schema validation, failure taxonomy tagging, and task filtering
by category, difficulty, or custom predicates.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Task schema (JSON Schema draft-07) used to validate every JSONL record
# ------------------------------------------------------------------

TASK_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "task_id",
        "category",
        "difficulty",
        "title",
        "description",
        "prompt",
        "test_suite",
        "grading",
    ],
    "properties": {
        "task_id": {
            "type": "string",
            "pattern": "^[A-Z]{2,5}_\\d{3,4}$",
            "description": "Unique task identifier: CATEGORY_NNN",
        },
        "category": {
            "type": "string",
            "enum": [
                "incomplete_error_handling",
                "incorrect_api_usage",
                "logic_errors",
                "edge_cases",
                "dependency_env",
            ],
        },
        "difficulty": {
            "type": "string",
            "enum": ["easy", "medium", "hard"],
        },
        "title": {"type": "string", "minLength": 5},
        "description": {"type": "string", "minLength": 20},
        "prompt": {"type": "string", "minLength": 20},
        "setup": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "setup_code": {"type": "string"},
            },
        },
        "test_suite": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["weight"],
                "properties": {
                    "input": {"type": "object"},
                    "expected": {},
                    "expected_exception": {"type": "string"},
                    "weight": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "description": {"type": "string"},
                },
            },
        },
        "grading": {
            "type": "string",
            "enum": ["weighted_test_pass", "exact_match", "partial_credit"],
        },
        "metadata": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "failure_pattern": {"type": "string"},
                "created_at": {"type": "string"},
                "author": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    },
    "additionalProperties": False,
}

# Canonical failure taxonomy mapping
FAILURE_TAXONOMY: dict[str, str] = {
    "incomplete_error_handling": "ERR",
    "incorrect_api_usage": "API",
    "logic_errors": "LOGIC",
    "edge_cases": "EDGE",
    "dependency_env": "DEP",
}


@dataclass
class Task:
    """A single benchmark evaluation task."""

    task_id: str
    category: str
    difficulty: str
    title: str
    description: str
    prompt: str
    test_suite: list[dict[str, Any]]
    grading: str
    setup: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def category_code(self) -> str:
        """Return the short taxonomy code (ERR, API, LOGIC, EDGE, DEP)."""
        return FAILURE_TAXONOMY.get(self.category, "UNK")

    @property
    def num_tests(self) -> int:
        return len(self.test_suite)

    @property
    def total_weight(self) -> float:
        return sum(t.get("weight", 0) for t in self.test_suite)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "title": self.title,
            "description": self.description,
            "prompt": self.prompt,
            "test_suite": self.test_suite,
            "grading": self.grading,
            "setup": self.setup,
            "metadata": self.metadata,
        }


class TaskLoader:
    """Load, validate, and filter JSONL benchmark tasks.

    Parameters
    ----------
    schema : dict | None
        Custom JSON Schema to validate tasks against.  Defaults to
        ``TASK_SCHEMA``.
    strict : bool
        If True, raise on the first validation error.  If False, skip
        invalid tasks and log warnings.
    """

    def __init__(
        self,
        schema: dict[str, Any] | None = None,
        strict: bool = False,
    ) -> None:
        self.schema = schema or TASK_SCHEMA
        self.strict = strict
        self._validator = jsonschema.Draft7Validator(self.schema)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_file(self, path: str | Path) -> list[Task]:
        """Load tasks from a single JSONL file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Task file not found: {path}")

        tasks: list[Task] = []
        with open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    msg = f"{path}:{line_no} — invalid JSON: {exc}"
                    if self.strict:
                        raise ValueError(msg) from exc
                    logger.warning(msg)
                    continue

                task = self._validate_and_build(record, source=f"{path}:{line_no}")
                if task is not None:
                    tasks.append(task)

        logger.info("Loaded %d tasks from %s", len(tasks), path)
        return tasks

    def load_directory(self, directory: str | Path) -> list[Task]:
        """Load all ``*.jsonl`` files from *directory* (non-recursive)."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        all_tasks: list[Task] = []
        for jsonl_path in sorted(directory.glob("*.jsonl")):
            all_tasks.extend(self.load_file(jsonl_path))

        logger.info(
            "Loaded %d tasks total from %s (%d files)",
            len(all_tasks),
            directory,
            len(list(directory.glob("*.jsonl"))),
        )
        return all_tasks

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @staticmethod
    def filter_by_category(
        tasks: list[Task],
        categories: list[str],
    ) -> list[Task]:
        """Return tasks whose category is in *categories*."""
        cat_set = set(categories)
        return [t for t in tasks if t.category in cat_set]

    @staticmethod
    def filter_by_difficulty(
        tasks: list[Task],
        difficulties: list[str],
    ) -> list[Task]:
        diff_set = set(difficulties)
        return [t for t in tasks if t.difficulty in diff_set]

    @staticmethod
    def filter_by_ids(
        tasks: list[Task],
        task_ids: list[str],
    ) -> list[Task]:
        id_set = set(task_ids)
        return [t for t in tasks if t.task_id in id_set]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate(self, record: dict[str, Any]) -> list[str]:
        """Return a list of validation error messages (empty = valid)."""
        errors: list[str] = []
        for err in self._validator.iter_errors(record):
            errors.append(f"{err.json_path}: {err.message}")
        # Additional semantic checks
        errors.extend(self._semantic_checks(record))
        return errors

    def _validate_and_build(
        self,
        record: dict[str, Any],
        source: str = "",
    ) -> Task | None:
        errors = self.validate(record)
        if errors:
            msg = f"Validation errors in {source}:\n" + "\n".join(f"  • {e}" for e in errors)
            if self.strict:
                raise ValueError(msg)
            logger.warning(msg)
            return None
        return self._build_task(record)

    @staticmethod
    def _semantic_checks(record: dict[str, Any]) -> list[str]:
        """Additional checks beyond JSON Schema."""
        errors: list[str] = []

        # Weights should sum to ~1.0
        test_suite = record.get("test_suite", [])
        total = sum(t.get("weight", 0) for t in test_suite)
        if test_suite and not (0.99 <= total <= 1.01):
            errors.append(
                f"Test suite weights sum to {total:.3f}, expected ~1.0"
            )

        # Each test must have either expected or expected_exception
        for i, tc in enumerate(test_suite):
            if "expected" not in tc and "expected_exception" not in tc:
                errors.append(
                    f"test_suite[{i}]: must have 'expected' or 'expected_exception'"
                )

        # task_id prefix should match category code
        task_id = record.get("task_id", "")
        category = record.get("category", "")
        expected_prefix = FAILURE_TAXONOMY.get(category, "")
        if expected_prefix and not task_id.startswith(expected_prefix):
            errors.append(
                f"task_id '{task_id}' prefix doesn't match category "
                f"'{category}' (expected prefix '{expected_prefix}')"
            )

        return errors

    @staticmethod
    def _build_task(record: dict[str, Any]) -> Task:
        return Task(
            task_id=record["task_id"],
            category=record["category"],
            difficulty=record["difficulty"],
            title=record["title"],
            description=record["description"],
            prompt=record["prompt"],
            test_suite=record["test_suite"],
            grading=record["grading"],
            setup=record.get("setup", {}),
            metadata=record.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def summary(tasks: list[Task]) -> dict[str, Any]:
        """Return a summary dict for a list of tasks."""
        from collections import Counter

        cats = Counter(t.category for t in tasks)
        diffs = Counter(t.difficulty for t in tasks)
        return {
            "total_tasks": len(tasks),
            "by_category": dict(cats),
            "by_difficulty": dict(diffs),
            "task_ids": [t.task_id for t in tasks],
        }
