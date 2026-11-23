# CodeAgent-Bench 

**A reproducible benchmark framework for evaluating coding agents on real-world failure cases.**

CodeAgent-Bench systematically captures, categorizes, and grades how AI coding agents fail — turning observed failure patterns into objectively scored evaluation tasks with contamination safeguards and statistical rigor.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](Dockerfile)

---

## Why CodeAgent-Bench?

Existing benchmarks (HumanEval, MBPP) test *whether* an agent can write code. They don't test *how agents fail in practice*. We observed that coding agents exhibit **systematic, repeatable failure patterns** — incomplete error handling, incorrect API usage, edge-case blindness, dependency confusion — that existing benchmarks miss entirely.

CodeAgent-Bench fills this gap:

1. **Failure-Driven Task Design:** Every task originates from a real, observed agent failure pattern, categorized into a failure taxonomy.
2. **Objective Grading:** Each task includes deterministic test suites — no model-as-judge subjectivity. Tasks are graded `pass`, `partial`, or `fail` with explicit rubrics.
3. **Contamination Guardrails:** Built-in n-gram overlap detection and embedding similarity checks prevent answer leakage from training data.
4. **Statistical Rigor:** Bootstrap confidence intervals, paired significance tests, and difficulty calibration ensure benchmarks measure genuine capability — not noise.
5. **Reproducible Execution:** Docker-based sandboxed agent runs ensure bit-for-bit reproducibility across environments. Works with any chat-completions-compatible endpoint (vLLM, TGI, llama.cpp, or custom HTTP servers).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       CodeAgent-Bench                           │
├─────────────┬──────────────┬──────────────┬─────────────────────┤
│ Task Loader │  Agent Runner│   Scorer     │   Reporter          │
│             │  (Docker)    │              │                     │
│ • JSONL     │ • Sandboxed  │ • pass@k     │ • Per-category      │
│ • Schema    │ • Timeout    │ • Partial    │ • Stat. summary     │
│   validation│ • Resource   │   credit     │ • Contamination     │
│ • Failure   │   limits     │ • Test-suite │   report            │
│   taxonomy  │ • Multi-agent│   grading    │ • Difficulty curve  │
│   tagging   │   support    │              │                     │
├─────────────┴──────────────┴──────────────┴─────────────────────┤
│              Statistics & Contamination Engine                   │
│  • Bootstrap CI  • McNemar's test  • N-gram dedup               │
│  • Difficulty calibration  • Train/test separation              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Failure Taxonomy

CodeAgent-Bench organizes tasks into a **5-category failure taxonomy** derived from analysis of 500+ real agent runs:

| Category | Code | Description | Example |
|---|---|---|---|
| **Incomplete Error Handling** | `ERR` | Agent produces code that works on the happy path but crashes on edge inputs | Missing `try/except` for file I/O, no null checks |
| **Incorrect API Usage** | `API` | Agent calls functions with wrong signatures, deprecated methods, or incorrect argument types | `os.path.join()` misuse, wrong `pandas` axis parameter |
| **Logic & Algorithm Errors** | `LOGIC` | Functionally incorrect logic: off-by-one, wrong operator, flawed algorithm | Sorting comparator returning wrong sign, boundary errors |
| **Edge Case Blindness** | `EDGE` | Agent ignores boundary conditions: empty inputs, unicode, concurrency, large-N | Empty list, negative index, unicode filenames |
| **Dependency & Environment** | `DEP` | Agent assumes packages, paths, or OS features that may not exist | Hardcoded paths, missing imports, version conflicts |

---

## Quick Start

### Installation

```bash
git clone https://github.com/AshishVipin/codeagent-bench.git
cd codeagent-bench
pip install -e .
```

### Run the Benchmark

```bash
# Run all tasks against a coding agent
python scripts/run_benchmark.py \
    --tasks benchmarks/tasks/ \
    --agent completions \
    --model my-model \
    --output results/run_001.jsonl

# Analyze results with statistical summary
python scripts/analyze_results.py \
    --results results/run_001.jsonl \
    --report results/report_001.md

# Validate task quality
python scripts/validate_tasks.py \
    --tasks benchmarks/tasks/
```

### Run with Docker (Reproducible)

```bash
docker compose up --build
```

---

## Task Schema

Each task is a self-contained JSONL record:

```json
{
    "task_id": "ERR_014",
    "category": "incomplete_error_handling",
    "difficulty": "medium",
    "title": "File processor missing encoding handler",
    "description": "Write a function that reads a CSV file and returns the sum of a numeric column. Handle all edge cases.",
    "prompt": "Write a Python function `sum_column(filepath: str, column: str) -> float` ...",
    "setup": {
        "files": {"data.csv": "name,value\nalice,10\nbob,twenty\ncharlie,30"},
        "dependencies": ["pandas"]
    },
    "test_suite": [
        {"input": {"filepath": "data.csv", "column": "value"}, "expected": 40.0, "weight": 0.3},
        {"input": {"filepath": "missing.csv", "column": "value"}, "expected_exception": "FileNotFoundError", "weight": 0.3},
        {"input": {"filepath": "data.csv", "column": "nonexistent"}, "expected_exception": "KeyError", "weight": 0.2},
        {"input": {"filepath": "empty.csv", "column": "value"}, "expected": 0.0, "weight": 0.2}
    ],
    "grading": "weighted_test_pass",
    "metadata": {
        "source": "observed_agent_failure",
        "failure_pattern": "Agent produces code that works for valid CSV but crashes on malformed data, missing files, or missing columns",
        "created_at": "2026-11-01",
        "author": "ashish-vipin"
    }
}
```

---

## Scoring

CodeAgent-Bench supports multiple scoring strategies:

| Metric | Description |
|---|---|
| **pass@k** | Probability of at least one correct solution in k samples (unbiased estimator) |
| **weighted_test_pass** | Weighted sum of individual test case pass rates |
| **partial_credit** | Graduated scoring: full pass, partial (some tests pass), or fail |
| **category_breakdown** | Aggregate scores per failure category |

---

## Contamination Detection

Before any benchmark release, CodeAgent-Bench runs automatic contamination checks:

- **N-gram overlap:** 8-gram Jaccard similarity between task prompts and known training corpora (The Stack, GitHub Code).
- **Embedding similarity:** Cosine similarity of task embeddings against a reference set, flagging tasks above a configurable threshold.
- **Train/test separation audit:** Ensures no task prompt or solution appears in publicly available fine-tuning datasets.
- **Canary string injection:** Hidden canary tokens in task descriptions to detect if a model has memorized the benchmark.

---

## Statistical Analysis

Every benchmark run produces rigorous statistical summaries:

- **Bootstrap confidence intervals** (95% CI) for all aggregate scores.
- **McNemar's test** for paired model comparisons (is Model A significantly better than Model B on this benchmark?).
- **Difficulty calibration curves** plotting human solve-rate vs. agent solve-rate.
- **Inter-annotator agreement** (Cohen's κ) for task quality validation.

---

## Project Structure

```
codeagent-bench/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── configs/
│   └── default.yaml            # Default benchmark configuration
├── benchmarks/
│   ├── tasks/                  # JSONL task files by category
│   │   ├── error_handling.jsonl
│   │   ├── api_usage.jsonl
│   │   ├── logic_errors.jsonl
│   │   ├── edge_cases.jsonl
│   │   └── dependency_env.jsonl
│   └── golden/
│       └── golden_solutions.jsonl
├── codeagent_bench/
│   ├── __init__.py
│   ├── task_loader.py          # Load, validate, and filter JSONL tasks
│   ├── runner.py               # Orchestrate benchmark runs
│   ├── agent_wrapper.py        # Abstract interface for coding agents
│   ├── docker_runner.py        # Docker-based sandboxed execution
│   ├── scorer.py               # Scoring and grading logic
│   ├── statistics.py           # Statistical analysis utilities
│   ├── contamination.py        # Contamination detection engine
│   ├── reporter.py             # Generate markdown/JSON reports
│   └── utils.py                # Shared utilities
├── scripts/
│   ├── run_benchmark.py        # CLI entry point for running benchmarks
│   ├── analyze_results.py      # CLI for analyzing results
│   └── validate_tasks.py       # CLI for task validation
├── tests/
│   ├── test_task_loader.py
│   ├── test_scorer.py
│   ├── test_statistics.py
│   └── test_contamination.py
└── results/
    └── .gitkeep
```

---

## Contributing

We welcome contributions! See our task authoring guidelines:

1. **Identify a real failure:** Run a coding agent, observe where it fails.
2. **Categorize:** Map the failure to our taxonomy (ERR, API, LOGIC, EDGE, DEP).
3. **Write deterministic tests:** Every task must have objective pass/fail criteria — no subjective grading.
4. **Validate:** Run `python scripts/validate_tasks.py` to check schema compliance.
5. **Check contamination:** Ensure your task isn't trivially searchable on the web.

---

## Citation

```bibtex
@software{korumelli2026codeagentbench,
  author = {Korumelli, Ashish Vipin},
  title = {CodeAgent-Bench: A Failure-Driven Benchmark for Coding Agent Evaluation},
  year = {2026},
  url = {https://github.com/AshishVipin/codeagent-bench}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
