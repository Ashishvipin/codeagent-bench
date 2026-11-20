"""
CodeAgent-Bench: A failure-driven benchmark framework for evaluating coding agents.

Provides tools for loading evaluation tasks, running coding agents in sandboxed
environments, scoring outputs objectively, and producing statistically rigorous
benchmark reports with contamination safeguards.
"""

__version__ = "0.1.0"
__author__ = "Ashish Vipin Korumelli"

from codeagent_bench.task_loader import TaskLoader, Task
from codeagent_bench.scorer import Scorer, TaskResult
from codeagent_bench.runner import BenchmarkRunner
from codeagent_bench.statistics import StatisticalAnalyzer
from codeagent_bench.contamination import ContaminationDetector
from codeagent_bench.reporter import ReportGenerator

__all__ = [
    "TaskLoader",
    "Task",
    "Scorer",
    "TaskResult",
    "BenchmarkRunner",
    "StatisticalAnalyzer",
    "ContaminationDetector",
    "ReportGenerator",
]
