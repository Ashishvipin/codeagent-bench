"""
Agent Wrapper: Abstract interface for coding agents.

Provides a uniform interface to query different coding agents (API-based
providers, local models, or mock agents) and collect their code outputs
for benchmark evaluation.
"""

from __future__ import annotations

import json
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from codeagent_bench.task_loader import Task

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from a coding agent."""

    task_id: str
    code_output: str
    raw_response: str
    model: str
    latency_s: float
    tokens_used: int = 0
    error: str | None = None


class BaseAgent(ABC):
    """Abstract base class for coding agents."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.config = kwargs

    @abstractmethod
    def generate(self, task: Task) -> AgentResponse:
        """Generate a code solution for the given task."""
        ...

    def generate_batch(self, tasks: list[Task]) -> list[AgentResponse]:
        """Generate solutions for a batch of tasks (sequential by default)."""
        return [self.generate(task) for task in tasks]

    def _build_system_prompt(self) -> str:
        return (
            "You are a Python coding assistant. Given a task description, "
            "write a Python function that satisfies all requirements. "
            "Output ONLY the Python function code, no explanations, no "
            "markdown fences, no imports unless necessary. Handle all edge "
            "cases described in the task."
        )

    def _build_user_prompt(self, task: Task) -> str:
        prompt = f"Task: {task.title}\n\n{task.prompt}"
        if task.setup.get("dependencies"):
            prompt += (
                f"\n\nAvailable libraries: "
                f"{', '.join(task.setup['dependencies'])}"
            )
        return prompt


class MockAgent(BaseAgent):
    """Mock agent that returns pre-defined or intentionally flawed code.

    Useful for testing the benchmark harness without calling external APIs.
    """

    # Pre-defined responses that demonstrate common failure patterns
    MOCK_RESPONSES: dict[str, str] = {
        "incomplete_error_handling": '''\
def sum_column(filepath, column):
    import pandas as pd
    df = pd.read_csv(filepath)
    return df[column].sum()
''',
        "incorrect_api_usage": '''\
def merge_configs(base, override):
    result = base.copy()
    result.update(override)
    return result
''',
        "logic_errors": '''\
def find_kth_largest(nums, k):
    nums.sort()
    return nums[k]
''',
        "edge_cases": '''\
def safe_divide_list(numbers, divisor):
    return [n / divisor for n in numbers]
''',
        "dependency_env": '''\
def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.load(f)
''',
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model="mock-v1", **kwargs)

    def generate(self, task: Task) -> AgentResponse:
        start = time.monotonic()
        code = self.MOCK_RESPONSES.get(
            task.category,
            "def solution(*args, **kwargs):\n    pass",
        )
        elapsed = time.monotonic() - start

        return AgentResponse(
            task_id=task.task_id,
            code_output=code,
            raw_response=code,
            model=self.model,
            latency_s=elapsed,
        )


class ChatCompletionsAgent(BaseAgent):
    """Agent that queries any chat-completions-compatible API endpoint.

    Works with self-hosted vLLM / TGI / llama.cpp servers, or any
    provider that exposes an ``/v1/chat/completions`` endpoint.
    """

    def __init__(
        self,
        model: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")

    def generate(self, task: Task) -> AgentResponse:
        if not self.api_key:
            return AgentResponse(
                task_id=task.task_id,
                code_output="",
                raw_response="",
                model=self.model,
                latency_s=0,
                error="LLM_API_KEY not set",
            )

        try:
            # We use the `openai` pip package purely as a generic HTTP
            # client for the /v1/chat/completions protocol — the same
            # protocol exposed by vLLM, TGI, llama.cpp, LiteLLM, etc.
            from openai import OpenAI as _CompletionsClient

            client = _CompletionsClient(api_key=self.api_key, base_url=self.base_url)
            start = time.monotonic()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": self._build_user_prompt(task)},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            elapsed = time.monotonic() - start
            raw = response.choices[0].message.content or ""
            code = self._extract_code(raw)
            tokens = response.usage.total_tokens if response.usage else 0

            return AgentResponse(
                task_id=task.task_id,
                code_output=code,
                raw_response=raw,
                model=self.model,
                latency_s=elapsed,
                tokens_used=tokens,
            )

        except Exception as exc:
            logger.error("LLM API error for %s: %s", task.task_id, exc)
            return AgentResponse(
                task_id=task.task_id,
                code_output="",
                raw_response="",
                model=self.model,
                latency_s=0,
                error=str(exc),
            )

    @staticmethod
    def _extract_code(raw: str) -> str:
        """Extract Python code from markdown-fenced responses."""
        if "```python" in raw:
            parts = raw.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()
        if "```" in raw:
            parts = raw.split("```")
            if len(parts) > 1:
                return parts[1].strip()
        return raw.strip()


class HTTPAgent(BaseAgent):
    """Agent that sends prompts to a custom HTTP endpoint.

    Useful for internal model servers, fine-tuned checkpoints, or
    any REST-based inference service.
    """

    def __init__(
        self,
        model: str = "custom-model",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = os.environ.get("HTTP_AGENT_API_KEY", "")
        self.endpoint = os.environ.get("HTTP_AGENT_ENDPOINT", "http://localhost:8080/generate")

    def generate(self, task: Task) -> AgentResponse:
        if not self.endpoint:
            return AgentResponse(
                task_id=task.task_id,
                code_output="",
                raw_response="",
                model=self.model,
                latency_s=0,
                error="HTTP_AGENT_ENDPOINT not set",
            )

        try:
            import urllib.request
            import urllib.error

            payload = json.dumps({
                "model": self.model,
                "prompt": self._build_user_prompt(task),
                "system": self._build_system_prompt(),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }).encode("utf-8")

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            req = urllib.request.Request(
                self.endpoint, data=payload, headers=headers, method="POST"
            )

            start = time.monotonic()
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            elapsed = time.monotonic() - start

            raw = body.get("text", body.get("output", body.get("response", "")))
            code = ChatCompletionsAgent._extract_code(raw)
            tokens = body.get("tokens_used", 0)

            return AgentResponse(
                task_id=task.task_id,
                code_output=code,
                raw_response=raw,
                model=self.model,
                latency_s=elapsed,
                tokens_used=tokens,
            )

        except Exception as exc:
            logger.error("HTTP agent error for %s: %s", task.task_id, exc)
            return AgentResponse(
                task_id=task.task_id,
                code_output="",
                raw_response="",
                model=self.model,
                latency_s=0,
                error=str(exc),
            )


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "mock": MockAgent,
    "completions": ChatCompletionsAgent,
    "http": HTTPAgent,
}


def create_agent(agent_type: str, **kwargs: Any) -> BaseAgent:
    """Create an agent instance by type name."""
    cls = AGENT_REGISTRY.get(agent_type)
    if cls is None:
        raise ValueError(
            f"Unknown agent type '{agent_type}'. "
            f"Available: {list(AGENT_REGISTRY.keys())}"
        )
    return cls(**kwargs)
