"""
Docker Runner: Sandboxed execution of agent code using Docker containers.

Provides reproducible, resource-limited environments for running
agent-generated code, preventing interference between tasks and
ensuring bit-for-bit reproducibility across different host machines.
"""

from __future__ import annotations

import json
import logging
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DockerRunResult:
    """Result of a Docker-sandboxed code execution."""

    task_id: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    execution_time_s: float
    container_id: str | None = None


class DockerRunner:
    """Execute agent code inside Docker containers with resource limits.

    Parameters
    ----------
    image : str
        Docker image to use for execution.
    timeout : int
        Maximum execution time in seconds per task.
    memory_limit : str
        Container memory limit (e.g., ``"512m"``).
    cpu_limit : float
        CPU core limit (e.g., ``2.0``).
    network_disabled : bool
        Whether to disable networking inside the container.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: int = 120,
        memory_limit: str = "512m",
        cpu_limit: float = 2.0,
        network_disabled: bool = True,
    ) -> None:
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network_disabled = network_disabled
        self._client = None

    @property
    def client(self) -> Any:
        """Lazy-init Docker client."""
        if self._client is None:
            try:
                import docker

                self._client = docker.from_env()
                self._client.ping()
                logger.info("Docker client connected")
            except Exception as exc:
                logger.error("Failed to connect to Docker: %s", exc)
                raise RuntimeError(
                    "Docker is not available. Install Docker and ensure "
                    "the daemon is running, or use --no-docker mode."
                ) from exc
        return self._client

    def run_code(
        self,
        task_id: str,
        code: str,
        test_script: str,
        setup_files: dict[str, str] | None = None,
        dependencies: list[str] | None = None,
    ) -> DockerRunResult:
        """Run agent code and test script inside a Docker container.

        Parameters
        ----------
        task_id : str
            Identifier for logging and result tracking.
        code : str
            The agent-generated Python code.
        test_script : str
            Python test script to validate the agent's code.
        setup_files : dict
            Mapping of filename → content to create in the container.
        dependencies : list
            Python packages to ``pip install`` before running.
        """
        import time

        # Build the entrypoint script
        entrypoint = self._build_entrypoint(
            code=code,
            test_script=test_script,
            dependencies=dependencies or [],
        )

        try:
            start = time.monotonic()

            # Create and run container
            container = self.client.containers.run(
                image=self.image,
                command=["python", "-c", entrypoint],
                detach=True,
                mem_limit=self.memory_limit,
                nano_cpus=int(self.cpu_limit * 1e9),
                network_disabled=self.network_disabled,
                remove=False,
                stdout=True,
                stderr=True,
            )

            # Wait for completion with timeout
            try:
                result = container.wait(timeout=self.timeout)
                exit_code = result.get("StatusCode", -1)
                timed_out = False
            except Exception:
                container.kill()
                exit_code = -1
                timed_out = True

            elapsed = time.monotonic() - start

            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")

            container_id = container.short_id
            container.remove(force=True)

            return DockerRunResult(
                task_id=task_id,
                exit_code=exit_code,
                stdout=stdout[:10000],
                stderr=stderr[:10000],
                timed_out=timed_out,
                execution_time_s=elapsed,
                container_id=container_id,
            )

        except Exception as exc:
            logger.error("Docker execution failed for %s: %s", task_id, exc)
            return DockerRunResult(
                task_id=task_id,
                exit_code=-1,
                stdout="",
                stderr=str(exc)[:5000],
                timed_out=False,
                execution_time_s=0.0,
            )

    @staticmethod
    def _build_entrypoint(
        code: str,
        test_script: str,
        dependencies: list[str],
    ) -> str:
        """Build a self-contained Python entrypoint script."""
        install_block = ""
        if dependencies:
            pkgs = " ".join(dependencies)
            install_block = textwrap.dedent(f"""\
                import subprocess, sys
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", {repr(pkgs)}],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            """)

        return f"""\
{install_block}

# === Agent Code ===
{code}
# === End Agent Code ===

# === Test Script ===
{test_script}
# === End Test Script ===
"""

    def is_available(self) -> bool:
        """Check if Docker is available and responsive."""
        try:
            _ = self.client
            return True
        except Exception:
            return False

    def cleanup(self) -> None:
        """Remove any dangling benchmark containers."""
        try:
            containers = self.client.containers.list(
                filters={"status": "exited"},
                all=True,
            )
            for c in containers:
                if "codeagent-bench" in (c.name or ""):
                    c.remove(force=True)
                    logger.info("Cleaned up container %s", c.short_id)
        except Exception as exc:
            logger.warning("Cleanup failed: %s", exc)
