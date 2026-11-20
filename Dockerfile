FROM python:3.11-slim

LABEL maintainer="Ashish Vipin Korumelli"
LABEL description="CodeAgent-Bench: Sandboxed coding agent evaluation environment"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for sandboxed execution
RUN useradd --create-home --shell /bin/bash benchrunner
WORKDIR /home/benchrunner/codeagent-bench

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .
RUN pip install --no-cache-dir -e .

# Resource limits for agent runs
ENV BENCH_TIMEOUT=120
ENV BENCH_MAX_MEMORY=512m
ENV PYTHONUNBUFFERED=1

USER benchrunner

ENTRYPOINT ["python", "scripts/run_benchmark.py"]
CMD ["--help"]
