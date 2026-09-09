# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""Benchmark candidate Ollama models against Aptly's three LLM call shapes.

This project targets CPU-only, GPU-free hardware by design (see
docs/MODEL_SELECTION.md for the full rationale). This script is how that
choice is backed by numbers instead of a guess: it runs the same three
real prompts this project actually sends to an LLM (extract_requirements_prompt,
extract_resume_chunks_prompt, judge_match_prompt) against each candidate
model, and records latency, throughput, and first-attempt schema validity
for each.

Uses generic, fictional sample data (a made-up JD and resume snippet) rather
than any real user's data, so this script is portable and safe to run as-is
by anyone who clones the repo — it does not read `data/resume_chunks/` or
depend on your local content, and the results describe the *models'*
behavior, not any real person's resume/JD analysis.

Run standalone:

    python -m aptly.eval.model_benchmark

Requires Ollama running locally with every model in `MODELS_TO_BENCHMARK`
already pulled (`ollama pull <model>` for each). Writes a summary table to
stdout and the full results to `docs/model_benchmark_results.json`.
"""

import json
import time
from dataclasses import asdict, dataclass

import requests
from pydantic import BaseModel, ValidationError

from aptly import config
from aptly.api.routes_resume import _clean_resume_text
from aptly.llm.prompts import extract_requirements_prompt, extract_resume_chunks_prompt, judge_match_prompt
from aptly.llm.schemas import ExtractedRequirements, ExtractedResumeChunks, MatchJudgment

#: Models to benchmark. Each must already be pulled locally (`ollama pull <name>`)
#: before running this script — pulling is not done automatically, since it's a
#: multi-GB download best done deliberately, not as a side effect of a benchmark run.
MODELS_TO_BENCHMARK = ["llama3.2:3b", "qwen2.5:3b", "phi3:mini"]

#: A fictional job description, sized and structured like a real one, used for the
#: requirement-extraction task. Deliberately not any real JD.
SAMPLE_JD = """
We're hiring a Backend Engineer to help build our data platform.

Requirements:
- 3+ years of experience building ETL pipelines in Python at scale
- Experience deploying containerized REST APIs on AWS
- Familiarity with vector databases and retrieval-augmented generation (RAG)
- Hands-on experience with Kubernetes for production container orchestration
- Experience with Terraform or another infrastructure-as-code tool
"""

#: A fictional resume snippet, used for the resume-chunking task. Deliberately
#: not any real person's resume — mirrors the kind of noisy, layout-flattened
#: text a real PDF extraction produces (see aptly.api.routes_resume._clean_resume_text).
SAMPLE_RESUME_TEXT = """
Jordan Rivera
Backend Engineer

Results-driven backend engineer with 4 years of experience building
data-intensive services.

CONTACT
jordan.rivera@example.com
https://linkedin.com/in/jordanrivera

TECHNICAL SKILLS
Python
Go
PostgreSQL
Docker
Kubernetes

EXPERIENCE
Backend Engineer
Nimbus Data · March 2022 - Present
Built a real-time event processing service in Go, handling 2M+ events/day
with sub-100ms p99 latency.
Migrated a monolithic Django application to a set of containerized
microservices, reducing deployment time from 45 minutes to 4 minutes.
Designed a PostgreSQL sharding strategy that cut query latency by 60% for
the platform's largest customer accounts.

Software Engineer
Fieldstone Labs · June 2020 - February 2022
Built ETL pipelines in Python processing 5M+ records/day from third-party
APIs into a data warehouse, with schema validation and automatic retries.
Implemented a caching layer using Redis that reduced database load by 70%
during peak traffic.

EDUCATION
B.S. Computer Science
State University
"""

#: A fictional requirement/resume-chunk pair for the match-judgment task —
#: deliberately chosen to be a genuinely borderline case (related but not a
#: clean match), which is the only case that ever reaches this prompt in
#: production (see aptly.retrieval.match.match_requirements).
SAMPLE_REQUIREMENT = "Experience with vector databases and retrieval-augmented generation (RAG)"
SAMPLE_CHUNK_TEXT = (
    "Built a real-time event processing service in Go, handling 2M+ events/day "
    "with sub-100ms p99 latency, using a Redis-backed cache for fast lookups."
)


@dataclass
class BenchmarkResult:
    """One (model, task) benchmark measurement.

    Attributes:
        model: The Ollama model tag benchmarked, e.g. "llama3.2:3b".
        task: Which of the three prompt shapes was tested — "requirement_extraction",
            "resume_chunking", or "match_judgment".
        wall_clock_seconds: End-to-end time for the HTTP call, measured from
            this script, including any model-load time if the model wasn't
            already resident in memory.
        total_duration_seconds: Ollama's own reported total duration for the
            request (model load + prompt eval + generation), from the
            response's `total_duration` field (nanoseconds, converted here).
        prompt_eval_count: Number of tokens Ollama counted in the prompt.
        eval_count: Number of tokens generated in the response.
        tokens_per_second: `eval_count / eval_duration`, i.e. generation
            throughput once prompt processing is done — the number to
            compare across models/hardware, since it excludes prompt-length
            differences and one-time model-load overhead.
        schema_valid: Whether the raw output parsed as JSON and validated
            against the expected schema on the *first* attempt, with no
            retry — this is what `aptly.llm.client.call`'s retry logic
            exists to paper over in production; here it's measured directly
            so the retry rate itself is visible as a quality signal.
        error: `None` on success; a short description of what went wrong
            otherwise (e.g. an HTTP error, or a schema validation failure
            message) — kept short since it can also be printed in the
            summary table.
    """

    model: str
    task: str
    wall_clock_seconds: float
    total_duration_seconds: float | None
    prompt_eval_count: int | None
    eval_count: int | None
    tokens_per_second: float | None
    schema_valid: bool
    error: str | None


def _call_model(model: str, prompt: str, schema: type[BaseModel]) -> BenchmarkResult:
    """Send one prompt to one model and measure the result — no retry.

    Internal helper — not part of the module's public interface.
    Deliberately bypasses `aptly.llm.client.call` (which retries once on
    failure and always targets `config.OLLAMA_MODEL`) since a benchmark
    needs the raw, single-attempt result for a specific model, not the
    production-hardened, always-succeeds-if-possible behavior.

    Args:
        model: The Ollama model tag to call.
        prompt: The full prompt text.
        schema: The Pydantic schema the output is expected to validate against.

    Returns:
        A `BenchmarkResult` for this one call. Never raises — any failure
        (HTTP error, invalid JSON, schema mismatch) is captured in the
        result's `error` field instead, so one bad model/task doesn't stop
        the rest of the benchmark.
    """
    start = time.time()
    try:
        response = requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": config.OLLAMA_NUM_CTX,
                    "num_predict": config.OLLAMA_NUM_PREDICT,
                },
            },
            timeout=config.OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        wall_clock = time.time() - start
        data = response.json()

        schema_valid = True
        error = None
        try:
            schema.model_validate(json.loads(data["response"]))
        except (json.JSONDecodeError, ValidationError) as e:
            schema_valid = False
            error = f"schema validation failed: {e}"[:200]

        eval_count = data.get("eval_count")
        eval_duration_ns = data.get("eval_duration")
        tokens_per_second = (
            eval_count / (eval_duration_ns / 1e9) if eval_count and eval_duration_ns else None
        )

        return BenchmarkResult(
            model=model,
            task="",  # filled in by the caller
            wall_clock_seconds=round(wall_clock, 2),
            total_duration_seconds=round(data["total_duration"] / 1e9, 2) if "total_duration" in data else None,
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=eval_count,
            tokens_per_second=round(tokens_per_second, 1) if tokens_per_second else None,
            schema_valid=schema_valid,
            error=error,
        )
    except requests.RequestException as e:
        return BenchmarkResult(
            model=model,
            task="",
            wall_clock_seconds=round(time.time() - start, 2),
            total_duration_seconds=None,
            prompt_eval_count=None,
            eval_count=None,
            tokens_per_second=None,
            schema_valid=False,
            error=f"request failed: {e}"[:200],
        )


def run_benchmark() -> list[BenchmarkResult]:
    """Run all three tasks against every model in `MODELS_TO_BENCHMARK`.

    Returns:
        A flat list of `BenchmarkResult`, one per (model, task) pair, in
        the order the calls were made — models outer loop, tasks inner
        loop, so all of one model's results are contiguous.
    """
    tasks = [
        ("requirement_extraction", extract_requirements_prompt(SAMPLE_JD), ExtractedRequirements),
        (
            "resume_chunking",
            extract_resume_chunks_prompt(_clean_resume_text(SAMPLE_RESUME_TEXT)),
            ExtractedResumeChunks,
        ),
        ("match_judgment", judge_match_prompt(SAMPLE_REQUIREMENT, SAMPLE_CHUNK_TEXT), MatchJudgment),
    ]

    results = []
    for model in MODELS_TO_BENCHMARK:
        for task_name, prompt, schema in tasks:
            print(f"Running {model} / {task_name}...", flush=True)
            result = _call_model(model, prompt, schema)
            result.task = task_name
            results.append(result)
            print(f"  -> {result.wall_clock_seconds}s, valid={result.schema_valid}, error={result.error}")
    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a plain-text summary table of benchmark results to stdout.

    Args:
        results: The full result list from `run_benchmark`.

    Returns:
        None. Output only.
    """
    print(f"\n{'Model':<16} {'Task':<24} {'Wall(s)':<10} {'Tok/s':<8} {'Valid?':<8} Error")
    for r in results:
        print(
            f"{r.model:<16} {r.task:<24} {r.wall_clock_seconds:<10} "
            f"{r.tokens_per_second if r.tokens_per_second else '-':<8} {r.schema_valid!s:<8} {r.error or ''}"
        )


def main() -> None:
    """Run the full benchmark, print a summary, and save results as JSON.

    Returns:
        None. Writes `docs/model_benchmark_results.json` as a side effect.
    """
    results = run_benchmark()
    print_summary(results)

    out_path = config.BASE_DIR / "docs" / "model_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
