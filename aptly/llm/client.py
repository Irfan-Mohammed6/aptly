# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""Ollama JSON-mode client wrapper with schema validation and one retry.

This is the single choke point through which every LLM call in the project
passes — `aptly.retrieval.match.match_requirements` and the API routes never
call `requests` or Ollama directly, they call `call()` here. Centralizing it
means the JSON-parsing, schema validation, and retry-on-malformed-output
logic is written exactly once. See docs/ARCHITECTURE.md §8.

No LangChain, no OpenAI-compatible SDK — just a plain HTTP POST to Ollama's
native `/api/generate` endpoint plus Pydantic. Deliberately minimal: at this
project's scale, an orchestration framework would add a dependency without
earning its abstractions.

Run standalone: not applicable. This is a library module with no
command-line entry point — it's imported wherever an LLM call is needed
(`aptly.retrieval.match`, `aptly.api.routes_jd`, `aptly.api.routes_notes`).
To exercise it manually, use a Python REPL:

    python -c "
    from aptly.llm.client import call
    from aptly.llm.prompts import extract_requirements_prompt
    from aptly.llm.schemas import ExtractedRequirements
    print(call(extract_requirements_prompt('3+ years of Python and AWS experience.'), ExtractedRequirements))
    "

(Requires Ollama running locally with the model in config.OLLAMA_MODEL pulled.)
"""

import json

import requests
from pydantic import BaseModel, ValidationError

from aptly import config


class LLMOutputError(Exception):
    """Raised when the model's output still fails schema validation after one retry.

    Callers (the FastAPI route handlers) are expected to let this propagate
    up into a 502 Bad Gateway response rather than silently falling back to
    some default value — a fit-score analysis built on a guessed-at default
    would be worse than an honest failure.
    """


def _generate(prompt: str) -> str:
    """Send one raw generation request to the local Ollama server.

    Internal helper — not part of the module's public interface. Always
    requests JSON-formatted output (`format: "json"`, Ollama's native
    constrained-decoding mode) and uses a low sampling temperature, since
    every call this module makes is a structured-extraction or
    structured-judgment task rather than open-ended creative generation. At
    default sampling temperatures, the small model this project targets
    (see `config.OLLAMA_MODEL`) tends to improvise around the requested
    schema; low temperature makes it follow the prompt's template far more
    reliably.

    Args:
        prompt: The full prompt text to send, typically produced by one of
            the functions in `aptly.llm.prompts`.

    Returns:
        The raw string from Ollama's `response` field — this is JSON text,
        but is not yet parsed or validated; that happens in `_parse`.

    Raises:
        requests.HTTPError: If Ollama returns a non-2xx response, e.g.
            because the configured model in `config.OLLAMA_MODEL` hasn't been
            pulled, or the Ollama server at `config.OLLAMA_HOST` isn't running.
    """
    response = requests.post(
        f"{config.OLLAMA_HOST}/api/generate",
        json={
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def _parse(raw: str, schema: type[BaseModel]) -> BaseModel:
    """Parse a raw JSON string and validate it against a Pydantic schema.

    Internal helper — not part of the module's public interface.

    Args:
        raw: A JSON-formatted string, typically Ollama's raw response text.
        schema: The Pydantic model class to validate the parsed JSON against.

    Returns:
        An instance of `schema` populated from the parsed JSON.

    Raises:
        json.JSONDecodeError: If `raw` is not valid JSON at all.
        pydantic.ValidationError: If `raw` parses as JSON but doesn't match
            `schema`'s field types/requirements.
    """
    return schema.model_validate(json.loads(raw))


def call(prompt: str, schema: type[BaseModel]) -> BaseModel:
    """Call the local LLM and return its output validated against a schema.

    This is the only function in the project that should be used to talk to
    Ollama. The flow is: generate once, try to parse+validate; if that fails
    (malformed JSON, or JSON that doesn't match `schema`), retry exactly once
    with a corrective follow-up prompt that includes the bad output and the
    validation error; if the retry also fails, raise `LLMOutputError` rather
    than returning anything — there is no silent fallback to a default value.

    This function makes a blocking, synchronous HTTP call that can take
    anywhere from a few seconds to over a minute on CPU-only inference.
    Callers running inside FastAPI's async event loop must not `await` this
    directly — run it via `fastapi.concurrency.run_in_threadpool` instead
    (see `aptly.api.routes_jd.analyze_jd` for the pattern), or it will block
    every other concurrent request.

    Args:
        prompt: The full prompt text, typically from `aptly.llm.prompts`.
        schema: The Pydantic model class the model's JSON output must
            conform to, e.g. `aptly.llm.schemas.ExtractedRequirements` or
            `aptly.llm.schemas.MatchJudgment`.

    Returns:
        An instance of `schema`, populated from the model's (possibly
        corrected-on-retry) output.

    Raises:
        LLMOutputError: If the model's output fails schema validation on
            both the first attempt and the single retry. The exception
            message includes the last raw output for debugging.
        requests.HTTPError: If the underlying HTTP call to Ollama fails
            (e.g. Ollama isn't running, or the model isn't pulled) — this
            propagates from `_generate` unchanged, on either attempt.
    """
    raw = _generate(prompt)
    try:
        return _parse(raw, schema)
    except (json.JSONDecodeError, ValidationError) as first_error:
        correction_prompt = (
            f"{prompt}\n\n"
            f"Your previous response was not valid JSON matching the required schema.\n"
            f"Previous response:\n{raw}\n\n"
            f"Error: {first_error}\n\n"
            f"Return ONLY valid JSON matching the schema, with no extra text."
        )
        raw_retry = _generate(correction_prompt)
        try:
            return _parse(raw_retry, schema)
        except (json.JSONDecodeError, ValidationError) as second_error:
            raise LLMOutputError(
                f"Model output failed validation twice. Last raw output: {raw_retry!r}"
            ) from second_error
