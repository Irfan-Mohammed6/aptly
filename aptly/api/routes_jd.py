# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""The POST /analyze-jd endpoint: resume/JD fit scoring with evidenced gaps.

See docs/ARCHITECTURE.md §3 for the full call sequence this route implements
(requirement extraction -> per-requirement retrieval -> deterministic
scoring) and the architectural rationale for why it's structured this way.

Run standalone: not applicable — this module only defines a FastAPI
`APIRouter`, it has no entry point of its own. Start the whole application
with `uvicorn aptly.api.main:app --reload` (see aptly/api/main.py), then:

    curl -X POST http://localhost:8000/analyze-jd \\
      -H "Content-Type: application/json" \\
      -d '{"jd_text": "Looking for a backend engineer with 3+ years of Python and AWS experience..."}'
"""

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from aptly.api.models import AnalyzeJDRequest, AnalyzeJDResponse
from aptly.ingestion.resumes import get_resume
from aptly.llm.client import call as llm_call
from aptly.llm.prompts import extract_requirements_prompt
from aptly.llm.schemas import ExtractedRequirements
from aptly.retrieval.match import match_requirements
from aptly.scoring import compute_fit_score

router = APIRouter()


@router.post("/analyze-jd", response_model=AnalyzeJDResponse)
async def analyze_jd(request: AnalyzeJDRequest) -> AnalyzeJDResponse:
    """Score how well the candidate's resume matches a job description.

    Implements the three-stage pipeline from docs/ARCHITECTURE.md §3:

    1. Extract the JD into atomic requirements (one LLM call —
       `extract_requirements_prompt` + `ExtractedRequirements` schema).
    2. Match each requirement against the resume_chunks Chroma collection
       independently (`match_requirements` — see that function's docstring
       for the retrieval-then-threshold-then-optional-LLM-judgment logic).
    3. Compute a deterministic 0-100 fit score from the match verdicts
       (`compute_fit_score`) and split them into `strengths` vs `gaps`.

    Both the LLM extraction call and `match_requirements` (which may itself
    make further blocking LLM calls for borderline cases, and always makes
    blocking Chroma/embedding calls) are dispatched via
    `fastapi.concurrency.run_in_threadpool` rather than awaited directly —
    they are synchronous, potentially slow (tens of seconds on CPU-only
    inference) calls that would otherwise block this process's entire async
    event loop for the duration of the request, stalling every other
    concurrent request.

    Args:
        request: The parsed `AnalyzeJDRequest` request body, validated by
            FastAPI from the incoming JSON.

    Returns:
        An `AnalyzeJDResponse` with the computed `fit_score` and the
        requirement matches split into `strengths` (verdict "match" or
        "partial") and `gaps` (verdict "gap").

    Raises:
        HTTPException: 404 if `request.resume_id` names a resume that
            doesn't exist — checked up front, before any slow LLM work, and
            worth failing loudly: matching against a nonexistent resume
            would otherwise report every requirement as a gap.
        aptly.llm.client.LLMOutputError: Propagates up (FastAPI will turn
            an unhandled exception into a 500 response) if the LLM's output
            fails schema validation twice in a row during either the
            requirement-extraction call or a borderline-case judgment call
            inside `match_requirements`.
    """
    if request.resume_id and await run_in_threadpool(get_resume, request.resume_id) is None:
        raise HTTPException(status_code=404, detail="Resume not found.")

    extracted = await run_in_threadpool(
        llm_call, extract_requirements_prompt(request.jd_text), ExtractedRequirements
    )
    matches = await run_in_threadpool(match_requirements, extracted.requirements, request.resume_id)
    fit_score = compute_fit_score(matches)

    return AnalyzeJDResponse(
        fit_score=fit_score,
        strengths=[m for m in matches if m.verdict in ("match", "partial")],
        gaps=[m for m in matches if m.verdict == "gap"],
    )
