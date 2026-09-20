# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""Per-requirement retrieval and gap/match verdict logic.

This module is the architectural core of Aptly's gap analysis. Rather than
embedding a whole job description and doing one top-k lookup against the
resume (which can only ever surface what similarity happens to rank
highest, never confirm that something is *absent*), each requirement
extracted from the JD is retrieved against the resume independently. A
requirement whose best match is weak *is* the gap signal, produced by
retrieval alone — no LLM judgment needed to notice an absence. See
docs/ARCHITECTURE.md §2 and §3 for the full rationale and call sequence.

Run standalone: not applicable. This is a library module — imported by
`aptly.scoring` (which consumes its `RequirementMatch` output) and
`aptly.api.routes_jd.analyze_jd`. To exercise it manually (requires Ollama
running and the resume_chunks collection populated via `scripts/reindex.py`):

    python -c "
    from aptly.llm.schemas import ExtractedRequirement
    from aptly.retrieval.match import match_requirements
    reqs = [ExtractedRequirement(label='Python', detail='3+ years of Python ETL experience')]
    print(match_requirements(reqs))
    "
"""

from typing import Literal

from pydantic import BaseModel

from aptly import config
from aptly.llm.client import call as llm_call
from aptly.llm.prompts import judge_match_prompt
from aptly.llm.schemas import ExtractedRequirement, MatchJudgment
from aptly.retrieval import store


class RequirementMatch(BaseModel):
    """The outcome of matching one job requirement against the resume.

    One of these is produced per `ExtractedRequirement` by
    `match_requirements`. A list of these is what `aptly.scoring.compute_fit_score`
    consumes to produce the final numeric fit score, and what
    `aptly.api.routes_jd.analyze_jd` splits into the API response's
    `strengths` and `gaps` lists.

    Attributes:
        requirement: The full requirement text that was matched against
            (the `detail` field of the originating `ExtractedRequirement`,
            not its short `label`).
        best_chunk_id: The Chroma id of the resume chunk that scored highest
            for this requirement, or `None` if the resume_chunks collection
            was empty at query time.
        similarity: The cosine similarity (roughly 0-1, higher = closer)
            between the requirement and `best_chunk_id`'s text. `0.0` when
            `best_chunk_id` is `None`.
        verdict: One of:
            - "match": either similarity cleared `config.CONFIDENT_MATCH_THRESHOLD`
              outright, or an LLM judgment in the borderline band decided
              "match".
            - "partial": an LLM judgment in the borderline band decided the
              resume chunk shows related but not clearly matching experience.
            - "gap": either similarity fell below `config.GAP_THRESHOLD`
              outright, or an LLM judgment in the borderline band decided
              "no_match" (the two cases are not distinguished in the output —
              both mean "the resume doesn't demonstrate this").
        evidence: A short quote/paraphrase from the matched resume chunk
            supporting the verdict, or `None` for a threshold-decided "gap"
            (there is nothing to quote in support of an absence).
    """

    requirement: str
    best_chunk_id: str | None
    similarity: float
    verdict: Literal["match", "partial", "gap"]
    evidence: str | None


def match_requirements(
    requirements: list[ExtractedRequirement], resume_id: str | None = None
) -> list[RequirementMatch]:
    """Match each extracted job requirement against the candidate's resume chunks.

    For every requirement, this queries the `resume_chunks` Chroma
    collection independently (via `aptly.retrieval.store.get_store().query`)
    and decides a verdict from the single best-matching chunk's similarity:

        similarity < config.GAP_THRESHOLD              -> verdict="gap", no LLM call
        similarity >= config.CONFIDENT_MATCH_THRESHOLD  -> verdict="match", no LLM call
        otherwise (the "borderline band")               -> one LLM judgment call
                                                            (aptly.llm.client.call with
                                                            judge_match_prompt / MatchJudgment)
                                                            breaks the tie

    Retrieval queries on `requirement.detail` (the full phrase preserving
    scale/context), never `requirement.label` (the short display name) — a
    bare keyword embeds too generically to discriminate between resume
    chunks reliably. See `aptly.llm.schemas.ExtractedRequirement` for why
    that distinction exists.

    This function makes blocking calls (both to Chroma, which runs the
    embedding model synchronously, and potentially to the LLM for borderline
    cases) and is not async itself — callers inside FastAPI route handlers
    must invoke it via `fastapi.concurrency.run_in_threadpool` rather than
    calling it directly from an `async def` route.

    Args:
        requirements: The requirements to match, typically the
            `requirements` field of an `aptly.llm.schemas.ExtractedRequirements`
            returned by an extraction LLM call.
        resume_id: If given, only that resume's chunks are searched — this is
            how an analysis runs against one chosen resume out of several.
            `None` searches every resume's chunks together.

    Returns:
        A list of `RequirementMatch`, one per input requirement, in the same
        order as `requirements`.
    """
    chroma_store = store.get_store()
    matches: list[RequirementMatch] = []
    where = {"resume_id": resume_id} if resume_id else None

    for requirement in requirements:
        hits = chroma_store.query(
            config.RESUME_COLLECTION, requirement.detail, k=config.TOP_K, where=where
        )

        if not hits:
            matches.append(
                RequirementMatch(
                    requirement=requirement.detail,
                    best_chunk_id=None,
                    similarity=0.0,
                    verdict="gap",
                    evidence=None,
                )
            )
            continue

        best = hits[0]
        similarity = best["similarity"]

        if similarity < config.GAP_THRESHOLD:
            verdict: Literal["match", "partial", "gap"] = "gap"
            evidence = None
        elif similarity >= config.CONFIDENT_MATCH_THRESHOLD:
            verdict = "match"
            evidence = best["document"]
        else:
            judgment = llm_call(judge_match_prompt(requirement.detail, best["document"]), MatchJudgment)
            verdict = "gap" if judgment.verdict == "no_match" else judgment.verdict
            evidence = judgment.evidence

        matches.append(
            RequirementMatch(
                requirement=requirement.detail,
                best_chunk_id=best["id"],
                similarity=similarity,
                verdict=verdict,
                evidence=evidence,
            )
        )

    return matches
