# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""The POST /add-note and POST /prep-list endpoints: the personal knowledge base.

See docs/ARCHITECTURE.md §4 (prep-list call sequence) and §5 (add-note call
sequence) for the architectural rationale.

Run standalone: not applicable — this module only defines a FastAPI
`APIRouter`, it has no entry point of its own. Start the whole application
with `uvicorn aptly.api.main:app --reload` (see aptly/api/main.py), then:

    curl -X POST http://localhost:8000/add-note \\
      -H "Content-Type: application/json" \\
      -d '{"title": "RAG vs Fine-tuning", "tags": ["GenAI", "RAG"], "body": "RAG retrieves context at query time..."}'

    curl -X POST http://localhost:8000/prep-list \\
      -H "Content-Type: application/json" \\
      -d '{"jd_text": "Looking for someone who understands transformer architecture and RAG..."}'
"""

from collections import defaultdict

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from aptly import config
from aptly.api.models import (
    AddNoteRequest,
    AddNoteResponse,
    PrepListRequest,
    PrepListResponse,
)
from aptly.ingestion.notes import ingest_concept_notes, write_concept_note
from aptly.llm.client import call as llm_call
from aptly.llm.prompts import extract_requirements_prompt
from aptly.llm.schemas import ExtractedRequirement, ExtractedRequirements
from aptly.retrieval.store import get_store

router = APIRouter()


@router.post("/add-note", response_model=AddNoteResponse)
async def add_note(request: AddNoteRequest) -> AddNoteResponse:
    """Add a new concept note to the personal knowledge base.

    Writes the note to disk first (`write_concept_note` — the file is the
    permanent source of truth), then embeds and indexes it
    (`ingest_concept_notes`) so it is immediately retrievable by a
    subsequent `/prep-list` call in the same process, with no separate
    reindex step required.

    Both steps are dispatched via `fastapi.concurrency.run_in_threadpool`
    since embedding is a blocking, synchronous operation that would
    otherwise stall the async event loop.

    Args:
        request: The parsed `AddNoteRequest` — `title`, `tags`, and `body`.

    Returns:
        An `AddNoteResponse` containing the new note's id (its filename
        stem, e.g. `rag-vs-fine-tuning`), which is also its Chroma document
        id in the `concept_notes` collection.
    """
    path = await run_in_threadpool(write_concept_note, request.title, request.tags, request.body)
    note = {"id": path.stem, "title": request.title, "tags": request.tags, "body": request.body}
    await run_in_threadpool(ingest_concept_notes, [note])
    return AddNoteResponse(note_id=path.stem)


def _gather_prep_notes(requirements: list[ExtractedRequirement]) -> list[dict]:
    """Rank concept notes by aggregate relevance across multiple requirements.

    Internal helper for `prep_list` — not part of the module's public
    interface. Unlike `aptly.retrieval.match.match_requirements` (which
    cares only about the single best-matching resume chunk per
    requirement), this function deliberately aggregates: a note that scores
    moderately well against *several* requirements should rank above one
    that scores very well against only a single requirement, since the
    former is more broadly useful prep material for the job description as
    a whole.

    For each requirement, the top `config.TOP_K` concept notes are queried;
    each note's similarity scores are summed across all requirements it
    appeared for (`scores`), and the single highest-similarity hit seen for
    each note is kept for display purposes (`best_hit`).

    Args:
        requirements: The requirements to gather relevant notes for,
            typically the `requirements` field of an `ExtractedRequirements`
            from a requirement-extraction LLM call.

    Returns:
        A list of up to 5 dicts, ranked by summed similarity score
        (descending), each with:
            - `chunk_id`: the note's Chroma document id.
            - `title`: from the note's stored metadata, falling back to
              `chunk_id` if missing.
            - `tags`: the note's tags, stored as a comma-joined string.
            - `snippet`: the first 200 characters of the note's body.
    """
    chroma_store = get_store()
    scores: dict[str, float] = defaultdict(float)
    best_hit: dict[str, dict] = {}

    for requirement in requirements:
        for hit in chroma_store.query(config.CONCEPT_NOTES_COLLECTION, requirement.detail, k=config.TOP_K):
            scores[hit["id"]] += hit["similarity"]
            if hit["id"] not in best_hit or hit["similarity"] > best_hit[hit["id"]]["similarity"]:
                best_hit[hit["id"]] = hit

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        {
            "chunk_id": note_id,
            "title": best_hit[note_id]["metadata"].get("title", note_id),
            "tags": best_hit[note_id]["metadata"].get("tags", ""),
            "snippet": best_hit[note_id]["document"][:200],
        }
        for note_id, _ in ranked[:5]
    ]


@router.post("/prep-list", response_model=PrepListResponse)
async def prep_list(request: PrepListRequest) -> PrepListResponse:
    """Retrieve the most relevant concept notes for a job description.

    Reuses the same requirement-extraction step as `/analyze-jd`
    (`extract_requirements_prompt` + `ExtractedRequirements`), but queries
    the `concept_notes` Chroma collection instead of `resume_chunks`, and
    ranks results by aggregate relevance across all requirements rather than
    per-requirement best match — see `_gather_prep_notes`.

    Both the LLM extraction call and the retrieval/ranking step
    (`_gather_prep_notes`) are dispatched via
    `fastapi.concurrency.run_in_threadpool` since both make blocking calls.

    Args:
        request: The parsed `PrepListRequest` — the job description text.

    Returns:
        A `PrepListResponse` with up to 5 ranked concept notes relevant to
        the job description's stack/requirements.

    Raises:
        aptly.llm.client.LLMOutputError: Propagates up if the requirement
            extraction LLM call fails schema validation twice in a row.
    """
    extracted = await run_in_threadpool(
        llm_call, extract_requirements_prompt(request.jd_text), ExtractedRequirements
    )
    notes = await run_in_threadpool(_gather_prep_notes, extracted.requirements)
    return PrepListResponse(notes=notes)
