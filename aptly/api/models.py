# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""Request and response schemas for the public HTTP API.

These are the shapes exposed at the FastAPI boundary — what a client sends
and receives over HTTP. They are deliberately kept separate from
`aptly.llm.schemas` (which shapes what the *LLM* is asked to produce
internally): the API's public contract and the LLM's internal output format
are allowed to diverge and evolve independently. See docs/ARCHITECTURE.md §2.

Run standalone: not applicable. Pure Pydantic data-model definitions, used
by FastAPI for request validation and response serialization (including
auto-generated OpenAPI docs at `/docs`) — imported by
`aptly.api.routes_jd` and `aptly.api.routes_notes`.
"""

from typing import Literal

from pydantic import BaseModel

from aptly.retrieval.match import RequirementMatch


class ChatMessage(BaseModel):
    """One message in a `/chat` conversation.

    Attributes:
        role: Who said it — "user", "assistant" (the model's earlier
            replies), or "system" (optional instructions).
        content: The message text.
    """

    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Request body for `POST /chat`.

    Attributes:
        messages: The conversation so far, oldest first, ending with the
            new user message. The server is stateless — the client resends
            the history on every request.
    """

    messages: list[ChatMessage]


class ChatResponse(BaseModel):
    """Response body for `POST /chat`.

    Attributes:
        reply: The model's reply text.
        model: The Ollama model tag that produced it (`config.OLLAMA_MODEL`),
            shown in the UI so it's clear which model is being tested.
    """

    reply: str
    model: str


class ResumeInfo(BaseModel):
    """One uploaded resume, as returned by `GET /resumes`.

    Attributes:
        id: Stable id (also the `resume_id` on each of its chunks).
        name: Display name shown in the UI.
        filename: The originally uploaded file's name ("" if unknown).
        chunk_count: How many bullet-level chunks it currently has.
        created_at: ISO-8601 UTC timestamp.
        updated_at: ISO-8601 UTC timestamp of the last change.
    """

    id: str
    name: str
    filename: str
    chunk_count: int
    created_at: str
    updated_at: str


class ResumeChunkInfo(BaseModel):
    """One chunk of a resume, for the "view resume" preview."""

    id: str
    company: str
    role: str
    text: str
    tags: list[str]


class RenameResumeRequest(BaseModel):
    """Request body for `PATCH /resumes/{id}`."""

    name: str


class NoteInfo(BaseModel):
    """One concept note, as returned by `GET /notes`.

    Attributes:
        id: The note's id (its filename stem).
        title: Display title.
        tags: Its tags.
        body: The full Markdown body.
    """

    id: str
    title: str
    tags: list[str]
    body: str


class AnalyzeJDRequest(BaseModel):
    """Request body for `POST /analyze-jd`.

    Attributes:
        jd_text: The full raw text of a job description, pasted as-is. Not
            expected to be pre-chunked or pre-processed by the caller —
            requirement extraction happens server-side.
        resume_id: Which resume to analyze against (an id from
            `GET /resumes`). `None` matches against every resume's chunks.
    """

    jd_text: str
    resume_id: str | None = None


class AnalyzeJDResponse(BaseModel):
    """Response body for `POST /analyze-jd`.

    Attributes:
        fit_score: An integer in [0, 100] indicating overall resume/JD fit.
            Computed deterministically by `aptly.scoring.compute_fit_score`
            from the `strengths`/`gaps` verdicts below — never generated
            directly by the LLM (see `aptly.scoring` for why).
        strengths: The requirements whose verdict was "match" or "partial" —
            i.e. the resume chunks demonstrate them, in whole or in part.
        gaps: The requirements whose verdict was "gap" — i.e. nothing in the
            resume closely matches them, whether that was decided by
            retrieval similarity alone or by an LLM judgment call.
    """

    fit_score: int
    strengths: list[RequirementMatch]
    gaps: list[RequirementMatch]


class AddNoteRequest(BaseModel):
    """Request body for `POST /add-note`.

    Attributes:
        title: Human-readable title for the note. Used to derive both the
            on-disk filename and the Chroma document id (via
            `aptly.ingestion.notes._slugify`).
        tags: A list of freeform tag strings for categorizing the note.
        body: The Markdown body content of the note.
    """

    title: str
    tags: list[str]
    body: str


class AddNoteResponse(BaseModel):
    """Response body for `POST /add-note`.

    Attributes:
        note_id: The id under which the new note was stored — both as the
            filename stem on disk (`data/concept_notes/{note_id}.md`) and as
            the document id in the `concept_notes` Chroma collection.
    """

    note_id: str


class UploadResumeResponse(BaseModel):
    """Response body for `POST /upload-resume`.

    Attributes:
        chunks_created: How many new resume chunk JSON files were written
            to `data/resume_chunks/` and indexed, from the uploaded PDF —
            see `aptly.api.routes_resume.upload_resume`.
        resume_id: Id of the resume these chunks were saved under.
        name: The resume's display name.
    """

    chunks_created: int
    resume_id: str
    name: str


class PrepListRequest(BaseModel):
    """Request body for `POST /prep-list`.

    Attributes:
        jd_text: The full raw text of a job description. Its extracted
            requirements are used to retrieve relevant concept notes from
            your personal knowledge base — see
            `aptly.api.routes_notes.prep_list`.
    """

    jd_text: str


class PrepListResponse(BaseModel):
    """Response body for `POST /prep-list`.

    Attributes:
        notes: Up to 5 concept notes ranked by aggregate relevance to the
            job description's requirements, most relevant first. Each dict
            has the keys `chunk_id`, `title`, `tags`, and `snippet` (the
            first 200 characters of the note body) — see
            `aptly.api.routes_notes._gather_prep_notes` for how this is
            built.
    """

    notes: list[dict]
