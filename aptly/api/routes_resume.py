# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""Resume endpoints: upload a PDF resume, and list / preview / rename / delete them.

`POST /upload-resume` automates what was previously a manual step —
hand-writing one JSON file per resume bullet in `data/resume_chunks/` (see
docs/ARCHITECTURE.md §5). A candidate uploads their resume as a PDF; the text
is extracted, an LLM call splits it into per-bullet chunks
(`aptly.llm.schemas.ExtractedResumeChunks`), and those chunks are written to
disk and indexed exactly as if they had been hand-authored — the file-first,
disk-is-the-source-of-truth model is unchanged, only how the files get
created is new.

Several resumes can coexist: each upload becomes its own named resume, and
`GET /resumes`, `GET /resumes/{id}/chunks`, `PATCH /resumes/{id}` and
`DELETE /resumes/{id}` manage them. `POST /analyze-jd` then takes a
`resume_id` to choose which one an analysis runs against.

Run standalone: not applicable — this module only defines a FastAPI
`APIRouter`, it has no entry point of its own. Start the whole application
with `uvicorn aptly.api.main:app --reload` (see aptly/api/main.py), then:

    curl -X POST http://localhost:8000/upload-resume \\
      -F "file=@/path/to/your/resume.pdf"
"""

import io
import re

import pypdf
from fastapi import APIRouter, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from aptly.api.models import RenameResumeRequest, ResumeChunkInfo, ResumeInfo, UploadResumeResponse
from aptly.ingestion.resume import ingest_resume_chunks, load_resume_chunks, write_resume_chunks
from aptly.ingestion.resumes import (
    create_resume,
    delete_resume,
    get_resume,
    list_resumes,
    rename_resume,
)
from aptly.llm.client import call as llm_call
from aptly.llm.prompts import extract_resume_chunks_prompt
from aptly.llm.schemas import ExtractedResumeChunks

router = APIRouter()

#: Section header names (case-insensitive, matched against a whole trimmed line)
#: that mark the *start* of a block to drop entirely — content that is never a
#: bullet/achievement (a skills list, contact details, education, etc).
_DROP_SECTION_HEADERS = {
    "contact",
    "technical skills",
    "areas of expertise",
    "skills",
    "short courses",
    "education",
    "certifications",
    "languages",
    "hobbies",
    "interests",
}

#: Section header names that mark the *end* of a drop block — dropping stops
#: once one of these is seen, since real bullet content follows.
_KEEP_SECTION_HEADERS = {
    "experience",
    "work experience",
    "professional experience",
    "summary",
    "objective",
    "projects",
    "highlights",
}

_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_URL_RE = re.compile(r"https?://\S+")
_PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{7,}\d")


def _clean_resume_text(raw_text: str) -> str:
    """Strip non-bullet noise from PDF-extracted resume text before it reaches the LLM.

    Internal helper for `_process_resume` — not part of the module's public
    interface. Added after a real failure: `aptly.llm.prompts.extract_resume_chunks_prompt`
    alone (even with explicit "skip contact info / skill lists / headers"
    instructions and a bad-output example) wasn't reliably enough for a 3B
    model to ignore that noise on its own — an upload produced 37 near-empty
    chunks (a bare job title, a bare LinkedIn URL, single skill words) instead
    of the actual ~12 achievement bullets, and processing all that extra,
    confusing content also meaningfully slowed generation down. Removing it
    with plain text processing before the LLM ever sees it is more reliable
    than asking the model to mentally filter it out, and shrinks the prompt.

    Two passes, line by line:
        1. Section-based: once a line exactly matches a name in
           `_DROP_SECTION_HEADERS` (e.g. "TECHNICAL SKILLS"), every
           subsequent line is dropped until one matches
           `_KEEP_SECTION_HEADERS` (e.g. "EXPERIENCE") or the text ends.
           This removes an entire skills/contact/education block in one
           shot, since within such a block every line is noise.
        2. Pattern-based, applied to whatever survives pass 1: email
           addresses, URLs, and phone-number-shaped digit runs are stripped
           wherever they appear, since a resume's contact line can appear
           outside a cleanly-headed block too.

    This is a heuristic, not a full resume parser — it assumes section
    headers appear as their own line and uses a fixed, English-language list
    of common header names. It won't catch every possible resume layout
    (and one genuine limitation was observed: content that a PDF's column
    layout flattened to appear *before* its own section header, e.g. a short
    course name appearing right above the literal words "SHORT COURSES",
    survives this filter). It only needs to remove *most* of the noise —
    `extract_resume_chunks_prompt`'s own instructions are the second line of
    defense for whatever slips through.

    Args:
        raw_text: Raw text as extracted by `_extract_pdf_text`.

    Returns:
        The same text with recognized noise sections and patterns removed,
        blank lines collapsed out.
    """
    lines = []
    dropping = False
    for line in raw_text.split("\n"):
        stripped = line.strip()
        lower = stripped.lower()
        if lower in _DROP_SECTION_HEADERS:
            dropping = True
            continue
        if lower in _KEEP_SECTION_HEADERS:
            dropping = False
            continue
        if dropping or not stripped:
            continue
        lines.append(stripped)

    cleaned = "\n".join(lines)
    cleaned = _EMAIL_RE.sub("", cleaned)
    cleaned = _URL_RE.sub("", cleaned)
    cleaned = _PHONE_RE.sub("", cleaned)
    return "\n".join(line.strip() for line in cleaned.split("\n") if line.strip())


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF's raw bytes, page by page.

    Internal helper for `upload_resume` — not part of the module's public
    interface. Uses `pypdf`, a pure-Python PDF library — no external
    binaries or network calls, keeping the project's fully-offline,
    dependency-light posture.

    Args:
        pdf_bytes: The raw bytes of a PDF file, as read from an uploaded
            `UploadFile`.

    Returns:
        The concatenated text of every page, separated by newlines. Pages
        pypdf fails to extract text from (e.g. a scanned image with no text
        layer) contribute an empty string rather than raising an error.
    """
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


#: Minimum character length for an extracted chunk's `text` to be kept. Real
#: achievement bullets are always full sentences (the shortest one observed
#: in testing was well over 100 characters); anything shorter is reliably a
#: leaked label or fragment (e.g. "Highlights:", a bare company name) rather
#: than genuine bullet content. Deliberately conservative — low enough that
#: no real bullet could plausibly be dropped, high enough to catch the junk
#: actually observed.
_MIN_CHUNK_TEXT_LENGTH = 20


def _dedupe_and_filter_chunks(chunks: list[dict]) -> list[dict]:
    """Drop empty/near-empty chunks and duplicate bullet text from one extraction batch.

    Internal helper for `_process_resume` — not part of the module's public
    interface. Added after a real observation: even with `_clean_resume_text`
    removing the bulk of the noise, one extraction run still produced a few
    junk chunks (bare labels like "Highlights:", a lone company name with no
    achievement text, one chunk with empty `text`) and duplicated one real
    bullet four times with different guessed `company`/`role` values each
    time. Both failure modes are cheap to catch here with plain Python,
    without needing another LLM call.

    Args:
        chunks: Chunk dicts as produced by `ExtractedResumeChunk.model_dump()`,
            before being written to disk.

    Returns:
        The input list with any chunk whose `text` is shorter than
        `_MIN_CHUNK_TEXT_LENGTH` removed, and subsequent chunks with a
        `text` identical to an earlier one in the same batch removed —
        first occurrence of each unique text wins, so the result preserves
        the model's original ordering.
    """
    seen_text: set[str] = set()
    kept = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if len(text) < _MIN_CHUNK_TEXT_LENGTH or text in seen_text:
            continue
        seen_text.add(text)
        kept.append(chunk)
    return kept


def _process_resume(pdf_bytes: bytes, name: str, filename: str) -> tuple[dict, int]:
    """Run the full extract -> register -> write -> ingest pipeline for one uploaded resume.

    Internal helper for `upload_resume` — not part of the module's public
    interface. Split out so the entire pipeline (PDF text extraction, the
    blocking LLM call, disk writes, and Chroma ingestion) can be dispatched
    as a single unit via `fastapi.concurrency.run_in_threadpool`.

    The LLM's raw output is passed through `_dedupe_and_filter_chunks`
    before anything is written — even with `_clean_resume_text` removing
    most non-bullet noise beforehand, the model can still occasionally
    duplicate a bullet or emit a near-empty chunk, and that's cheaper to
    catch here than to prevent perfectly via prompting alone.

    After writing the newly extracted chunks to disk
    (`write_resume_chunks`), this re-ingests the *entire* resume_chunks
    directory (`ingest_resume_chunks(load_resume_chunks())`) rather than
    just the new chunks — deliberately, so the Chroma index always reflects
    every file on disk exactly, with no risk of the index and the
    filesystem drifting apart. At the scale of a single resume (a few dozen
    chunks at most), re-embedding everything on each upload is cheap.

    The resume is only registered (`aptly.ingestion.resumes.create_resume`)
    *after* extraction succeeds, so a failed upload doesn't leave an empty
    resume in the list.

    Args:
        pdf_bytes: The raw bytes of the uploaded PDF resume.
        name: Display name for the new resume.
        filename: The uploaded file's original name.

    Returns:
        A tuple of the new resume's registry entry and the number of chunks
        extracted and written for it.

    Raises:
        HTTPException: 422 if no extractable text is found in the PDF (e.g.
            a scanned/image-only resume with no text layer).
        aptly.llm.client.LLMOutputError: If the chunk-extraction LLM call
            fails schema validation twice in a row.
    """
    resume_text = _clean_resume_text(_extract_pdf_text(pdf_bytes))
    if not resume_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not extract any text from the uploaded PDF (it may be a scanned image with no text layer).",
        )

    extracted = llm_call(extract_resume_chunks_prompt(resume_text), ExtractedResumeChunks)
    new_chunks = [chunk.model_dump() for chunk in extracted.chunks]
    new_chunks = _dedupe_and_filter_chunks(new_chunks)
    if not new_chunks:
        raise HTTPException(status_code=422, detail="No resume bullets could be extracted from that PDF.")

    resume = create_resume(name, filename)
    write_resume_chunks(new_chunks, resume["id"])
    ingest_resume_chunks(load_resume_chunks())
    return resume, len(new_chunks)


@router.post("/upload-resume", response_model=UploadResumeResponse)
async def upload_resume(file: UploadFile, name: str | None = Form(default=None)) -> UploadResumeResponse:
    """Upload a PDF resume and add it as a new, separately named resume.

    Extracts text from the uploaded PDF, asks the LLM to split it into
    per-bullet chunks, writes each as a new JSON file in
    `data/resume_chunks/` tagged with the new resume's id, and re-indexes the
    resume_chunks Chroma collection from the full contents of that
    directory — see `_process_resume` for the pipeline details. Existing
    resumes are left untouched: each upload creates a new resume.

    The entire pipeline is dispatched via
    `fastapi.concurrency.run_in_threadpool` since it involves a blocking LLM
    call and blocking embedding/Chroma writes that would otherwise stall the
    async event loop.

    Args:
        file: The uploaded file, expected to be a PDF (validated by
            filename extension / content type before processing).
        name: Optional display name (form field). Defaults to the file's
            name without its extension.

    Returns:
        An `UploadResumeResponse` with the new resume's id and name and the
        count of chunks created.

    Raises:
        HTTPException: 422 if the uploaded file isn't a PDF, or if no text
            could be extracted from it.
        aptly.llm.client.LLMOutputError: Propagates up if the chunk
            extraction LLM call fails schema validation twice in a row.
    """
    is_pdf = (file.content_type == "application/pdf") or (file.filename or "").lower().endswith(".pdf")
    if not is_pdf:
        raise HTTPException(status_code=422, detail="Only PDF resumes are supported.")

    filename = file.filename or ""
    display_name = (name or "").strip() or re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE) or "Resume"

    pdf_bytes = await file.read()
    resume, chunks_created = await run_in_threadpool(_process_resume, pdf_bytes, display_name, filename)
    return UploadResumeResponse(chunks_created=chunks_created, resume_id=resume["id"], name=resume["name"])


@router.get("/resumes", response_model=list[ResumeInfo])
async def get_resumes() -> list[ResumeInfo]:
    """List all uploaded resumes, oldest first, each with its chunk count."""
    return await run_in_threadpool(list_resumes)


@router.get("/resumes/{resume_id}/chunks", response_model=list[ResumeChunkInfo])
async def get_resume_chunks(resume_id: str) -> list[ResumeChunkInfo]:
    """Return one resume's chunks, for previewing what was extracted from it.

    Raises:
        HTTPException: 404 if there is no such resume.
    """
    if await run_in_threadpool(get_resume, resume_id) is None:
        raise HTTPException(status_code=404, detail="Resume not found.")
    chunks = await run_in_threadpool(load_resume_chunks, resume_id)
    return [
        ResumeChunkInfo(
            id=c["id"],
            company=c.get("company", ""),
            role=c.get("role", ""),
            text=c["text"],
            tags=c.get("tags", []),
        )
        for c in chunks
    ]


@router.patch("/resumes/{resume_id}", response_model=ResumeInfo)
async def patch_resume(resume_id: str, request: RenameResumeRequest) -> ResumeInfo:
    """Rename a resume. Its id, and so its chunks' link to it, doesn't change.

    Raises:
        HTTPException: 422 for an empty name; 404 if there is no such resume.
    """
    if not request.name.strip():
        raise HTTPException(status_code=422, detail="Name must not be empty.")
    updated = await run_in_threadpool(rename_resume, resume_id, request.name)
    if updated is None:
        raise HTTPException(status_code=404, detail="Resume not found.")
    return updated


@router.delete("/resumes/{resume_id}", status_code=204)
async def remove_resume(resume_id: str) -> Response:
    """Delete a resume along with its chunk files and index entries.

    Raises:
        HTTPException: 404 if there is no such resume.
    """
    if not await run_in_threadpool(delete_resume, resume_id):
        raise HTTPException(status_code=404, detail="Resume not found.")
    return Response(status_code=204)
