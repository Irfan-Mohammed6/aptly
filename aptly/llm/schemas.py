# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""Pydantic schemas for every shape the LLM is asked to fill in.

These are strictly the *output* contracts of LLM calls made via
`aptly.llm.client.call(prompt, schema)` — they describe what JSON the model
must produce, and `call` validates the model's raw output against them
(retrying once on failure) before handing back a typed object. See
docs/ARCHITECTURE.md §2 and §8.

Run standalone: not applicable. Pure data-model definitions, imported by
`aptly.llm.client`, `aptly.llm.prompts`, and `aptly.retrieval.match`.
"""

from typing import Literal

from pydantic import BaseModel


class ExtractedRequirement(BaseModel):
    """One atomic requirement extracted from a job description.

    Deliberately split into two fields rather than a single string. In
    practice, a small (3B-parameter) model asked to produce one
    "requirement" string per item reliably collapses it down to a bare
    keyword (e.g. "Kubernetes") regardless of how the prompt is worded —
    it has a strong bias toward keyword-style output for this kind of task.
    A bare keyword embeds too generically to discriminate between resume
    chunks during retrieval.

    Splitting the schema into an explicit short field (`label`, which
    satisfies the model's pull toward brevity) and an explicit long field
    (`detail`, which is what retrieval actually queries against) works with
    that bias instead of fighting it, and reliably produces a fuller,
    more specific phrase in `detail`. See `aptly.llm.prompts.extract_requirements_prompt`
    for the prompt that elicits this shape, and `aptly.retrieval.match.match_requirements`
    for where `detail` gets used.

    Attributes:
        label: A short, human-readable name for the requirement (2-4 words),
            e.g. "Kubernetes experience". Intended for display purposes only —
            never used as a retrieval query.
        detail: The full requirement phrase, preserving whatever qualifying
            context (scope, scale, technology names) the original job
            description wording had, e.g. "Hands-on experience with
            Kubernetes for production container orchestration". This is the
            string embedded and queried against the resume_chunks Chroma
            collection.
    """

    label: str
    detail: str


class ExtractedRequirements(BaseModel):
    """The full output of one requirement-extraction LLM call.

    Produced by `aptly.llm.client.call(extract_requirements_prompt(jd_text), ExtractedRequirements)`
    — see `aptly.api.routes_jd.analyze_jd` and `aptly.api.routes_notes.prep_list`
    for the two call sites.

    Attributes:
        requirements: One `ExtractedRequirement` per atomic requirement found
            in the job description text. Order is whatever order the model
            produced them in (typically matching their order of appearance
            in the source JD).
    """

    requirements: list[ExtractedRequirement]


class ExtractedResumeChunk(BaseModel):
    """One resume bullet/achievement extracted from an uploaded resume.

    Mirrors the shape of a hand-written resume chunk JSON file (see
    docs/ARCHITECTURE.md §5) — produced automatically by
    `aptly.llm.prompts.extract_resume_chunks_prompt` from raw resume text
    instead of being hand-authored. `aptly.api.routes_resume.upload_resume`
    converts a list of these into on-disk JSON files via
    `aptly.ingestion.resume.write_resume_chunks`.

    Attributes:
        company: The employer this bullet is from, e.g. "Acme Corp". Empty
            string if the model couldn't determine it (e.g. for a resume
            section without a clear company heading, like a projects list).
        role: The job title held at that company, e.g. "Associate Software
            Engineer". Empty string if not determinable.
        text: The bullet/achievement text itself, copied as close to
            verbatim as possible from the source resume — this is what gets
            embedded and retrieved against, so it should read like the
            original bullet, not a summary of it.
        tags: A list of specific skills/technologies/concepts mentioned in
            this bullet, e.g. ["ETL", "Python", "MongoDB"].
    """

    company: str
    role: str
    text: str
    tags: list[str]


class ExtractedResumeChunks(BaseModel):
    """The full output of one resume-chunking LLM call.

    Produced by `aptly.llm.client.call(extract_resume_chunks_prompt(resume_text), ExtractedResumeChunks)`
    — see `aptly.api.routes_resume.upload_resume`.

    Attributes:
        chunks: One `ExtractedResumeChunk` per bullet/achievement found in
            the resume text, in the order they appeared in the source
            document.
    """

    chunks: list[ExtractedResumeChunk]


class MatchJudgment(BaseModel):
    """The output of one borderline-similarity match-judgment LLM call.

    Only requested for requirements whose best retrieval similarity falls
    strictly between `config.GAP_THRESHOLD` and `config.CONFIDENT_MATCH_THRESHOLD`
    — see `aptly.retrieval.match.match_requirements`. Clear matches and clear
    gaps are decided by the similarity score alone, with no LLM call at all;
    this schema exists only to break the tie for the ambiguous middle band.

    Attributes:
        verdict: One of:
            - "match": the resume chunk clearly demonstrates the requirement.
            - "partial": the chunk shows related/adjacent experience, but not
              a clear, direct match.
            - "no_match": the chunk does not demonstrate the requirement at
              all. `match_requirements` maps this to the final verdict
              "gap" (the `RequirementMatch` schema has no "no_match" state
              of its own — a judged non-match and a retrieval-confirmed
              absence are reported identically as "gap").
        evidence: A short quote or close paraphrase from the resume chunk
            text that supports the verdict — surfaced back to the API caller
            so a human can sanity-check the model's reasoning rather than
            trusting the verdict blindly.
    """

    verdict: Literal["match", "partial", "no_match"]
    evidence: str
