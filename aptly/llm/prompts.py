# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""Prompt templates for every LLM call in the project.

Plain Python f-strings — no LangChain prompt-template abstraction. Each
function here returns a complete, ready-to-send prompt string for
`aptly.llm.client.call`, paired with a specific schema in
`aptly.llm.schemas`. Keeping prompts as plain functions (rather than
external template files or a templating library) keeps the exact text the
model sees fully visible and greppable in one place.

Both prompts here were tuned empirically against a real small local model
(llama3.2:3b) and a real resume — see the git history / project notes for
what didn't work: prose-only instructions asking the model to "keep the full
phrase" were not sufficient on their own to stop it collapsing requirements
to bare keywords; a concrete few-shot example plus a two-field schema
(`label` + `detail`, see `aptly.llm.schemas.ExtractedRequirement`) is what
made it reliable.

Run standalone: not applicable. Pure string-building functions with no side
effects — imported by `aptly.retrieval.match` and the API route modules.
"""


def extract_requirements_prompt(jd_text: str) -> str:
    """Build the prompt that asks the model to decompose a JD into requirements.

    Paired with `aptly.llm.schemas.ExtractedRequirements` as the expected
    output schema. This is the first LLM call in the `/analyze-jd` and
    `/prep-list` request flows (see `aptly.api.routes_jd.analyze_jd` and
    `aptly.api.routes_notes.prep_list`) — its output drives every downstream
    retrieval query.

    The prompt includes one concrete worked example (a different, unrelated
    JD) showing both a correct two-field output and a common failure mode,
    because prose instructions alone were not sufficient to stop a 3B model
    from collapsing requirements down to bare keywords (e.g. "Kubernetes"
    instead of "Hands-on experience with Kubernetes for production container
    orchestration") — see the module docstring.

    Args:
        jd_text: The raw job description text, as submitted by the API
            caller. Not chunked or truncated here — this string is expected
            to be a reasonably sized JD (a few hundred words); it is sent to
            the LLM directly, not embedded (so the 256-word-piece embedding
            limit in `config.EMBEDDING_MODEL` does not apply to this call).

    Returns:
        A complete prompt string, ready to pass to
        `aptly.llm.client.call(prompt, ExtractedRequirements)`.
    """
    return f"""You are analyzing a job description to extract its concrete requirements.

Extract a list of atomic requirements — one specific skill, technology, qualification, or
responsibility per item. Split compound requirements apart. Do not include generic filler
("team player", "fast-paced environment") — only concrete, checkable requirements.

For each requirement, return two fields:
- "label": a short name for it (2-4 words), e.g. "Kubernetes experience"
- "detail": the full requirement, preserving the qualifying context from the original
  wording (what kind of work, at what scale, in what context) — always a full phrase,
  never just the bare skill name.

Example job description:
\"\"\"
Looking for a backend engineer with 5+ years of Java experience building high-throughput
trading systems, and familiarity with Kafka for event streaming.
\"\"\"
Correct output:
{{"requirements": [
  {{"label": "Java experience", "detail": "5+ years of Java experience building high-throughput trading systems"}},
  {{"label": "Kafka", "detail": "Familiarity with Kafka for event streaming"}}
]}}

Now extract requirements from this job description:
\"\"\"
{jd_text}
\"\"\"

Return ONLY JSON matching this exact shape, no other text:
{{"requirements": [{{"label": "...", "detail": "..."}}]}}"""


def extract_resume_chunks_prompt(resume_text: str) -> str:
    """Build the prompt that asks the model to split raw resume text into per-bullet chunks.

    Paired with `aptly.llm.schemas.ExtractedResumeChunks` as the expected
    output schema. Used by `aptly.api.routes_resume.upload_resume` to
    automate what was previously a hand-written step (manually authoring one
    JSON file per resume bullet) — see docs/ARCHITECTURE.md §5 for the
    target chunk shape this mirrors.

    The prompt explicitly instructs the model to copy bullet text close to
    verbatim rather than summarizing it, since a paraphrased bullet embeds
    differently (and often more vaguely) than the candidate's actual wording
    — the same underlying concern that motivated the `label`/`detail` split
    in `extract_requirements_prompt`, applied here to a different failure
    mode (summarization/compression instead of keyword-collapse).

    It also explicitly instructs the model to skip non-bullet noise (contact
    info, bare skill lists, section headers, URLs) and shows a bad-output
    example alongside the good one. This was added after a real failure: a
    PDF resume's text, extracted via pypdf, comes out with two-column
    layouts flattened and interleaved — a sidebar list of skills ends up as
    a run of one-or-two-word lines rather than a clearly-marked list. Without
    this guidance the model treated *every* such line as its own "bullet,"
    producing dozens of near-empty chunks (a job title on its own, a bare
    LinkedIn URL, a single skill name with no context) instead of the actual
    handful of substantive achievement sentences.

    Args:
        resume_text: Raw text extracted from an uploaded resume file (see
            `aptly.api.routes_resume.upload_resume`, which extracts this
            from a PDF before calling this function). Not chunked or
            truncated here — sent to the LLM directly, not embedded.

    Returns:
        A complete prompt string, ready to pass to
        `aptly.llm.client.call(prompt, ExtractedResumeChunks)`.

    Note:
        A single LLM call processing an entire resume at once works well
        for a typical 1-2 page resume, but a very long or densely-bulleted
        resume could produce an output long enough to strain the model's
        output length before every bullet is captured. If chunks are
        missing after an upload, the existing hand-written-JSON workflow
        (docs/ARCHITECTURE.md §5) remains available as a fallback for
        anything the automated extraction dropped.
    """
    return f"""You are converting a resume into structured, per-bullet data chunks.

Read the resume text below and extract every distinct bullet/achievement as a separate
chunk. For each chunk, return four fields:
- "company": the employer this bullet is from (empty string "" if not applicable, e.g.
  a projects section with no employer)
- "role": the job title held at that company (empty string "" if not applicable)
- "text": the bullet text itself, copied as close to verbatim as possible from the
  original wording — do NOT summarize, shorten, or paraphrase it
- "tags": a list of specific skills/technologies/concepts mentioned in that bullet

Resume text extracted from a PDF often loses its visual layout — a sidebar list of
skills or a contact-info block can end up as a run of short, disconnected lines mixed
in with the real content. Extract a chunk ONLY for a full sentence describing an
achievement or responsibility. Do NOT create a chunk for: a job title or section
heading on its own line, a bare list of skill/tool names, a URL or email or phone
number, an education entry, or any line that isn't a complete sentence about something
the candidate did.

Example resume text (note the messy contact/skills lines mixed in with real bullets):
\"\"\"
Backend Engineer
Acme Corp · Jan 2022 - Present
CONTACT
jane@example.com
https://linkedin.com/in/jane
- Built a payments reconciliation service in Go, processing 1M+ transactions/day with
  99.99% accuracy.
- Migrated legacy cron jobs to Kubernetes CronJobs, cutting infra cost by 30%.
TECHNICAL SKILLS
Go
Kubernetes
Docker
\"\"\"
Correct output — only the two real achievement sentences become chunks; the job
title, contact block, and skills list are all skipped entirely:
{{"chunks": [
  {{"company": "Acme Corp", "role": "Backend Engineer", "text": "Built a payments reconciliation service in Go, processing 1M+ transactions/day with 99.99% accuracy.", "tags": ["Go", "payments", "reconciliation"]}},
  {{"company": "Acme Corp", "role": "Backend Engineer", "text": "Migrated legacy cron jobs to Kubernetes CronJobs, cutting infra cost by 30%.", "tags": ["Kubernetes", "infrastructure", "cost optimization"]}}
]}}
Incorrect output — do NOT do this (treating the heading, contact info, and each bare
skill name as their own chunks):
{{"chunks": [
  {{"company": "", "role": "", "text": "Backend Engineer", "tags": []}},
  {{"company": "", "role": "", "text": "jane@example.com", "tags": []}},
  {{"company": "", "role": "", "text": "https://linkedin.com/in/jane", "tags": []}},
  {{"company": "", "role": "", "text": "Go", "tags": ["Go"]}},
  {{"company": "", "role": "", "text": "Kubernetes", "tags": ["Kubernetes"]}},
  {{"company": "", "role": "", "text": "Docker", "tags": ["Docker"]}}
]}}

Now extract chunks from this resume — remember, skip headings/contact info/skill
lists entirely, only extract real achievement sentences:
\"\"\"
{resume_text}
\"\"\"

Return ONLY JSON matching this exact shape, no other text:
{{"chunks": [{{"company": "...", "role": "...", "text": "...", "tags": ["..."]}}]}}"""


def judge_match_prompt(requirement: str, chunk_text: str) -> str:
    """Build the prompt that asks the model to judge a borderline requirement/chunk pair.

    Paired with `aptly.llm.schemas.MatchJudgment` as the expected output
    schema. Only called for the "borderline band" of retrieval similarity —
    see `aptly.retrieval.match.match_requirements` — where the cosine
    similarity between a requirement and its best-matching resume chunk is
    too ambiguous for a threshold cutoff to decide alone. Clear matches and
    clear gaps never reach this prompt; it exists purely to break ties.

    Args:
        requirement: The requirement text being evaluated — in practice, the
            `detail` field of an `aptly.llm.schemas.ExtractedRequirement`
            (the full phrase, not the short `label`).
        chunk_text: The `text` of the single resume chunk that retrieval
            ranked highest for this requirement.

    Returns:
        A complete prompt string, ready to pass to
        `aptly.llm.client.call(prompt, MatchJudgment)`.
    """
    return f"""You are judging whether a candidate's resume bullet satisfies a job requirement.

Requirement: "{requirement}"

Resume bullet: "{chunk_text}"

Decide whether the resume bullet demonstrates the requirement. Respond with one of:
- "match" if the bullet clearly demonstrates the requirement
- "partial" if the bullet shows related/adjacent experience but not a clear match
- "no_match" if the bullet does not demonstrate the requirement at all

Return JSON matching this exact shape:
{{"verdict": "match" | "partial" | "no_match", "evidence": "short quote or paraphrase from the resume bullet supporting your verdict"}}

Return ONLY the JSON object, no other text."""
