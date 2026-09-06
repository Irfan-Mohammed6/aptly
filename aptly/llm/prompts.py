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
