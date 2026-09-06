# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""Central configuration for Aptly.

Every tunable constant used anywhere in the project — model names, file paths,
Chroma collection names, and the similarity thresholds that drive gap/match
verdicts — lives here and only here. No other module should hard-code a path,
model name, or threshold; import it from this module instead.

This module has no side effects and does not read the network or the
filesystem beyond resolving its own path — it is safe to import from
anywhere, including at the top of test files.

Run standalone: not applicable. This is a pure library module (a namespace of
constants) — it is imported by every other module in the project and is never
executed directly.
"""

from pathlib import Path

# --- LLM settings -----------------------------------------------------------

#: The Ollama model tag used for every LLM call (requirement extraction and
#: borderline match judgment). Must already be pulled locally — see README.md
#: for the `ollama pull` command.
OLLAMA_MODEL = "llama3.2:3b"

#: Base URL of the local Ollama server. Aptly never calls any other network
#: endpoint for inference — this is what makes the tool fully offline once
#: the model is pulled.
OLLAMA_HOST = "http://localhost:11434"

# --- Embedding settings ------------------------------------------------------

#: sentence-transformers model used to embed all text for retrieval (resume
#: chunks, concept notes, and JD requirements). This particular model has a
#: hard 256 word-piece input limit and truncates silently beyond that — never
#: feed it a raw, whole job description; only short, atomic requirement
#: phrases and resume/notes chunks, which comfortably fit.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Retrieval / scoring thresholds ------------------------------------------

#: Number of nearest neighbours returned per Chroma query.
TOP_K = 3

#: Cosine similarity floor for a requirement to be considered a match at all.
#: Below this, the requirement is marked verdict="gap" immediately, with no
#: LLM call — retrieval alone is treated as sufficient evidence of absence.
GAP_THRESHOLD = 0.45

#: Cosine similarity ceiling above which a requirement is considered a
#: confident verdict="match" immediately, with no LLM call. Between
#: GAP_THRESHOLD and this value is the "borderline band", where a single LLM
#: judgment call (see aptly.llm.client.call with MatchJudgment) breaks the tie.
#:
#: Both thresholds are starting guesses, not measured constants — tune them
#: using aptly.eval.retrieval_eval against your own hand-labeled JDs once you
#: have real usage data. See docs/ARCHITECTURE.md §6.
CONFIDENT_MATCH_THRESHOLD = 0.75

# --- Filesystem paths ---------------------------------------------------------

#: Absolute path to the project root (the directory containing this package).
BASE_DIR = Path(__file__).resolve().parent.parent

#: Root of all on-disk data. Everything under here except `chroma/` is the
#: source of truth for the project — see docs/ARCHITECTURE.md §1.
DATA_DIR = BASE_DIR / "data"

#: Where Chroma persists its vector index. This directory is entirely
#: disposable and rebuildable from `RESUME_CHUNKS_DIR` and
#: `CONCEPT_NOTES_DIR` via scripts/reindex.py — never treat it as a source of
#: truth, and never commit it to git (see .gitignore).
CHROMA_PERSIST_DIR = DATA_DIR / "chroma"

#: Directory of hand-written resume chunk JSON files. One file per
#: bullet/achievement. See docs/ARCHITECTURE.md §5 for the expected shape.
RESUME_CHUNKS_DIR = DATA_DIR / "resume_chunks"

#: Directory of concept note Markdown files (YAML frontmatter + body). Grows
#: over time both by hand and via the POST /add-note endpoint.
CONCEPT_NOTES_DIR = DATA_DIR / "concept_notes"

# --- Chroma collection names ---------------------------------------------------

#: Name of the Chroma collection holding embedded resume chunks.
RESUME_COLLECTION = "resume_chunks"

#: Name of the Chroma collection holding embedded concept notes.
CONCEPT_NOTES_COLLECTION = "concept_notes"
