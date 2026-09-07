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

#: Timeout, in seconds, for a single call to Ollama's /api/generate. CPU-only
#: inference on a 3B model is slow, and scales with both input and output
#: length — a short JD's worth of requirements extracts in well under a
#: minute, but a full multi-bullet resume (see aptly.api.routes_resume) can
#: take several minutes to fully extract. Generous by design: a slow-but-
#: correct answer beats a fast timeout on a legitimately large input. This
#: is a client-side backstop, not the primary defense against a slow call —
#: see OLLAMA_NUM_PREDICT below for why. Set high (600s) deliberately: once
#: OLLAMA_NUM_PREDICT bounds the server side, there's no risk in waiting
#: longer client-side — a resume-extraction call was observed taking longer
#: than the previous 300s value on typical CPU throughput, timing out the
#: client even though the (now-bounded) server-side call would have
#: finished if given more time.
OLLAMA_TIMEOUT_SECONDS = 600

#: Context window size (prompt + completion, in tokens) requested from
#: Ollama per call, via the `num_ctx` generation option. Left at Ollama's
#: own default deliberately, not raised — on CPU, generation speed is
#: largely memory-bandwidth-bound, and a bigger context window means a
#: bigger KV cache read at *every* generated token, so raising this has a
#: real, measured throughput cost. An earlier version of this fix raised it
#: to 8192 to give a struggling resume-extraction prompt more headroom, but
#: that slowed down *every* call (including the previously-fast JD calls)
#: without actually fixing the underlying problem — see
#: aptly.api.routes_resume._clean_resume_text for the fix that did (shrink
#: the prompt itself, at the source, instead of giving the model more room
#: to be slow in). OLLAMA_NUM_PREDICT below is what actually prevents the
#: runaway-generation failure that motivated touching this in the first
#: place; leaving num_ctx alone keeps normal-sized calls fast.
OLLAMA_NUM_CTX = 4096

#: Hard cap, in tokens, on how much a single call is allowed to generate,
#: via the `num_predict` generation option. This is the actual fix for a
#: real observed failure: without it, a request once lost track of the JSON
#: structure it was building (after Ollama's context-shift behavior kicked
#: in near the context window limit) and never produced a natural stop
#: token — generation ran for over an hour before being killed by hand.
#: With this set, Ollama stops generating and returns whatever it has once
#: this many tokens are produced, rather than continuing until a stop token
#: that might never come — bounding the worst case regardless of prompt or
#: context-window size.
OLLAMA_NUM_PREDICT = 2048

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

# --- CORS ----------------------------------------------------------------------

#: Browser origins allowed to call this API cross-origin. Needed because the
#: intended deployment shape is: a static frontend hosted publicly (e.g. on
#: Vercel/Netlify), talking to this FastAPI backend running locally on
#: *that visitor's own machine* (http://localhost:8000) — Ollama and Chroma
#: are never hosted remotely, only the frontend is. The browser enforces
#: CORS on that cross-origin call (hosted frontend -> localhost:8000), so
#: the hosted frontend's origin must be listed here. The two local Vite dev
#: server ports are included so `npm run dev` works out of the box; add your
#: deployed frontend's URL (e.g. "https://aptly-yourname.vercel.app") once
#: you know it.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
