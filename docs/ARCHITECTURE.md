# Aptly — Architecture

Companion to [AI_Job_Search_Copilot_POC.md](AI_Job_Search_Copilot_POC.md). That doc says *what* and *why*; this one pins down *how* — module boundaries, data contracts, and call sequences — so implementation has a fixed shape to build against.

---

## 1. Project Layout

```
aptly/
├── docs/
│   ├── AI_Job_Search_Copilot_POC.md
│   └── ARCHITECTURE.md
├── aptly/
│   ├── __init__.py
│   ├── config.py                # model names, paths, thresholds — single source of truth
│   ├── ingestion/
│   │   ├── resume.py            # load resume_chunks/*.json -> embed -> upsert into Chroma
│   │   └── notes.py             # load concept_notes/*.md -> embed -> upsert into Chroma
│   ├── llm/
│   │   ├── client.py            # Ollama HTTP wrapper: call(prompt, schema) -> validated pydantic obj, 1 retry
│   │   ├── schemas.py           # pydantic models for every LLM call's output
│   │   └── prompts.py           # prompt templates (plain f-strings, no LangChain)
│   ├── retrieval/
│   │   ├── store.py             # ChromaStore singleton — client + embedding fn + collection accessors
│   │   └── match.py             # per-requirement retrieval, gap/match verdict logic
│   ├── scoring.py                # deterministic fit-score computation from RequirementMatch list
│   ├── api/
│   │   ├── main.py                # FastAPI app, wiring, CORS
│   │   ├── routes_jd.py           # POST /analyze-jd
│   │   ├── routes_notes.py        # POST /add-note, POST /prep-list
│   │   ├── routes_resume.py       # POST /upload-resume
│   │   └── models.py              # request/response pydantic models (API boundary, distinct from llm/schemas.py)
│   └── eval/
│       └── retrieval_eval.py      # recall@k CLI script against hand-labeled fixtures
├── data/
│   ├── resume_chunks/*.json       # source of truth — hand-written, or auto-written via /upload-resume
│   ├── concept_notes/*.md         # source of truth — hand-written + grown via /add-note
│   └── chroma/                    # persisted vector index — derived, gitignored, rebuildable
├── tests/
│   ├── fixtures/labeled_jds.json  # hand-labeled JD -> expected resume_chunk_ids, for eval
│   └── test_*.py
├── scripts/
│   └── reindex.py                 # wipes + rebuilds both Chroma collections from data/*
├── frontend/                       # React (Vite) UI — see §10. Talks to the backend at
│   └── src/                        # localhost:8000 even when this frontend is hosted publicly.
├── requirements.txt
└── README.md
```

**Rule:** `data/resume_chunks/` and `data/concept_notes/` are the source of truth. `data/chroma/` is a disposable index — `scripts/reindex.py` must be able to rebuild it from nothing at any time. Never write logic that assumes Chroma has state the files don't.

---

## 2. Core Data Contracts

All schemas are Pydantic models. Two separate schema files intentionally:
- `llm/schemas.py` — shapes the model is asked to fill in (`format: json` + validation + retry)
- `api/models.py` — shapes the API exposes (may combine/reshape multiple LLM/retrieval results)

```python
# llm/schemas.py

class ExtractedRequirement(BaseModel):
    label: str                       # short name for display, e.g. "Kubernetes experience"
    detail: str                      # full phrase w/ scale/context — this is what gets embedded for
                                      # retrieval. Two fields, not one: a 3B model asked for a single
                                      # "requirement" string reliably collapses it to a bare keyword
                                      # regardless of prompting; splitting label vs. detail works with
                                      # that bias instead of fighting it.

class ExtractedRequirements(BaseModel):
    requirements: list[ExtractedRequirement]

class MatchJudgment(BaseModel):      # only asked for borderline-similarity cases
    verdict: Literal["match", "partial", "no_match"]
    evidence: str                    # quoted/paraphrased snippet from the resume chunk

class ExtractedResumeChunk(BaseModel):   # one bullet extracted from an uploaded resume PDF
    company: str
    role: str
    text: str                        # copied close to verbatim, never summarized
    tags: list[str]

class ExtractedResumeChunks(BaseModel):
    chunks: list[ExtractedResumeChunk]

# retrieval/match.py (plain dataclass/pydantic, not LLM output)

class RequirementMatch(BaseModel):
    requirement: str
    best_chunk_id: str | None
    similarity: float
    verdict: Literal["match", "partial", "gap"]
    evidence: str | None

# api/models.py

class AnalyzeJDRequest(BaseModel):
    jd_text: str

class AnalyzeJDResponse(BaseModel):
    fit_score: int                   # 0-100, computed in scoring.py — never LLM-generated
    strengths: list[RequirementMatch]
    gaps: list[RequirementMatch]

class AddNoteRequest(BaseModel):
    title: str
    tags: list[str]
    body: str

class AddNoteResponse(BaseModel):
    note_id: str

class PrepListRequest(BaseModel):
    jd_text: str

class PrepListResponse(BaseModel):
    notes: list[dict]                # title, tags, snippet, chunk_id

class UploadResumeResponse(BaseModel):
    chunks_created: int
```

---

## 3. Call Sequence — `POST /analyze-jd`

1. `jd_text` in → `llm.client.call(prompt=extract_requirements_prompt(jd_text), schema=ExtractedRequirements)`
2. For each requirement string, independently: `store.get_store().query("resume_chunks", requirement, k=TOP_K)`
3. Per requirement, look at the best (highest-similarity) hit:
   - similarity < `GAP_THRESHOLD` → `verdict="gap"`, no LLM call needed — retrieval alone proves it
   - similarity ≥ `CONFIDENT_MATCH_THRESHOLD` → `verdict="match"`, no LLM call needed — retrieval alone proves it
   - in between (borderline band) → one `llm.client.call(prompt=judge_match_prompt(...), schema=MatchJudgment)` per borderline requirement
4. Assemble `list[RequirementMatch]` → `scoring.compute_fit_score(matches)` → deterministic weighted score (e.g. `matches / total`, weighted by requirement count, no LLM involved)
5. Split matches into `strengths` (verdict ∈ {match, partial}) and `gaps` (verdict = gap) → `AnalyzeJDResponse`

Only step 1 and the borderline cases in step 3 touch the LLM. Everything else is retrieval + arithmetic — deterministic and cheap to re-run.

---

## 4. Call Sequence — `POST /prep-list`

(POST, not GET — `jd_text` can be long and doesn't belong in a query string.)

1. `jd_text` in → reuse `ExtractedRequirements` from step 1 above (same extraction call — cache/pass through if called right after `/analyze-jd`)
2. For each requirement, `store.get_store().query("concept_notes", requirement, k=TOP_K)`
3. Dedupe + rank concept notes by aggregate relevance across requirements
4. Return top N notes as `PrepListResponse`

---

## 5. Call Sequence — `POST /add-note`

1. Validate `AddNoteRequest`
2. Write `data/concept_notes/<slug>.md` to disk (frontmatter: `title`, `tags`) — **file first**
3. Embed the new file's body, upsert into the `concept_notes` Chroma collection
4. Return `note_id`

If step 3 fails, the file on disk still exists and `scripts/reindex.py` will pick it up later — the API is never the only place the note lives.

---

## 5b. Call Sequence — `POST /upload-resume`

Automates what §1's "hand-written" resume chunks otherwise require:

1. Validate the upload is a PDF (by content-type or filename extension)
2. Extract raw text from the PDF (`pypdf`, page by page, concatenated)
3. `routes_resume._clean_resume_text(resume_text)` — strip contact info, skill lists, education, and other non-bullet sections *before* the LLM sees them (see below for why this step exists)
4. `llm.client.call(prompt=extract_resume_chunks_prompt(cleaned_text), schema=ExtractedResumeChunks)` — one call splits the remaining text into per-bullet chunks
5. `ingestion.resume.write_resume_chunks(chunks)` — write each as a new `data/resume_chunks/exp_NNN.json` file, ids continuing from the highest existing number (file-first, same principle as `/add-note`)
6. `ingestion.resume.ingest_resume_chunks(load_resume_chunks())` — re-ingest the *entire* directory rather than just the new chunks, so the Chroma index can never drift from what's on disk
7. Return `UploadResumeResponse(chunks_created=...)`

**Why step 3 exists:** `pypdf` flattens a PDF's visual layout, so a two-column
resume's sidebar (skills list, contact block) comes out interleaved with the
main content as a run of short, disconnected lines. Asking the LLM to simply
*ignore* that noise (via prompt instructions alone) wasn't reliable enough —
one real run produced 37 near-empty chunks (a bare job title, a bare
LinkedIn URL, single skill words) instead of the actual ~12 achievement
bullets, and processing all that extra confusing content also meaningfully
slowed generation down. Stripping it with plain text processing, before the
prompt is even built, fixes both problems at once: fewer, better chunks, and
a smaller prompt. `_clean_resume_text` is a heuristic (a fixed list of
common section header names), not a full resume parser — see its docstring
for the known limitation and why that's an acceptable tradeoff here.

A single LLM call over a whole resume is simpler than per-bullet calls, but pushes more load onto one generation — see §6's `OLLAMA_TIMEOUT_SECONDS` for why the timeout here is generous.

---

## 6. Config (`config.py`)

Single place for every tunable, no magic numbers scattered in logic files:

```python
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT_SECONDS = 600       # generous client-side backstop — num_predict bounds the server
                                    # side now, so a long client timeout carries no hang risk
OLLAMA_NUM_CTX = 4096              # left at Ollama's default — raising it slowed every call down
                                    # (bigger KV cache = more memory bandwidth per token on CPU)
                                    # without fixing anything; the real fix was shrinking the
                                    # resume prompt itself (routes_resume._clean_resume_text)
OLLAMA_NUM_PREDICT = 2048          # hard cap on output length — prevents a real observed failure
                                    # mode: no cap let a generation that lost track of the JSON
                                    # structure run for over an hour without ever emitting a stop
                                    # token; this bounds the worst case regardless of the cause
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 256 word-piece max — never feed it a raw JD
TOP_K = 3
GAP_THRESHOLD = 0.45              # below this: no LLM call, straight to verdict=gap
CONFIDENT_MATCH_THRESHOLD = 0.75  # above this: no LLM call, straight to verdict=match
CHROMA_PERSIST_DIR = "data/chroma"
RESUME_CHUNKS_DIR = "data/resume_chunks"
CONCEPT_NOTES_DIR = "data/concept_notes"
ALLOWED_ORIGINS = ["http://localhost:5173", ...]  # CORS — see §10
```

Thresholds start as guesses — the eval script (§7) is what tunes them with real numbers instead of vibes.

---

## 7. Retrieval Eval (`eval/retrieval_eval.py`)

- Input: `tests/fixtures/labeled_jds.json` — a handful of real JDs, each with a hand-picked list of `resume_chunk_id`s a human believes *should* surface for its requirements.
- For each labeled JD: run requirement extraction + retrieval, compute recall@k (did the expected chunk appear in the top-k for its requirement?).
- Output: a plain printed table (JD, requirement, expected, retrieved, hit/miss) plus an overall recall@k number. This is the artifact that backs the "how did you evaluate retrieval quality" interview answer — keep it runnable with one command (`python -m aptly.eval.retrieval_eval`).

---

## 8. LLM Call Wrapper (`llm/client.py`)

```python
def call(prompt: str, schema: type[BaseModel]) -> BaseModel:
    """Call Ollama with format=json, validate against schema, retry once on failure."""
```

- First attempt: send `prompt`, `format: json`, parse response, validate against `schema`.
- On `ValidationError` or JSON decode failure: retry once with the original prompt plus the raw bad output and a one-line correction instruction appended.
- Second failure: raise a typed exception (`LLMOutputError`) — callers (routes) turn this into a 502 with the offending raw text attached, not a silent fallback.

---

## 9. What's deliberately *not* here

- No LangChain — Chroma's client, Ollama's HTTP API, and f-string prompts cover everything this POC needs.
- No async LLM/embedding calls — they run in a threadpool (`fastapi.concurrency.run_in_threadpool`) from otherwise-async routes so the event loop isn't blocked for 30-90s (or several minutes, for `/upload-resume`) per call.
- No auth, no multi-user, no JD scraping — out of scope per the POC doc.
- No remotely-hosted Ollama or Chroma — see §10. Multi-tenant hosting (one shared backend serving many users' resumes) would require both, plus per-user data isolation, and is explicitly out of scope.

---

## 10. Frontend & CORS

`frontend/` is a React (Vite) SPA — see docs/ARCHITECTURE.md's sibling, README.md, for setup. It is the *only* piece of this project meant to be hosted on a public platform (e.g. Vercel/Netlify). The backend, Ollama, and the Chroma index are never hosted remotely: they run locally on whoever is using the frontend, at `http://localhost:8000`.

This constraint is a direct consequence of Ollama being local-only — a hosted backend server has no way to reach `http://localhost:11434` on a *visitor's* machine ("localhost" is always relative to whichever machine is making the request). So the backend has to run wherever Ollama runs: the user's own machine, exactly as already documented for local development.

The one thing this shape requires that pure local development doesn't: **CORS**. The browser, loaded from a hosted origin (e.g. `https://aptly-yourname.vercel.app`), makes a cross-origin request to `http://localhost:8000`. `config.ALLOWED_ORIGINS` and `CORSMiddleware` in `api/main.py` are what permit that. Ollama itself needs no CORS configuration — the browser never talks to it directly, only to the local FastAPI backend, which then talks to Ollama server-to-server (not subject to CORS at all).

Practical implication: the hosted frontend is only *functional* for a visitor who has cloned the repo and is running the backend + Ollama locally per the README — not a "works instantly for any visitor" public tool. True multi-tenant hosting is out of scope (see §9).
