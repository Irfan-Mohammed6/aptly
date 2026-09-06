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
│   │   ├── main.py                # FastAPI app, wiring
│   │   ├── routes_jd.py           # POST /analyze-jd
│   │   ├── routes_notes.py        # POST /add-note, GET /prep-list
│   │   └── models.py              # request/response pydantic models (API boundary, distinct from llm/schemas.py)
│   └── eval/
│       └── retrieval_eval.py      # recall@k CLI script against hand-labeled fixtures
├── data/
│   ├── resume_chunks/*.json       # source of truth — hand-written
│   ├── concept_notes/*.md         # source of truth — hand-written + grown via /add-note
│   └── chroma/                    # persisted vector index — derived, gitignored, rebuildable
├── tests/
│   ├── fixtures/labeled_jds.json  # hand-labeled JD -> expected resume_chunk_ids, for eval
│   └── test_*.py
├── scripts/
│   └── reindex.py                 # wipes + rebuilds both Chroma collections from data/*
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

## 6. Config (`config.py`)

Single place for every tunable, no magic numbers scattered in logic files:

```python
OLLAMA_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 256 word-piece max — never feed it a raw JD
TOP_K = 3
GAP_THRESHOLD = 0.45              # below this: no LLM call, straight to verdict=gap
CONFIDENT_MATCH_THRESHOLD = 0.75  # above this: no LLM call, straight to verdict=match
CHROMA_PERSIST_DIR = "data/chroma"
RESUME_CHUNKS_DIR = "data/resume_chunks"
CONCEPT_NOTES_DIR = "data/concept_notes"
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
- No async LLM/embedding calls — they run in a threadpool (`fastapi.concurrency.run_in_threadpool`) from otherwise-async routes so the event loop isn't blocked for 30-90s per call.
- No auth, no multi-user, no JD scraping — out of scope per the POC doc.
