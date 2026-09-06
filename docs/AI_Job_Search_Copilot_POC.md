# Aptly — AI Job Search Co-Pilot — POC Requirements Document

**Owner:** Irfan Mohammed
**Purpose:** Portfolio project + genuine daily-use tool for active job search
**Stack:** 100% open-source, self-hosted, no API keys

---

## 1. Problem Statement

**Aptly** — manually assessing JD fit and prepping for interviews is repetitive and inconsistent. This tool automates:
1. Scoring how well your experience matches a given job description (RAG-based gap analysis)
2. Generating a targeted prep list based on the JD's stack and requirements
3. Building a growing, searchable personal knowledge base from your own interview notes over time

This mirrors real production patterns you've built before (two-stage validation with retrieval-augmented LLM judgment) — same architecture, new domain.

---

## 2. Scope for POC (v1)

**In scope:**
- Ingest your resume/experience as structured, chunked text
- Ingest a JD (pasted text), extract it into atomic requirements, and retrieve matching resume chunks **per requirement**
- Fit score computed in Python from per-requirement match confidence (not asked of the LLM directly); LLM output limited to matched strengths and specific gaps
- Ingest personal concept notes (manually written .md files to start)
- Given a JD, retrieve relevant concept notes and generate a short prep list
- A small retrieval-quality eval: hand-labeled JD/resume-chunk pairs, recall@k measured and reported
- Expose everything via FastAPI endpoints (no frontend needed for POC)

**Out of scope for v1 (future iterations):**
- Auto-scraping JDs from job boards
- Quiz generation (nice-to-have, add after core loop works)
- Web UI (CLI/API calls are fine for POC)
- Multi-user support

---

## 3. Architecture

Retrieval is **not** JD-vs-resume top-k. Naive whole-JD embedding only ever returns your most-similar chunks — it structurally cannot detect that a requirement is *absent* from your resume, which is the entire point of a gap analysis. Instead, the JD is decomposed into atomic requirements first, and each requirement is retrieved against independently. A requirement with no close match *is* the gap signal, before the LLM ever judges anything.

```
                              ┌─────────────────┐
                              │  JD text         │
                              └────────┬────────┘
                                       ▼
                          ┌───────────────────────┐
                          │  LLM: extract JD into  │
                          │  atomic requirements   │
                          │  (llama3.2:3b)         │
                          └───────────┬───────────┘
                                       ▼
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────────┐
│  Resume chunks   │────▶│                  │     │ Per-requirement       │
│  (JSON/MD)       │     │   Chroma DB      │────▶│ retrieval (top-k      │
├─────────────────┤     │  (2 collections) │     │ resume chunks each)   │
│  Concept notes   │────▶│                  │     └──────────┬────────────┘
│  (MD files)      │     └──────────────────┘                │
└─────────────────┘                                          ▼
                                                    ┌─────────────────────┐
                                                    │ Low max-similarity   │
                                                    │  = gap; else LLM     │
                                                    │  judges borderline   │
                                                    │  matches (llama3.2:3b)  │
                                                    └──────────┬───────────┘
                                                                ▼
                                                    ┌─────────────────────┐
                                                    │ Score computed in    │
                                                    │ Python from weighted │
                                                    │ per-requirement      │
                                                    │ match confidence     │
                                                    └──────────┬───────────┘
                                                                ▼
                                                    ┌─────────────────┐
                                                    │  FastAPI         │
                                                    │  /analyze-jd     │
                                                    │  /prep-list      │
                                                    │  /add-note       │
                                                    └─────────────────┘
```

**Two Chroma collections:**
- `resume_chunks` — your experience broken into retrievable units (one bullet/project per chunk, with metadata: company, role, dates)
- `concept_notes` — your own study notes, one concept per chunk (title + explanation + tags like "GenAI", "coding", "cloud")

**Why the fit score isn't asked of the LLM directly:** a 3B model producing a raw 0-100 number is uncalibrated and non-deterministic — the same JD can score differently across runs, with no way to defend the number in an interview. The LLM's job is extraction (requirements, per-requirement match/no-match plus a quoted evidence span); the score itself is a deterministic weighted computation over that structured output in Python.

---

## 4. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| LLM | Ollama, `llama3.2:3b` | Matches your WSL hardware constraints (CPU-only, ~5.7GB RAM). CPU inference is 30-90s/call — run via FastAPI threadpool (`run_in_threadpool` / `asyncio.to_thread`), never awaited directly, or it stalls the event loop |
| Embeddings | `sentence-transformers`, `all-MiniLM-L6-v2` | Small, fast, CPU-friendly. **Caveat:** max sequence length is 256 word-pieces (~180 words) and truncates silently beyond that. Fine for atomic requirements and resume bullets; never embed a whole JD as one string |
| Vector DB | Chroma | Local, zero-config, persists to disk |
| Orchestration | Plain Python (Chroma client + Ollama HTTP client + f-string prompts) | No LangChain — at this scale (15-20 resume chunks, single-hop retrieval) it adds a dependency and hides the mechanics an interviewer will ask about, without earning its abstractions |
| Structured output | Pydantic schema + Ollama `format: json`, with one retry on parse failure | Small models drift out of valid JSON regularly; validate and retry rather than trust |
| API | FastAPI | Async endpoints, with blocking LLM/embedding calls off-loaded to a threadpool |
| Data format | Markdown/JSON files | Human-editable, version-controllable in git; files are the source of truth, Chroma is a rebuildable index over them |

---

## 5. Data Model

### Resume chunk (example)
```json
{
  "id": "exp_001",
  "company": "ZipLabs",
  "role": "Associate Software Engineer",
  "text": "Designed and implemented high-throughput ETL pipelines in Python, processing 25,000 records/min into MongoDB with error handling and schema validation.",
  "tags": ["ETL", "Python", "MongoDB", "data pipelines"]
}
```

### Concept note (example, markdown file)
```markdown
---
title: RAG vs Fine-tuning
tags: [GenAI, LLM, RAG]
---

RAG retrieves relevant context at query time and injects it into the prompt.
No weight changes. Fine-tuning updates model weights permanently...
```

### Extracted JD requirement (example, LLM output)
```json
{
  "requirement": "3+ years building production ETL pipelines in Python",
  "best_match_chunk_id": "exp_001",
  "similarity": 0.81,
  "verdict": "match",
  "evidence": "Designed and implemented high-throughput ETL pipelines in Python, processing 25,000 records/min into MongoDB..."
}
```
A requirement whose best-match similarity falls below threshold (e.g. Kubernetes, if never mentioned in any resume chunk) is flagged `"verdict": "gap"` without needing an LLM judgment call — the retrieval step alone surfaces it.

---

## 6. Build Order (suggested sequence)

1. **Set up environment** — Ollama installed, `llama3.2:3b` pulled, Python venv with `chromadb`, `sentence-transformers`, `fastapi`, `uvicorn`
2. **Write resume chunks** — manually break your resume into 15-20 JSON chunks (one per bullet/achievement)
3. **Build ingestion script** — embeds and loads resume chunks into Chroma
4. **Build requirement extraction** — LLM call that turns raw JD text into a list of atomic requirement strings (Pydantic-validated JSON, one retry on failure)
5. **Build `/analyze-jd` endpoint** — extracts requirements, retrieves top-k resume chunks *per requirement*, flags low-similarity ones as gaps, has the LLM judge only the borderline matches, computes the fit score in Python
6. **Test with a real JD** — use an actual listing you're evaluating, sanity-check the output quality
7. **Retrieval eval** — hand-label which resume chunks *should* match for 5 real JDs' requirements, measure recall@k while the corpus is still small enough to label by hand; this is the evidence that retrieval quality was actually checked, not assumed
8. **Write 5-10 concept notes** — start with tonight's session topics (RAG, transformers, bias-variance, etc.)
9. **Build `/prep-list` endpoint** — retrieves relevant concept notes based on JD stack keywords
10. **Add `/add-note` endpoint** — writes the note to disk *and* embeds it; include a `reindex` command that rebuilds Chroma from the on-disk files (files are the source of truth, the vector store is a disposable index)
11. **Write README** — architecture diagram, setup instructions, example output
12. **Push to GitHub**

---

## 7. Definition of Done (POC)

- [ ] Can paste any JD text and get back a fit score + 3-5 specific gaps
- [ ] Gaps are driven by per-requirement retrieval misses, not just an LLM's unstructured opinion
- [ ] Fit score is computed deterministically from structured LLM output, not asked of the LLM as a raw number
- [ ] Can retrieve relevant prep notes for a given JD's stack
- [ ] Can add a new note via API and have it immediately searchable
- [ ] Retrieval recall@k measured against a hand-labeled set of 5 JDs and reported (even informally)
- [ ] Runs fully offline (no API keys, no internet dependency after setup)
- [ ] README lets a stranger clone and run it in under 10 minutes

---

## 8. Interview Talking Points (once built)

- Why gap detection can't be top-k similarity against the whole JD — absence isn't a retrieval result, it's what's left after per-requirement retrieval comes up empty
- Retrieval earns its place here for a different reason than token savings: at 15-20 resume chunks the whole resume fits in a prompt trivially; the real payoff is per-requirement precision (so gaps are evidenced, not guessed) and the `concept_notes` collection, which keeps growing and wouldn't fit as a full prompt over time
- Chunking strategy decisions (why per-bullet vs per-role vs per-paragraph)
- How you evaluated retrieval quality (recall@k against a hand-labeled set — did the right chunks actually come back for a given JD?)
- Why the fit score is computed in Python from structured LLM output rather than asked of the LLM directly (determinism, explainability, reproducibility)
- Tradeoffs of running a small local model (llama3.2:3b) vs a larger hosted one — speed/quality/cost
- How this mirrors your production Orgchart validation pattern (two-stage: retrieval + LLM judgment)

---

## 9. Naming Conventions

- **Project name:** Aptly
- **Repo name:** `aptly` or `aptly-rag`
- **Package/module name:** `aptly`
- **FastAPI app title:** "Aptly — AI Job Search Co-Pilot"
