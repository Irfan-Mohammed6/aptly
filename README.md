# Aptly — AI Job Search Co-Pilot

A local, fully offline RAG tool that scores how well your resume matches a job
description — with evidenced gaps, not guesses — and builds a growing, searchable
personal knowledge base of interview prep notes. No API keys, no cloud calls for
inference: everything runs on your machine via [Ollama](https://ollama.com) and
[Chroma](https://www.trychroma.com/).

**License:** MIT (see [LICENSE](LICENSE)) · **Author:** Irfan Mohammed

---

## Table of contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Preparing your data](#preparing-your-data)
  - [Resume chunks](#resume-chunks)
  - [Concept notes](#concept-notes)
  - [Retrieval eval fixtures](#retrieval-eval-fixtures)
- [Indexing your data](#indexing-your-data)
- [Running the API](#running-the-api)
- [API reference](#api-reference)
- [Frontend](#frontend)
- [Evaluating retrieval quality](#evaluating-retrieval-quality)
- [Running each script standalone](#running-each-script-standalone)
- [Configuration](#configuration)
- [Model selection](#model-selection)
- [Known limitations](#known-limitations)
- [License](#license)

---

## How it works

A job description is not scored against your resume as one big blob of text. Naive
"embed the whole JD, retrieve top-k resume chunks" retrieval can only ever return
whatever happens to rank highest — it has no way to notice that something is
*missing*, which is the entire point of a gap analysis.

Instead:

1. The JD is decomposed into **atomic requirements** by an LLM call (e.g. "3+ years
   building ETL pipelines in Python", "Hands-on experience with Kubernetes for
   production container orchestration").
2. Each requirement is retrieved against your resume chunks **independently**.
3. A requirement whose best match is weak *is* the gap signal — found by retrieval
   alone, before any LLM judgment is needed. Only genuinely ambiguous ("borderline")
   cases get a single LLM judgment call to break the tie.
4. The final 0–100 fit score is computed **deterministically** in plain Python from
   those verdicts — it is never asked of the LLM as a raw number, which would be
   uncalibrated and non-reproducible.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full call sequences, data
contracts, and the reasoning behind every architectural choice, and
[docs/AI_Job_Search_Copilot_POC.md](docs/AI_Job_Search_Copilot_POC.md) for the original
product requirements.

---

## Project structure

```
aptly/
├── aptly/
│   ├── config.py                # every tunable constant — models, paths, thresholds
│   ├── ingestion/                # load resume chunks & concept notes from disk into Chroma
│   │   ├── resume.py
│   │   └── notes.py
│   ├── llm/                      # Ollama client, prompt templates, output schemas
│   │   ├── client.py
│   │   ├── prompts.py
│   │   └── schemas.py
│   ├── retrieval/                # the Chroma singleton + per-requirement matching logic
│   │   ├── store.py
│   │   └── match.py
│   ├── scoring.py                 # deterministic fit-score computation
│   ├── api/                       # the FastAPI application
│   │   ├── main.py                  # wiring + CORS
│   │   ├── routes_jd.py
│   │   ├── routes_notes.py
│   │   ├── routes_resume.py         # POST /upload-resume
│   │   └── models.py
│   └── eval/
│       └── retrieval_eval.py      # recall@k against hand-labeled fixtures
├── scripts/
│   └── reindex.py                 # rebuild the Chroma index from data/*
├── data/
│   ├── resume_chunks/*.json       # your resume, source of truth (hand-written or auto-uploaded)
│   ├── concept_notes/*.md         # your study notes, source of truth
│   └── chroma/                    # disposable vector index (gitignored)
├── tests/fixtures/labeled_jds.json
├── frontend/                      # React (Vite) UI — see Frontend section below
│   └── src/
└── docs/
    ├── ARCHITECTURE.md
    └── AI_Job_Search_Copilot_POC.md
```

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and runnable locally
- ~2 GB free disk space for the local LLM, plus a small amount for the embedding model
- Node.js 18+ and npm, only if you want to run the [frontend](#frontend)

---

## Setup

1. **Clone the repo and enter it:**
   ```bash
   git clone https://github.com/Irfan-Mohammed6/aptly.git
   cd aptly
   ```

2. **Install Ollama and pull the model** (one-time; requires internet the first time only):
   ```bash
   ollama pull llama3.2:3b
   ```

3. **Create a virtualenv and install Python dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Start Ollama**, if it isn't already running as a background service:
   ```bash
   ollama serve
   ```

At this point the project is installed but has no data indexed yet — see the next
section.

---

## Preparing your data

Everything under `data/` (except `data/chroma/`) is **hand-maintained and is the
source of truth** for the project. The vector index is entirely disposable and
rebuilt from these files — never edit it directly.

### Resume chunks

One JSON file per bullet/achievement, in `data/resume_chunks/`. Filename doesn't
matter (it's used as a fallback id if the JSON has none) — but a stable, descriptive
name like `exp_001.json` is recommended.

```json
{
  "id": "exp_001",
  "company": "ZipLabs",
  "role": "Associate Software Engineer",
  "text": "Designed and implemented high-throughput ETL pipelines in Python, processing 25,000 records/min into MongoDB with error handling and schema validation.",
  "tags": ["ETL", "Python", "MongoDB", "data pipelines"]
}
```

| Field | Required | Description |
|---|---|---|
| `id` | No | Stable identifier. Falls back to the filename stem if omitted. |
| `company` | No | Stored as Chroma metadata; shown in the UI/response context. |
| `role` | No | Stored as Chroma metadata. |
| `text` | **Yes** | The bullet text itself — this is what gets embedded and retrieved against. |
| `tags` | No | A list of freeform tags; stored as a comma-joined string in Chroma metadata. |

Chunk per bullet, not per role or per resume — finer granularity means a requirement
can match precisely one relevant accomplishment instead of a whole job's worth of
unrelated text diluting the similarity score.

### Concept notes

One Markdown file per concept, in `data/concept_notes/`, with YAML frontmatter:

```markdown
---
title: RAG vs Fine-tuning
tags: [GenAI, LLM, RAG]
---

RAG retrieves relevant context at query time and injects it into the prompt. No weight
changes — the base model stays frozen, and the knowledge lives in an external index
that's cheap to update...
```

| Field | Required | Description |
|---|---|---|
| `title` | No | Falls back to the filename stem if omitted from frontmatter. |
| `tags` | No | A list of freeform tags. |
| *(body)* | **Yes** | The Markdown content below the frontmatter — this is what gets embedded. |

This collection is meant to **grow over time** — add notes by hand after study
sessions, or via `POST /add-note` after each interview.

### Retrieval eval fixtures

`tests/fixtures/labeled_jds.json` — a hand-labeled mapping used only by
`aptly.eval.retrieval_eval` (see [Evaluating retrieval quality](#evaluating-retrieval-quality)),
never read by the API itself:

```json
[
  {
    "name": "ai-data-backend-role",
    "requirements": [
      { "requirement": "Experience building high-throughput ETL pipelines in Python", "expected_chunk_id": "exp_001" }
    ]
  }
]
```

For each entry, decide **by hand** which resume chunk id ought to be the top retrieval
result for that requirement text, and record it. There's no way to automate this
labeling step — that's the point of an eval fixture: it encodes a human's judgment of
correctness that the retrieval pipeline is then checked against.

---

## Indexing your data

After adding or changing anything in `data/resume_chunks/` or `data/concept_notes/`,
rebuild the Chroma index:

```bash
python scripts/reindex.py
```

This is idempotent and safe to run as often as you like — it fully deletes and
recreates both collections from the current on-disk files each time, so nothing stale
ever lingers. Expected output:

```
Reindexed 12 resume chunks.
Reindexed 3 concept notes.
```

---

## Running the API

```bash
uvicorn aptly.api.main:app --reload
```

Then visit **http://127.0.0.1:8000/docs** for interactive Swagger UI covering all
four endpoints, use `curl` as shown below, or run the [frontend](#frontend) for a full UI.

---

## API reference

### `POST /upload-resume`

Upload a PDF resume and automatically populate `data/resume_chunks/` from it — an
alternative to hand-writing the JSON files described in
[Resume chunks](#resume-chunks). Extracts the PDF's text, asks the LLM to split it
into per-bullet chunks, writes them to disk, and re-indexes.

```bash
curl -X POST http://localhost:8000/upload-resume -F "file=@/path/to/your/resume.pdf"
```

```json
{ "chunks_created": 12 }
```

This can take several minutes on CPU-only inference — a whole resume is a larger
extraction task than a single job description.

### `POST /analyze-jd`

Score how well your resume matches a job description, with evidenced strengths and gaps.

```bash
curl -X POST http://localhost:8000/analyze-jd \
  -H "Content-Type: application/json" \
  -d '{"jd_text": "We are hiring a Backend Engineer. Requirements: 3+ years building ETL pipelines in Python, experience deploying containerized REST APIs on AWS, and hands-on experience with Kubernetes in production."}'
```

```json
{
  "fit_score": 75,
  "strengths": [
    {
      "requirement": "3+ years building ETL pipelines in Python",
      "best_chunk_id": "exp_001",
      "similarity": 0.70,
      "verdict": "match",
      "evidence": "processing large volumes of records into MongoDB with error handling and schema validation."
    }
  ],
  "gaps": [
    {
      "requirement": "hands-on experience with Kubernetes in production",
      "best_chunk_id": "exp_004",
      "similarity": 0.26,
      "verdict": "gap",
      "evidence": null
    }
  ]
}
```

### `POST /prep-list`

Retrieve the most relevant concept notes for a job description's stack.

```bash
curl -X POST http://localhost:8000/prep-list \
  -H "Content-Type: application/json" \
  -d '{"jd_text": "Looking for someone who understands transformer architecture and RAG systems."}'
```

```json
{
  "notes": [
    { "chunk_id": "rag-vs-fine-tuning", "title": "RAG vs Fine-tuning", "tags": "GenAI, LLM, RAG", "snippet": "RAG retrieves relevant context..." }
  ]
}
```

### `POST /add-note`

Add a note to your personal knowledge base — written to disk and immediately searchable.

```bash
curl -X POST http://localhost:8000/add-note \
  -H "Content-Type: application/json" \
  -d '{"title": "Bias-Variance Tradeoff", "tags": ["ML fundamentals"], "body": "Bias is error from an overly simple model..."}'
```

```json
{ "note_id": "bias-variance-tradeoff" }
```

**Note on latency:** every request that touches the LLM (`/analyze-jd`, `/prep-list`)
runs entirely on local CPU inference and can take anywhere from ~20 seconds to a few
minutes depending on your hardware and how many requirements the JD contains. This is
expected — it's the tradeoff for zero API cost and full offline operation.

---

## Frontend

A React (Vite) UI lives in `frontend/`, covering all four endpoints (analyze a JD,
upload a resume, add a note) as tabs.

**Important — read before hosting this anywhere:** the frontend can be hosted
publicly (e.g. on Vercel or Netlify), but the *backend* cannot — Ollama only ever
runs locally, and a remote server has no way to reach `http://localhost:11434` on
your machine. So this frontend is built to talk to a backend running on
**your own computer** at `http://localhost:8000`, exactly as described in
[Running the API](#running-the-api), regardless of where the frontend itself is
served from. A hosted deployment of this frontend is only functional for a visitor
who has cloned this repo and is running the backend + Ollama locally — see
[docs/ARCHITECTURE.md §10](docs/ARCHITECTURE.md) for the full reasoning.

### Run locally

```bash
cd frontend
npm install
npm run dev
```

Visit **http://localhost:5173**. The backend (`uvicorn aptly.api.main:app --reload`)
must also be running — see [Running the API](#running-the-api).

### Deploy (optional)

1. Push this repo to GitHub (already done if you're reading this from the repo).
2. Import it into [Vercel](https://vercel.com) or [Netlify](https://netlify.com),
   setting the project root to `frontend/`. Both auto-detect Vite.
3. Add `frontend/.env.example`'s content as an environment variable
   (`VITE_API_BASE_URL=http://localhost:8000`) in the hosting platform's dashboard —
   it's the same value for every visitor, since it always points at *their own*
   local backend, not the hosting platform's servers.
4. Add the deployed URL (e.g. `https://aptly-yourname.vercel.app`) to
   `config.ALLOWED_ORIGINS` in `aptly/config.py` and restart your local backend —
   otherwise the browser will block the hosted frontend's requests to your
   `localhost:8000` as a CORS violation.

---

## Evaluating retrieval quality

```bash
python -m aptly.eval.retrieval_eval
```

Runs every requirement in `tests/fixtures/labeled_jds.json` against the live
`resume_chunks` collection and reports a per-requirement hit/miss table plus an
overall recall@k:

```
JD                           Requirement                                             Expected       Retrieved                       Hit?
ai-data-backend-role         Experience building high-throughput ETL pipelines...   exp_001        exp_001,exp_007,exp_010         HIT

Recall@3: 6/6 = 100.00%
```

Does not call the LLM — this measures retrieval alone. Add more labeled JDs as you use
the tool on real postings; a handful of examples is a much weaker signal than dozens.

---

## Running each script standalone

Every entry point in the project and the exact command to run it on its own:

| Script | Command | Requires Ollama? | Purpose |
|---|---|---|---|
| `aptly/api/main.py` | `uvicorn aptly.api.main:app --reload` | Yes (for `/analyze-jd`, `/prep-list`) | Starts the HTTP API. |
| `scripts/reindex.py` | `python scripts/reindex.py` | No | Rebuilds both Chroma collections from `data/*`. |
| `aptly/eval/retrieval_eval.py` | `python -m aptly.eval.retrieval_eval` | No | Prints a retrieval recall@k report. |
| `aptly/eval/model_benchmark.py` | `python -m aptly.eval.model_benchmark` | Yes (all models in `MODELS_TO_BENCHMARK`) | Benchmarks candidate models against Aptly's real prompts — see [Model selection](#model-selection). |

Every other module under `aptly/` (`config.py`, `llm/*.py`, `retrieval/*.py`,
`scoring.py`, `ingestion/*.py`, `api/routes_*.py`, `api/models.py`) is a library module
with no standalone entry point of its own — each one's module docstring says so
explicitly, along with a REPL snippet for exercising it directly if you want to poke at
it in isolation.

---

## Configuration

All tunables live in [`aptly/config.py`](aptly/config.py), fully documented inline:

| Constant | Default | What it controls |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2:3b` | Which local model handles requirement extraction and match judgment. |
| `OLLAMA_HOST` | `http://localhost:11434` | Where the Ollama server is reachable. |
| `OLLAMA_NUM_CTX` | `8192` | Context window (tokens) requested per call — see [Known limitations](#known-limitations). |
| `OLLAMA_NUM_PREDICT` | `2048` | Hard cap on generated tokens per call — prevents runaway generation. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model used for all retrieval embeddings. |
| `TOP_K` | `3` | How many nearest neighbours to retrieve per query. |
| `GAP_THRESHOLD` | `0.45` | Below this similarity, a requirement is a "gap" with no LLM call. |
| `CONFIDENT_MATCH_THRESHOLD` | `0.75` | Above this similarity, a requirement is a "match" with no LLM call. |

`GAP_THRESHOLD` and `CONFIDENT_MATCH_THRESHOLD` are starting guesses, not measured
constants — tune them against your own data using
[the retrieval eval script](#evaluating-retrieval-quality).

---

## Model selection

`llama3.2:3b` (the default in `config.py`) was chosen specifically for CPU-only, no-GPU
hardware — see [docs/MODEL_SELECTION.md](docs/MODEL_SELECTION.md) for the full
rationale, an empirical benchmark of it against two alternatives
(`qwen2.5:3b`, `phi3:mini`) on real project prompts, and researched (not yet
benchmarked here) recommendations for GPU-equipped hardware. Re-run the comparison
yourself:

```bash
ollama pull qwen2.5:3b && ollama pull phi3:mini
python -m aptly.eval.model_benchmark
```

---

## Known limitations

- **Small-model extraction quirks:** the default 3B model has a strong bias toward
  compressing extracted requirements into bare keywords unless carefully prompted —
  see `aptly.llm.schemas.ExtractedRequirement`'s docstring for how this is worked
  around (a two-field `label`/`detail` schema). Swapping in a larger model may need
  less scaffolding here, but would cost more RAM/latency.
- **Thresholds are guesses**, not tuned constants — see [Configuration](#configuration).
- **CPU-only inference is slow** — expect tens of seconds per JD analysis, and
  potentially several minutes for a full resume upload, on typical consumer hardware.
- **Runaway generation is possible, and was observed once during development:** a
  request with no `num_predict` cap and a too-small context window caused the model to
  lose track of the JSON structure it was building and never emit a stop token,
  running for over an hour before being killed by hand. `OLLAMA_NUM_CTX` and
  `OLLAMA_NUM_PREDICT` in `config.py` bound this now, but if a request seems to hang
  far longer than its usual latency, check `ollama ps` — `ollama stop <model>` unloads
  a stuck model without needing to kill the OS process.
- **No multi-user support, no auth** — this is a single-user, local-first tool by
  design (see the original scope in [docs/AI_Job_Search_Copilot_POC.md](docs/AI_Job_Search_Copilot_POC.md)).

---

## License

MIT — see [LICENSE](LICENSE). Every source file also carries a header identifying the
author; see the file headers themselves for details.
