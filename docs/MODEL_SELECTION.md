# Model Selection — Why `llama3.2:3b`, and What to Use Instead If You Have a GPU

This doc exists because the model choice in [`config.py`](../aptly/config.py) is a
hardware-driven decision, not a quality-driven one — worth being explicit about, since
it's the single biggest lever on both output quality and latency in this project.

---

## Why a 3B model, CPU-only

Aptly targets a specific, deliberately constrained environment: **no GPU, no API
keys, fully local**. That constraint is the project's whole point (see
[docs/AI_Job_Search_Copilot_POC.md](AI_Job_Search_Copilot_POC.md)) — but it comes at
a real cost, which this project ran into directly during development:

- A 3B-parameter model on CPU is slow. A single job-description analysis takes
  ~20-115 seconds; a full resume upload (a bigger structured-extraction task) takes
  several minutes even after tuning (see [ARCHITECTURE.md §6](ARCHITECTURE.md) for
  the `num_ctx`/`num_predict` history).
- A 3B model's instruction-following is measurably weaker on structured-output tasks.
  Two real failure modes were hit and worked around in this codebase, not
  hypothetically:
  - It reliably collapsed extracted job requirements down to bare keywords
    ("Kubernetes") regardless of prompt wording, needing a two-field
    `label`/`detail` schema split to work around (see
    `aptly.llm.schemas.ExtractedRequirement`).
  - It treated PDF-layout noise (a skills list, contact info) as genuine resume
    bullets unless that noise was stripped by plain Python *before* the prompt was
    built (see `aptly.api.routes_resume._clean_resume_text`).

None of this means the project doesn't work — it does, reliably, as verified by real
end-to-end tests (see the project's development history). It means the small-model
scaffolding in this codebase (two-field schemas, text pre-filtering, low temperature,
a hard `num_predict` cap) is a direct, load-bearing consequence of the CPU-only
constraint, not incidental complexity. A larger model would need less of it.

**Practical takeaway:** if you're running this on CPU-only hardware, `llama3.2:3b`
(the default) is a reasonable choice among similarly-sized models — see the
[CPU benchmark](#cpu-benchmark-empirical) below for how it compares to two
alternatives in its own size class. If you have GPU access, skip straight to
[If you have a GPU](#if-you-have-a-gpu-researched-not-benchmarked-here).

---

## CPU benchmark (empirical)

Run via [`aptly/eval/model_benchmark.py`](../aptly/eval/model_benchmark.py):

```bash
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
ollama pull phi3:mini
python -m aptly.eval.model_benchmark
```

This sends the same three real prompts Aptly actually uses in production
(requirement extraction, resume chunking, borderline match judgment) to each
model, using generic fictional sample data — not any real user's resume or JD — so
the results describe the *models'* behavior, not any specific person's content. Full
results are written to `docs/model_benchmark_results.json`.

<!-- BENCHMARK_RESULTS_TABLE -->
*(Results below from a single run on the author's hardware — CPU-only, no GPU. Your
numbers will vary with your CPU; re-run the script on your own machine for numbers
that reflect your actual hardware. Raw data: [`model_benchmark_results.json`](model_benchmark_results.json).)*

| Model | Task | Wall-clock (s) | Tokens/sec | Valid on first try? |
|---|---|---|---|---|
| `llama3.2:3b` | requirement_extraction | 48.2 | 8.5 | ✅ |
| `llama3.2:3b` | resume_chunking | 66.9 | 8.3 | ✅ |
| `llama3.2:3b` | match_judgment | 10.2 | 10.6 | ✅ |
| `qwen2.5:3b` | requirement_extraction | 70.9 | 3.9 | ✅ |
| `qwen2.5:3b` | resume_chunking | 66.0 | 8.5 | ✅ |
| `qwen2.5:3b` | match_judgment | 14.1 | 10.5 | ✅ |
| `phi3:mini` | requirement_extraction | **598.1** | 3.8 | ❌ |
| `phi3:mini` | resume_chunking | 144.6 | 4.1 | ✅ |
| `phi3:mini` | match_judgment | 17.6 | 7.4 | ✅ |

**Takeaways from this run:**

- **`llama3.2:3b` (the current default) won on every task** — fastest wall-clock time
  across all three, and 100% first-attempt schema validity. The original model choice
  holds up against two real alternatives, not just convenience.
- **`qwen2.5:3b` is a close second** — comparable throughput and full validity, but
  meaningfully slower on requirement extraction specifically (70.9s vs 48.2s,
  and roughly half the tokens/sec of the other two tasks). Worth re-checking on
  different hardware; this could be a quirk of this run.
- **`phi3:mini` struggled badly on requirement extraction** — 598 seconds (essentially
  the `OLLAMA_TIMEOUT_SECONDS` ceiling) and still produced invalid JSON (an
  unterminated string). Its `eval_count` in the raw results is exactly `2048` —
  it hit `OLLAMA_NUM_PREDICT`'s hard cap without ever finishing its output. This is
  the exact failure mode described in [ARCHITECTURE.md §6](ARCHITECTURE.md) (a model
  that doesn't reliably emit a stop token for this task), just caught safely this
  time — bounded at ~10 minutes instead of running forever, but still unusable output.
  `phi3:mini` did fine on the other two (shorter-output) tasks, so this looks
  specific to the longer, more structurally complex requirement-extraction prompt.

---

## If you have a GPU (researched, not benchmarked here)

The following is based on published model characteristics and general community
consensus as of early 2026, **not** empirical testing on this project's actual
prompts — take it as a starting point for your own evaluation, not a guarantee. If
you do have GPU access, re-running `model_benchmark.py` against these models (Ollama
runs GPU-accelerated models identically to CPU ones from this script's point of
view — no code changes needed, just pull a bigger model and set `OLLAMA_MODEL`) would
turn this into an empirical comparison too.

| Tier | VRAM needed (quantized) | Suggested models | Why |
|---|---|---|---|
| **Consumer GPU** | ~8-12 GB | `llama3.1:8b`, `qwen2.5:7b`, `mistral:7b` | The step up from 3B to 7-8B is where instruction-following on structured JSON tasks typically becomes noticeably more reliable — expect to need less of this project's small-model scaffolding (the `label`/`detail` split, the aggressive text pre-filtering) at this tier, though it's still worth keeping as defense-in-depth. GPU inference at this size should also be *faster* than the current CPU 3B setup despite the larger parameter count. |
| **Prosumer / workstation GPU** | ~16-24 GB | `qwen2.5:14b`, `phi3.5` (medium), `gemma2:9b`-`27b` depending on VRAM | Meaningfully stronger reasoning, likely more than this project's specific tasks need, but relevant if you extend Aptly's prompts to more complex judgment calls than requirement extraction and match scoring. |
| **Multi-GPU / cloud GPU** | 40 GB+ | `llama3.1:70b`, `qwen2.5:72b` | Well past what this use case requires — mentioned for completeness, not a realistic recommendation for a personal tool like this. |

**Qwen-family models are called out specifically** because they have a strong
community reputation for structured/function-call-style output reliability — if you
have GPU access and want to minimize the small-model workarounds in this codebase,
`qwen2.5:7b` or `qwen2.5:14b` are reasonable first things to try and re-benchmark.

**How to switch:** pull the model (`ollama pull <name>`) and change
`OLLAMA_MODEL` in [`config.py`](../aptly/config.py). Everything else in the
pipeline is model-agnostic — no other code changes required. Larger/GPU models may
tolerate a smaller `OLLAMA_NUM_PREDICT`/looser `OLLAMA_NUM_CTX` headroom than the 3B
default, but re-verify rather than assume, especially for the resume-upload task
which produces the largest output.

---

## Future: a frontend "model comparison" page

The benchmark results JSON (`docs/model_benchmark_results.json`) is meant to be
consumed by a frontend page as well as read here — see the `frontend/` app (built
separately) for where such a page would live. This doc and that page are meant to
stay in sync manually: re-run the benchmark script, update this table, and update
whatever the frontend renders, whenever you add a model or change hardware.
