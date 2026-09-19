# Running Aptly — Frontend & Backend

A single reference for every command needed to run the project locally. See
[README.md](../README.md) for first-time setup (installing Ollama, pulling the
model, Python venv) — this doc assumes that's already done once.

---

## Every time: start these three things, in order

### 1. Ollama

```bash
ollama serve
```

Skip this if it's already running as a background service — check with:

```bash
curl -s http://localhost:11434/api/tags
```

If that returns JSON (even `{"models":[]}`), it's already up.

### 2. Backend (FastAPI)

```bash
cd aptly   # repo root
source .venv/bin/activate      # Windows: .venv\Scripts\activate
uvicorn aptly.api.main:app --reload
```

Runs at **http://localhost:8000**. Visit `/docs` for interactive Swagger UI.
Requires `data/resume_chunks/` and `data/concept_notes/` to already be indexed —
run `python scripts/reindex.py` once beforehand if you haven't (see README.md).

### 3. Frontend (Vite/React)

```bash
cd frontend
npm install     # only needed once, or after pulling changes to package.json
npm run dev
```

Runs at **http://localhost:5173**.

Then open **http://localhost:5173** in a browser. The frontend talks to the
backend at `http://localhost:8000` by default (`frontend/.env.example` documents
the `VITE_API_BASE_URL` variable if you need to point it elsewhere).

---

## Windows + WSL note

If your project lives under a WSL path (`\\wsl.localhost\...` or `\\wsl$\...`) and
you're running commands from Windows PowerShell rather than inside WSL directly,
Vite's dev server and `npm`/`esbuild` can hit path-resolution bugs across that
boundary (UNC paths, ESM `file://` URL resolution, platform-specific native
binaries). The most reliable fix is to **not cross the boundary**: install Node.js
directly inside WSL (`sudo apt install nodejs npm`, or use
[nvm](https://github.com/nvm-sh/nvm)) and run every command in this doc from a WSL
shell (`wsl` from PowerShell, or Windows Terminal's WSL profile), operating on the
project's native Linux path (`~/ws/aptly` or wherever you cloned it), not the
`\\wsl.localhost\...` Windows-side view of it. This avoids the whole class of
cross-filesystem issues rather than working around them one at a time.

---

## Building the frontend for deployment

```bash
cd frontend
npm run build
```

Outputs static files to `frontend/dist/` — see the root [README.md](../README.md#frontend)
for hosting instructions (Vercel/Netlify), and remember: only the frontend is meant
to be hosted publicly. The backend, Ollama, and the Chroma index always run locally
on whoever is using the app — see [ARCHITECTURE.md §10](ARCHITECTURE.md) for why.

---

## Stopping everything

Ctrl+C in each terminal running `ollama serve`, `uvicorn`, and `npm run dev`
respectively. There's no shared shutdown command since each is an independent
process — this is a local-first tool, not an orchestrated stack.

---

## Quick troubleshooting

| Symptom | Likely cause |
|---|---|
| Frontend loads but every action fails/spins forever | Backend isn't running, or Ollama isn't running (backend needs Ollama for `/analyze-jd`, `/prep-list`, `/upload-resume`) |
| `CORS` error in the browser console | The frontend's origin isn't in `config.ALLOWED_ORIGINS` — see `aptly/config.py` |
| `/upload-resume` takes several minutes | Expected on CPU-only hardware for a full resume extraction — see [MODEL_SELECTION.md](MODEL_SELECTION.md) for why, and the "Known limitations" section of the README for the runaway-generation history |
| A request seems to hang far longer than usual | Check `ollama ps` — `ollama stop <model>` unloads a stuck model without needing to kill the OS process |
