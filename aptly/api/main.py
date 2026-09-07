# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""The Aptly FastAPI application entry point.

Wires together the three route modules (`aptly.api.routes_jd` for
`/analyze-jd`, `aptly.api.routes_notes` for `/add-note` and `/prep-list`,
`aptly.api.routes_resume` for `/upload-resume`) into a single FastAPI `app`
object, and configures CORS. This module intentionally contains no business
logic of its own — it is pure composition.

CORS is configured because the intended deployment shape is a frontend
hosted publicly (e.g. on Vercel) making cross-origin requests to this
backend running locally on the visitor's own machine at
`http://localhost:8000` — see `config.ALLOWED_ORIGINS`.

Run standalone:

    uvicorn aptly.api.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive Swagger UI covering
all four endpoints, or see README.md for example curl commands and the
`frontend/` app for a full UI.

Requires Ollama running locally with `config.OLLAMA_MODEL` pulled, and the
Chroma collections populated via `python scripts/reindex.py` beforehand
(otherwise `/analyze-jd` and `/prep-list` will retrieve against empty
collections).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aptly import config
from aptly.api import routes_jd, routes_notes, routes_resume

#: The ASGI application object. `uvicorn aptly.api.main:app` looks up this
#: exact name — renaming it requires updating the uvicorn invocation to match.
app = FastAPI(title="Aptly — AI Job Search Co-Pilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_jd.router)
app.include_router(routes_notes.router)
app.include_router(routes_resume.router)
