# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""The Aptly FastAPI application entry point.

Wires together the two route modules (`aptly.api.routes_jd` for
`/analyze-jd`, `aptly.api.routes_notes` for `/add-note` and `/prep-list`)
into a single FastAPI `app` object. This module intentionally contains no
business logic of its own — it is pure composition.

Run standalone:

    uvicorn aptly.api.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive Swagger UI covering
all three endpoints, or see README.md for example curl commands.

Requires Ollama running locally with `config.OLLAMA_MODEL` pulled, and the
Chroma collections populated via `python scripts/reindex.py` beforehand
(otherwise `/analyze-jd` and `/prep-list` will retrieve against empty
collections).
"""

from fastapi import FastAPI

from aptly.api import routes_jd, routes_notes

#: The ASGI application object. `uvicorn aptly.api.main:app` looks up this
#: exact name — renaming it requires updating the uvicorn invocation to match.
app = FastAPI(title="Aptly — AI Job Search Co-Pilot")

app.include_router(routes_jd.router)
app.include_router(routes_notes.router)
