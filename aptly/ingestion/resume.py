# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""Load resume chunks from disk and embed them into Chroma.

A "resume chunk" is one bullet/achievement from your resume, stored as a
small JSON file in `data/resume_chunks/` (see docs/ARCHITECTURE.md §5 for the
exact shape). This module is the bridge between those on-disk files — which
are the permanent source of truth for your resume content — and the
`resume_chunks` Chroma collection, which is a disposable, rebuildable index
over them.

Typical usage, e.g. from scripts/reindex.py:

    from aptly.ingestion.resume import load_resume_chunks, ingest_resume_chunks

    chunks = load_resume_chunks()
    ingest_resume_chunks(chunks)

Run standalone: not applicable. This is a library module with no
`if __name__ == "__main__"` entry point — invoke it via `scripts/reindex.py`
(command: `python scripts/reindex.py`), which is the intended way to
(re)populate the resume collection from disk.
"""

import json

from aptly import config
from aptly.retrieval import store


def load_resume_chunks() -> list[dict]:
    """Read every resume chunk JSON file from disk.

    Scans `config.RESUME_CHUNKS_DIR` for `*.json` files (sorted by filename
    for deterministic ordering) and parses each one into a dict. Each dict is
    expected to contain `company`, `role`, `text`, and `tags` keys — see the
    example in docs/ARCHITECTURE.md §5. If a file's JSON body omits an `id`
    field, the filename's stem (e.g. `exp_001` for `exp_001.json`) is used as
    a fallback so every chunk is guaranteed to have a stable identifier.

    Returns:
        A list of dicts, one per resume chunk file, each guaranteed to have
        an `id` key. Order matches the sorted filename order on disk.

    Note:
        This function only reads from disk — it does not touch Chroma or the
        network. Pair it with `ingest_resume_chunks` to actually index the
        result.
    """
    chunks = []
    for file in sorted(config.RESUME_CHUNKS_DIR.glob("*.json")):
        with open(file, "r", encoding="utf-8") as f:
            chunk = json.load(f)
            chunk.setdefault("id", file.stem)
            chunks.append(chunk)
    return chunks


def ingest_resume_chunks(chunks: list[dict]) -> None:
    """Embed and upsert resume chunks into the `resume_chunks` Chroma collection.

    For each chunk, the `text` field is what gets embedded (this is the
    string retrieval will later be matched against); `company`, `role`, and
    `tags` are stored as Chroma metadata alongside it. `tags` — a Python list
    in the source JSON — is joined into a comma-separated string here because
    Chroma's metadata values must be scalars (str/int/float/bool), not lists.

    This function always goes through the process-wide `ChromaStore`
    singleton (`aptly.retrieval.store.get_store()`), never constructing its
    own client — see docs/ARCHITECTURE.md §6 for why that matters
    (re-creating the client/embedding model per call would be expensive and
    wasteful).

    Args:
        chunks: A list of chunk dicts as returned by `load_resume_chunks`.
            Each must have `id` and `text` keys; `company`, `role`, and
            `tags` are optional and default to empty if missing.

    Returns:
        None. This is a side-effecting write to the Chroma collection named
        by `config.RESUME_COLLECTION`. Upserting is idempotent: re-ingesting
        a chunk with the same `id` overwrites it rather than duplicating it,
        which is what makes `scripts/reindex.py` safe to run repeatedly.
    """
    chroma_store = store.get_store()
    chroma_store.upsert(
        config.RESUME_COLLECTION,
        [chunk["id"] for chunk in chunks],
        [chunk["text"] for chunk in chunks],
        [
            {
                "company": chunk.get("company", ""),
                "role": chunk.get("role", ""),
                "tags": ", ".join(chunk.get("tags", [])),
            }
            for chunk in chunks
        ],
    )
