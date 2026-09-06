# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""Wipe and rebuild both Chroma collections from the files on disk.

`data/resume_chunks/*.json` and `data/concept_notes/*.md` are the permanent
source of truth for this project (see docs/ARCHITECTURE.md §1); the Chroma
index under `data/chroma/` is a disposable, derived artifact. This script is
how you go from "I edited/added/removed files on disk" to "the vector index
reflects that" — run it any time those files change, including right after
cloning the repo for the first time.

It is safe to run repeatedly: both collections are fully deleted and
recreated each time (via `ChromaStore.reset_collection`), so there is no
risk of accumulating stale documents for chunks/notes that were since
removed from disk.

Run standalone:

    python scripts/reindex.py

Does not require Ollama — this script only performs embedding (via the
local sentence-transformers model) and Chroma writes, no LLM calls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aptly import config
from aptly.ingestion.notes import ingest_concept_notes, load_concept_notes
from aptly.ingestion.resume import ingest_resume_chunks, load_resume_chunks
from aptly.retrieval.store import get_store


def main() -> None:
    """Reset and repopulate both Chroma collections from the current on-disk files.

    Sequence:
        1. Reset `config.RESUME_COLLECTION` and `config.CONCEPT_NOTES_COLLECTION`
           (delete if present, so the next write starts from empty).
        2. Load every resume chunk from `config.RESUME_CHUNKS_DIR` and
           ingest them (skipped if the directory has no `*.json` files yet).
        3. Load every concept note from `config.CONCEPT_NOTES_DIR` and
           ingest them (skipped if the directory has no `*.md` files yet).
        4. Print a one-line summary of how many of each were indexed.

    Returns:
        None. This is a script entry point, not a library function — its
        output is the side effect of rebuilding the Chroma collections plus
        the printed summary.
    """
    chroma_store = get_store()
    chroma_store.reset_collection(config.RESUME_COLLECTION)
    chroma_store.reset_collection(config.CONCEPT_NOTES_COLLECTION)

    resume_chunks = load_resume_chunks()
    if resume_chunks:
        ingest_resume_chunks(resume_chunks)
    print(f"Reindexed {len(resume_chunks)} resume chunks.")

    concept_notes = load_concept_notes()
    if concept_notes:
        ingest_concept_notes(concept_notes)
    print(f"Reindexed {len(concept_notes)} concept notes.")


if __name__ == "__main__":
    main()
