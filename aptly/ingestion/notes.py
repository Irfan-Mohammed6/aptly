# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""Load, write, and embed concept notes — your personal interview-prep knowledge base.

A "concept note" is a short Markdown file with YAML frontmatter (`title`,
`tags`) and a body explaining one concept, stored in `data/concept_notes/`.
Unlike resume chunks, this collection is meant to keep growing over time —
either by hand, or via the `POST /add-note` endpoint, which calls
`write_concept_note` followed by `ingest_concept_notes` on every request.

Example note file (`data/concept_notes/rag-vs-fine-tuning.md`):

    ---
    title: RAG vs Fine-tuning
    tags: [GenAI, LLM, RAG]
    ---

    RAG retrieves relevant context at query time and injects it into the
    prompt. No weight changes...

Run standalone: not applicable. This is a library module — invoked by
`scripts/reindex.py` (command: `python scripts/reindex.py`) to bulk-load
notes from disk, and by `aptly.api.routes_notes.add_note` for single-note
writes via the API.
"""

import re
from pathlib import Path

import frontmatter

from aptly import config
from aptly.retrieval import store


def _slugify(title: str) -> str:
    """Turn a human-readable title into a filesystem- and Chroma-id-safe slug.

    Lowercases the title, replaces every run of non-alphanumeric characters
    with a single hyphen, and strips leading/trailing hyphens. Used to derive
    both the on-disk filename and the Chroma document id for a new note, so
    that "RAG vs Fine-tuning" becomes "rag-vs-fine-tuning".

    Args:
        title: The note's human-readable title, e.g. "RAG vs Fine-tuning".

    Returns:
        A slug safe to use as a filename stem or Chroma document id. Falls
        back to the literal string "note" if the title contains no
        alphanumeric characters at all (so callers never get an empty id).
    """
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "note"


def load_concept_notes() -> list[dict]:
    """Read every concept note Markdown file from disk.

    Scans `config.CONCEPT_NOTES_DIR` for `*.md` files (sorted by filename)
    and parses each one's YAML frontmatter and body using `python-frontmatter`.

    Returns:
        A list of dicts, one per note file, each with:
            - `id`: the filename stem (e.g. `rag-vs-fine-tuning`)
            - `title`: from frontmatter, falling back to the filename stem
              if the frontmatter has no `title` key
            - `tags`: from frontmatter, falling back to an empty list
            - `body`: the Markdown body below the frontmatter, whitespace-stripped

    Note:
        This function only reads from disk — it does not touch Chroma. Pair
        it with `ingest_concept_notes` to actually index the result.
    """
    notes = []
    for file in sorted(config.CONCEPT_NOTES_DIR.glob("*.md")):
        post = frontmatter.load(str(file))
        notes.append(
            {
                "id": file.stem,
                "title": post.get("title", file.stem),
                "tags": post.get("tags", []),
                "body": post.content.strip(),
            }
        )
    return notes


def write_concept_note(title: str, tags: list[str], body: str) -> Path:
    """Write a new concept note file to disk, with YAML frontmatter.

    This is deliberately "file-first": the Markdown file on disk in
    `config.CONCEPT_NOTES_DIR` is written before any embedding or Chroma
    write happens (that's the caller's job — see `ingest_concept_notes`). If
    the embedding/upsert step ever fails partway through, the note is not
    lost: it already exists on disk and `scripts/reindex.py` will pick it up
    on the next rebuild. The file is the permanent source of truth; the
    vector index is a disposable derivative of it.

    Args:
        title: Human-readable title for the note. Used to derive both the
            filename (via `_slugify`) and the frontmatter `title` field.
        tags: A list of tag strings, stored in the frontmatter as a YAML list.
        body: The Markdown body content of the note.

    Returns:
        The `Path` to the newly written file, e.g.
        `data/concept_notes/rag-vs-fine-tuning.md`. Its filename stem
        (`path.stem`) is the id that should be used as the Chroma document id
        when this note is subsequently ingested.
    """
    post = frontmatter.Post(body, title=title, tags=tags)
    path = config.CONCEPT_NOTES_DIR / f"{_slugify(title)}.md"
    with open(path, "w", encoding="utf-8") as f:
        frontmatter.dump(post, f)
    return path


def ingest_concept_notes(notes: list[dict]) -> None:
    """Embed and upsert concept notes into the `concept_notes` Chroma collection.

    For each note, the `body` field is what gets embedded; `title` and `tags`
    are stored as Chroma metadata alongside it. As with resume chunks, `tags`
    is joined into a comma-separated string because Chroma metadata values
    must be scalars, not lists.

    Always goes through the process-wide `ChromaStore` singleton
    (`aptly.retrieval.store.get_store()`) rather than constructing a fresh
    client — see docs/ARCHITECTURE.md §6.

    Args:
        notes: A list of note dicts as returned by `load_concept_notes`, or
            a single-item list built inline (as `aptly.api.routes_notes.add_note`
            does for a freshly written note). Each must have `id` and `body`
            keys; `title` and `tags` are optional.

    Returns:
        None. Side-effecting write to the collection named by
        `config.CONCEPT_NOTES_COLLECTION`. Upserting is idempotent — the same
        `id` is overwritten, not duplicated.
    """
    chroma_store = store.get_store()
    chroma_store.upsert(
        config.CONCEPT_NOTES_COLLECTION,
        [note["id"] for note in notes],
        [note["body"] for note in notes],
        [
            {"title": note["title"], "tags": ", ".join(note.get("tags", []))}
            for note in notes
        ],
    )
