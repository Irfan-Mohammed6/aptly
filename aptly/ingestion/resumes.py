# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""The registry of uploaded resumes.

Aptly keeps several resumes side by side, and each analysis runs against one
of them. Every resume chunk carries a `resume_id` (see
`aptly.ingestion.resume`); this module owns the small list that gives each id
a human-readable name and some bookkeeping (source filename, timestamps),
stored in `data/resumes.json` (`config.RESUMES_REGISTRY_PATH`).

Like the chunk files, the registry is a source of truth on disk — the Chroma
index is derived from it and can always be rebuilt (`scripts/reindex.py`).

Chunks written before multiple resumes were supported have no `resume_id`;
they're grouped under `config.DEFAULT_RESUME_ID`, and `list_resumes` creates
the registry entry for them automatically the first time it runs, so an
existing single-resume setup keeps working untouched.

Run standalone: not applicable. Library module, used by
`aptly.api.routes_resume`.
"""

import functools
import json
import re
import threading
from collections import Counter
from datetime import datetime, timezone

from aptly import config
from aptly.ingestion.resume import delete_chunks_for_resume, load_resume_chunks
from aptly.retrieval import store

#: Serializes every read-modify-write of the registry file. The API runs these
#: functions from a thread pool, so two requests (say, a double-clicked Delete,
#: or an upload finishing while a rename is in flight) could otherwise each
#: read the file, change it, and write it back, silently losing one of the
#: changes. Reentrant because the public functions call each other.
_lock = threading.RLock()


def _locked(fn):
    """Decorator: run `fn` while holding the registry lock."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _lock:
            return fn(*args, **kwargs)

    return wrapper


def _now() -> str:
    """Current UTC time as an ISO-8601 string, second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read() -> list[dict]:
    """Load the registry file, or an empty list if it doesn't exist yet."""
    if not config.RESUMES_REGISTRY_PATH.exists():
        return []
    with open(config.RESUMES_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write(entries: list[dict]) -> None:
    """Save the registry file."""
    config.RESUMES_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.RESUMES_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def _slugify(name: str) -> str:
    """Turn a display name into a short, URL-safe id ("AI Engineer" -> "ai-engineer")."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "resume"


@_locked
def _sync_registry() -> list[dict]:
    """Return the registry, adding entries for any resume ids found only in chunk files.

    This is what makes existing data work with no manual migration: chunk
    files with no `resume_id` count as the default resume, and any resume id
    present on disk but missing from the registry (e.g. a hand-edited chunk
    file) gets an entry rather than being invisible.

    Returns:
        The up-to-date list of registry entries (each with `id`, `name`,
        `filename`, `created_at`, `updated_at`).
    """
    entries = _read()
    known = {e["id"] for e in entries}
    changed = False
    for resume_id in dict.fromkeys(c["resume_id"] for c in load_resume_chunks()):
        if resume_id not in known:
            now = _now()
            name = "My resume" if resume_id == config.DEFAULT_RESUME_ID else resume_id
            entries.append(
                {"id": resume_id, "name": name, "filename": "", "created_at": now, "updated_at": now}
            )
            changed = True
    if changed:
        _write(entries)
    return entries


def list_resumes() -> list[dict]:
    """List every resume with its current chunk count.

    Returns:
        A list of dicts with the registry fields plus `chunk_count`, ordered
        oldest first (so an existing default resume stays at the top).
    """
    counts = Counter(c["resume_id"] for c in load_resume_chunks())
    return [{**e, "chunk_count": counts.get(e["id"], 0)} for e in _sync_registry()]


def get_resume(resume_id: str) -> dict | None:
    """Look up one resume (with its chunk count), or `None` if it doesn't exist."""
    return next((r for r in list_resumes() if r["id"] == resume_id), None)


@_locked
def create_resume(name: str, filename: str = "") -> dict:
    """Register a new resume and return its entry.

    The id is derived from the name (`"AI Engineer"` -> `ai-engineer`), with a
    numeric suffix added if that id is already taken, so two resumes with the
    same name never collide.

    Args:
        name: Display name shown in the UI.
        filename: The uploaded file's original name, kept for reference.

    Returns:
        The new entry (`id`, `name`, `filename`, `created_at`, `updated_at`).
    """
    entries = _sync_registry()
    taken = {e["id"] for e in entries}
    base = _slugify(name)
    resume_id, n = base, 2
    while resume_id in taken:
        resume_id = f"{base}-{n}"
        n += 1
    now = _now()
    entry = {"id": resume_id, "name": name.strip(), "filename": filename, "created_at": now, "updated_at": now}
    _write([*entries, entry])
    return entry


@_locked
def touch_resume(resume_id: str) -> None:
    """Set a resume's `updated_at` to now (call after its chunks change)."""
    entries = _read()
    for e in entries:
        if e["id"] == resume_id:
            e["updated_at"] = _now()
    _write(entries)


@_locked
def rename_resume(resume_id: str, name: str) -> dict | None:
    """Change a resume's display name. The id, and so every chunk's link to it, stays the same.

    Returns:
        The updated resume (with chunk count), or `None` if it doesn't exist.
    """
    entries = _sync_registry()
    found = False
    for e in entries:
        if e["id"] == resume_id:
            e["name"] = name.strip()
            e["updated_at"] = _now()
            found = True
    if not found:
        return None
    _write(entries)
    return get_resume(resume_id)


@_locked
def delete_resume(resume_id: str) -> bool:
    """Delete a resume: its registry entry, its chunk files, and its Chroma entries.

    Returns:
        `True` if the resume existed and was deleted, `False` if there was
        no such resume.
    """
    entries = _sync_registry()
    if not any(e["id"] == resume_id for e in entries):
        return False
    delete_chunks_for_resume(resume_id)
    store.get_store().delete_where(config.RESUME_COLLECTION, {"resume_id": resume_id})
    _write([e for e in entries if e["id"] != resume_id])
    return True
