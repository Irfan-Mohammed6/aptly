# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""The Chroma vector store, wrapped in a process-wide singleton.

Every other module that needs to embed or query text goes through
`get_store()` here rather than constructing its own `chromadb.PersistentClient`
or loading its own copy of the embedding model. The client and the embedding
model are both expensive to set up — the embedding model alone means loading
a neural network into memory — so doing that once per process instead of
once per call matters both for latency and for memory usage.

See docs/ARCHITECTURE.md §1 and §6 for the surrounding design rationale,
including why cosine similarity is used instead of Chroma's raw default
(squared L2 distance).

Run standalone: not applicable. This is a library module — imported by
`aptly.ingestion.resume`, `aptly.ingestion.notes`, `aptly.retrieval.match`,
`aptly.api.routes_notes`, `aptly.eval.retrieval_eval`, and
`scripts/reindex.py`. To poke at it directly, use a Python REPL:

    python -c "
    from aptly.retrieval.store import get_store
    s = get_store()
    s.upsert('scratch', ids=['a'], documents=['hello world'], metadatas=[{}])
    print(s.query('scratch', 'hello', k=1))
    "
"""

import threading
from typing import Any

from chromadb import PersistentClient
from chromadb.utils import embedding_functions

from aptly import config


class ChromaStore:
    """Singleton wrapper around one Chroma client, one embedding function, and both collections.

    Do not construct this class with the expectation of getting a fresh
    instance — `ChromaStore()` always returns the same object within a
    process (see `__new__`). Use the module-level `get_store()` function
    instead of calling `ChromaStore()` directly; it exists purely to make
    that intent explicit at call sites.

    Attributes:
        client: The underlying `chromadb.PersistentClient`, rooted at
            `config.CHROMA_PERSIST_DIR`.
        embedding_function: A `SentenceTransformerEmbeddingFunction` using
            `config.EMBEDDING_MODEL`, shared by every collection this store
            manages so that all vectors in the index come from the same
            embedding space.
        _collections: An internal cache of already-fetched Chroma collection
            objects, keyed by collection name, so repeated calls to
            `get_or_create_collection` for the same name don't repeat the
            (cheap, but non-zero) lookup.
    """

    _instance: "ChromaStore | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "ChromaStore":
        """Return the process-wide `ChromaStore` instance, creating it on first use.

        This is where singleton setup happens — deliberately not in
        `__init__`. Python calls `__init__` on every `ChromaStore()`
        invocation, even when `__new__` returns the already-cached instance;
        if the client/embedding-model setup lived in `__init__`, it would
        silently redo that expensive work on every call. Putting it here,
        guarded by `if cls._instance is None`, is what makes the singleton
        pattern actually skip the setup after the first call.

        Creation is also guarded by a lock (double-checked), because the API
        calls this from a thread pool: without it, several requests arriving
        while the embedding model is still loading each saw "no instance yet"
        and each loaded their own copy of the model — observed as the model
        loading six times over at once and the first requests taking many
        seconds. With the lock, concurrent callers wait for the one load in
        progress and then share its result.

        Returns:
            The single shared `ChromaStore` instance for this process.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    self = super().__new__(cls)
                    self.client = PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
                    self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=config.EMBEDDING_MODEL
                    )
                    self._collections: dict[str, Any] = {}
                    cls._instance = self
        return cls._instance

    def get_or_create_collection(self, name: str) -> Any:
        """Return a Chroma collection by name, creating it if it doesn't exist yet.

        Results are cached on the instance (`self._collections`) so repeated
        calls for the same `name` don't re-fetch from Chroma. Collections are
        always created with cosine similarity space explicitly
        (`metadata={"hnsw:space": "cosine"}`) rather than Chroma's raw
        default (squared L2 distance), so that `query()` can report a
        "higher is more similar" similarity score that matches the semantics
        `config.GAP_THRESHOLD` and `config.CONFIDENT_MATCH_THRESHOLD` are
        written against.

        Args:
            name: The collection name — in practice always one of
                `config.RESUME_COLLECTION` or `config.CONCEPT_NOTES_COLLECTION`,
                though this method itself is agnostic to which.

        Returns:
            The Chroma `Collection` object for `name`.
        """
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    def reset_collection(self, name: str) -> None:
        """Delete a collection (if it exists) and clear it from the local cache.

        After calling this, the next `get_or_create_collection(name)` call
        recreates the collection from scratch, empty. Used by
        `scripts/reindex.py` to guarantee a clean rebuild rather than
        accumulating stale documents from a previous run (e.g. resume
        chunks that were since deleted from `data/resume_chunks/`).

        Args:
            name: The collection name to reset.

        Returns:
            None. Deleting a collection that doesn't exist is treated as a
            no-op rather than an error, since the desired end state (an
            empty/absent collection) is already achieved.
        """
        try:
            self.client.delete_collection(name)
        except Exception:
            pass  # collection didn't exist yet — nothing to delete
        self._collections.pop(name, None)

    def upsert(self, collection_name: str, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        """Embed `documents` and write them into a collection under `ids`.

        This is an upsert, not an insert: calling it again with an id that
        already exists in the collection overwrites that document rather
        than creating a duplicate. That idempotency is what makes
        `scripts/reindex.py` safe to run repeatedly without accumulating
        stale copies.

        Args:
            collection_name: Which collection to write into (created via
                `get_or_create_collection` if it doesn't exist yet).
            ids: A list of unique string ids, one per document. Must be the
                same length as `documents` and `metadatas`.
            documents: The raw text of each document — this is what gets
                embedded by `self.embedding_function`. Same length as `ids`.
            metadatas: A dict of scalar (str/int/float/bool) metadata per
                document — Chroma does not accept list-valued metadata, so
                any list-shaped data (like resume chunk tags) must be
                flattened to a string by the caller before reaching this
                method. Same length as `ids`.

        Returns:
            None. This is a side-effecting write.
        """
        self.get_or_create_collection(collection_name).upsert(
            ids=ids, documents=documents, metadatas=metadatas
        )

    def delete_where(self, collection_name: str, where: dict) -> None:
        """Delete every document in a collection whose metadata matches a filter.

        Used when a resume is deleted, to remove exactly that resume's chunks
        from the index (`where={"resume_id": ...}`) and leave every other
        resume untouched.

        Args:
            collection_name: Which collection to delete from.
            where: A Chroma metadata filter, e.g. `{"resume_id": "ai-engineer"}`.

        Returns:
            None. Deleting when nothing matches is a no-op.
        """
        self.get_or_create_collection(collection_name).delete(where=where)

    def query(
        self,
        collection_name: str,
        text: str,
        k: int = config.TOP_K,
        where: dict | None = None,
    ) -> list[dict]:
        """Find the k nearest documents to `text` in a collection.

        Args:
            collection_name: Which collection to search.
            text: The query text. Embedded using the same embedding function
                the collection's documents were embedded with, so the
                resulting vector lives in the same space and distances are
                meaningful.
            k: How many nearest neighbours to return. Defaults to
                `config.TOP_K`.
            where: Optional Chroma metadata filter restricting which
                documents can match, e.g. `{"resume_id": "default"}` to
                search a single resume's chunks. `None` searches everything.

        Returns:
            A list of up to `k` dicts (fewer if the collection has fewer
            than `k` documents), ordered from most to least similar. Each
            dict has:
                - `id`: the document's Chroma id.
                - `document`: the original text that was embedded.
                - `metadata`: the dict of metadata attached at upsert time.
                - `similarity`: a float in roughly [0, 1] where higher means
                  more similar, computed as `1 - cosine_distance`. This
                  relies on the collection having been created with cosine
                  space (see `get_or_create_collection`) — against a
                  collection created with a different distance metric, this
                  arithmetic would not produce a meaningful similarity score.
        """
        kwargs = {"where": where} if where else {}
        result = self.get_or_create_collection(collection_name).query(
            query_texts=[text], n_results=k, **kwargs
        )
        return [
            {"id": id_, "document": document, "metadata": metadata, "similarity": 1 - distance}
            for id_, document, metadata, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]


def get_store() -> ChromaStore:
    """Return the process-wide `ChromaStore` singleton.

    This is the intended entry point for every other module — prefer
    `get_store()` over calling `ChromaStore()` directly, purely as a
    readability convention (both are equivalent, since `ChromaStore.__new__`
    already enforces the singleton behavior).

    Returns:
        The single shared `ChromaStore` instance for this process. The same
        object is returned on every call.
    """
    return ChromaStore()
