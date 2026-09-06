# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""Retrieval quality evaluation: recall@k against hand-labeled fixtures.

Gap detection in this project depends entirely on retrieval similarity being
a trustworthy signal (see docs/ARCHITECTURE.md §3) — if the resume chunk
that *should* match a requirement doesn't show up in the top-k results, no
amount of downstream LLM judgment can fix that. This script is the answer to
"how do you know retrieval actually works": it runs every requirement in
`tests/fixtures/labeled_jds.json` — a hand-picked mapping of "this
requirement should retrieve this resume chunk" — against the live
resume_chunks collection, and reports which ones hit and which missed.

The fixture file must be maintained by hand: add a JD, decide (as a human)
which resume chunk each of its requirements should retrieve, and record that
expectation. There's no way to automate the labeling itself — that's the
point of an eval fixture.

Run standalone:

    python -m aptly.eval.retrieval_eval

Requires `python scripts/reindex.py` to have been run first, so the
resume_chunks collection reflects your current `data/resume_chunks/` files.
Does not require Ollama — this script only exercises retrieval, no LLM
calls are made.
"""

import json

from aptly import config
from aptly.retrieval.store import get_store

#: Path to the hand-labeled fixture file this script evaluates against.
FIXTURES_PATH = config.BASE_DIR / "tests" / "fixtures" / "labeled_jds.json"


def main() -> None:
    """Run recall@k evaluation over every labeled JD in the fixture file and print a report.

    For each requirement in each labeled JD, queries the `resume_chunks`
    Chroma collection (via the same `ChromaStore.query` that
    `aptly.retrieval.match.match_requirements` uses in production) and
    checks whether the fixture's `expected_chunk_id` appears anywhere in the
    top `config.TOP_K` results. Prints one row per requirement (JD name,
    requirement text, expected id, actually-retrieved ids, hit/miss), then a
    final aggregate recall@k percentage across every requirement in every
    JD.

    Returns:
        None. This function's output is entirely side-effecting (printed to
        stdout) — it is a reporting script, not a library function with a
        return value for other code to consume. If `FIXTURES_PATH` contains
        an empty list, prints a message explaining how to add fixtures and
        returns without further output.
    """
    with open(FIXTURES_PATH, "r", encoding="utf-8") as f:
        labeled_jds = json.load(f)

    if not labeled_jds:
        print(f"No fixtures in {FIXTURES_PATH} — add labeled examples first (see ARCHITECTURE.md §7).")
        return

    chroma_store = get_store()
    total = 0
    hits = 0

    print(f"{'JD':<28} {'Requirement':<55} {'Expected':<14} {'Retrieved':<30} Hit?")
    for jd in labeled_jds:
        jd_name = jd.get("name", "unnamed")
        for case in jd["requirements"]:
            requirement = case["requirement"]
            expected_id = case["expected_chunk_id"]
            results = chroma_store.query(config.RESUME_COLLECTION, requirement, k=config.TOP_K)
            retrieved_ids = [r["id"] for r in results]
            hit = expected_id in retrieved_ids
            total += 1
            hits += int(hit)
            print(
                f"{jd_name:<28} {requirement:<55} {expected_id:<14} "
                f"{','.join(retrieved_ids):<30} {'HIT' if hit else 'miss'}"
            )

    recall = hits / total if total else 0.0
    print(f"\nRecall@{config.TOP_K}: {hits}/{total} = {recall:.2%}")


if __name__ == "__main__":
    main()
