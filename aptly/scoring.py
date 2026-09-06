# =============================================================================
# Aptly — AI Job Search Co-Pilot
# Author: Irfan Mohammed
# License: MIT — see LICENSE file in the project root.
# =============================================================================
"""Deterministic fit-score computation.

The 0-100 fit score returned by `POST /analyze-jd` is never asked of the
LLM directly — a small model producing a raw number is uncalibrated and
non-deterministic (the same JD could score differently run to run, with no
way to defend the number). Instead, the LLM's job stops at *extraction*
(requirements) and *judgment* (per-requirement match/partial/gap verdicts,
see `aptly.retrieval.match`); the score itself is a plain, reproducible
arithmetic function over that structured output, computed here in pure
Python. See docs/ARCHITECTURE.md §3 and §6.

Run standalone: not applicable. This is a single pure function with no
side effects or external dependencies — imported by
`aptly.api.routes_jd.analyze_jd`.
"""

from aptly.retrieval.match import RequirementMatch

#: Weight contributed by each possible verdict toward the final score.
#: A "match" counts fully, a "partial" counts as half credit, and a "gap"
#: contributes nothing. These weights are a deliberately simple, legible
#: starting point — not something to over-engineer for a portfolio-scale
#: POC — but they are the one place to adjust if, say, partial matches
#: should be weighted differently.
_VERDICT_WEIGHTS = {"match": 1.0, "partial": 0.5, "gap": 0.0}


def compute_fit_score(matches: list[RequirementMatch]) -> int:
    """Compute a 0-100 fit score from a list of per-requirement match verdicts.

    The score is the mean of each match's verdict weight (see
    `_VERDICT_WEIGHTS`), scaled to a 0-100 integer range and rounded to the
    nearest whole number. For example, 4 "match" verdicts and 3 "gap"
    verdicts out of 7 total requirements yields `round(100 * 4/7) == 57`.

    This function is intentionally trivial and fully deterministic: the same
    list of matches always produces the same score, with no randomness and
    no model call involved — see the module docstring for why that matters.

    Args:
        matches: The per-requirement match results to score, typically the
            full output of `aptly.retrieval.match.match_requirements` for
            one job description (not pre-filtered into strengths/gaps).

    Returns:
        An integer in [0, 100]. Returns `0` for an empty `matches` list
        (a job description from which no requirements could be extracted)
        rather than raising a division-by-zero error.
    """
    if not matches:
        return 0
    total = sum(_VERDICT_WEIGHTS[m.verdict] for m in matches)
    return round(100 * total / len(matches))
