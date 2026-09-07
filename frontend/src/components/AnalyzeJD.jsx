// =============================================================================
// Aptly — AI Job Search Co-Pilot
// Author: Irfan Mohammed
// License: MIT — see LICENSE file in the project root.
// =============================================================================
import { useState } from "react";

import { analyzeJD, prepList } from "../api.js";

/**
 * One row in the strengths/gaps list — a single `RequirementMatch` from the backend.
 *
 * @param {{ match: object }} props - `match` is a RequirementMatch dict:
 *   { requirement, best_chunk_id, similarity, verdict, evidence }.
 */
function MatchRow({ match }) {
  return (
    <li className={`match-row match-${match.verdict}`}>
      <div className="match-requirement">{match.requirement}</div>
      <div className="match-meta">
        similarity {match.similarity.toFixed(2)}
        {match.best_chunk_id && ` · closest chunk: ${match.best_chunk_id}`}
      </div>
      {match.evidence && <div className="match-evidence">"{match.evidence}"</div>}
    </li>
  );
}

/**
 * Job description analysis panel: pastes a JD, gets back a fit score with
 * evidenced strengths/gaps (`POST /analyze-jd`) and relevant prep notes
 * (`POST /prep-list`) for the same JD, fetched together since both endpoints
 * take the same input. See `aptly.api.routes_jd.analyze_jd` and
 * `aptly.api.routes_notes.prep_list` on the backend.
 *
 * Both calls trigger LLM inference server-side and can take anywhere from
 * ~20 seconds to a few minutes on CPU-only hardware — the loading state
 * reflects that explicitly.
 */
export default function AnalyzeJD() {
  const [jdText, setJdText] = useState("");
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [analysis, setAnalysis] = useState(null);
  const [notes, setNotes] = useState(null);
  const [error, setError] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!jdText.trim()) return;

    setStatus("loading");
    setError(null);
    setAnalysis(null);
    setNotes(null);

    try {
      const [analysisResult, notesResult] = await Promise.all([analyzeJD(jdText), prepList(jdText)]);
      setAnalysis(analysisResult);
      setNotes(notesResult);
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  return (
    <section className="panel">
      <h2>Analyze a job description</h2>
      <p className="hint">
        Paste a job description below. Aptly extracts its requirements, matches each one against
        your indexed resume chunks, and surfaces relevant prep notes from your knowledge base.
      </p>

      <form onSubmit={handleSubmit}>
        <textarea
          rows={10}
          placeholder="Paste the full job description here…"
          value={jdText}
          onChange={(event) => setJdText(event.target.value)}
        />
        <button type="submit" disabled={!jdText.trim() || status === "loading"}>
          {status === "loading" ? "Analyzing… (this can take a minute or two)" : "Analyze"}
        </button>
      </form>

      {status === "error" && <p className="error">Failed: {error}</p>}

      {status === "done" && analysis && (
        <div className="results">
          <h3>
            Fit score: <span className="fit-score">{analysis.fit_score}</span>/100
          </h3>

          <h4>Strengths ({analysis.strengths.length})</h4>
          {analysis.strengths.length === 0 ? (
            <p className="hint">No clear matches found.</p>
          ) : (
            <ul>
              {analysis.strengths.map((match, i) => (
                <MatchRow key={i} match={match} />
              ))}
            </ul>
          )}

          <h4>Gaps ({analysis.gaps.length})</h4>
          {analysis.gaps.length === 0 ? (
            <p className="hint">No gaps found — your resume covers every extracted requirement.</p>
          ) : (
            <ul>
              {analysis.gaps.map((match, i) => (
                <MatchRow key={i} match={match} />
              ))}
            </ul>
          )}

          {notes && notes.notes.length > 0 && (
            <>
              <h4>Relevant prep notes</h4>
              <ul>
                {notes.notes.map((note) => (
                  <li key={note.chunk_id} className="note-row">
                    <strong>{note.title}</strong>
                    {note.tags && <span className="match-meta"> · {note.tags}</span>}
                    <div className="match-evidence">{note.snippet}</div>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </section>
  );
}
