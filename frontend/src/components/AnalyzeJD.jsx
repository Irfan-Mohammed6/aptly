// =============================================================================
// Aptly — AI Job Search Co-Pilot
// Author: Irfan Mohammed
// License: MIT — see LICENSE file in the project root.
// =============================================================================
import { useState } from "react";

import { analyzeJD, prepList } from "../api.js";
import Icon from "./Icon.jsx";
import Thinking from "./Thinking.jsx";

const RING_RADIUS = 52;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;

/**
 * Headline and one-line summary for a fit score. The thresholds are a
 * presentation choice only — the score itself is computed deterministically
 * on the backend (`aptly.scoring.compute_fit_score`).
 *
 * @param {number} score - 0-100 fit score.
 * @param {object[]} gaps - The gap matches, used to name what's missing.
 * @param {number} total - Total number of requirements analyzed.
 * @param {number} matched - How many were strengths.
 * @returns {{ title: string, detail: string }}
 */
function describeScore(score, gaps, total, matched) {
  const title = score >= 75 ? "A strong fit" : score >= 50 ? "A solid fit with a few gaps" : "Significant gaps to close";
  const detail =
    gaps.length === 0
      ? `All ${total} requirements are covered by this resume.`
      : `${matched} of ${total} requirements matched. ${gaps.length} ${gaps.length === 1 ? "isn't" : "aren't"} covered by this resume.`;
  return { title, detail };
}

/**
 * The circular fit-score gauge: a track ring, an accent arc proportional to
 * the score, and the number in the middle.
 *
 * @param {{ score: number }} props
 */
function ScoreRing({ score }) {
  const offset = RING_CIRCUMFERENCE * (1 - score / 100);
  return (
    <div className="ring" role="img" aria-label={`Fit score ${score} out of 100`}>
      <svg width="116" height="116" viewBox="0 0 116 116" aria-hidden="true">
        <circle cx="58" cy="58" r={RING_RADIUS} fill="none" stroke="var(--ring-track)" strokeWidth="10" />
        <circle
          cx="58"
          cy="58"
          r={RING_RADIUS}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={RING_CIRCUMFERENCE}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="ring-num">
        <strong>{score}</strong>
        <span>out of 100</span>
      </div>
    </div>
  );
}

/**
 * One requirement result row: a status icon, the requirement, its supporting
 * evidence from the resume (for matches), and a badge with the verdict and
 * retrieval similarity.
 *
 * @param {{ match: object }} props - A `RequirementMatch` from the backend:
 *   { requirement, best_chunk_id, similarity, verdict, evidence }.
 */
function MatchRow({ match }) {
  const tone = match.verdict === "match" ? "good" : match.verdict === "partial" ? "warn" : "bad";
  const label = match.verdict === "match" ? "Match" : match.verdict === "partial" ? "Partial" : "Gap";
  return (
    <li className="row">
      <span className={`row-icon ${tone}`}>
        <Icon name={tone === "bad" ? "x" : "check"} size={15} />
      </span>
      <div>
        <div className="row-title">{match.requirement}</div>
        {match.evidence && <div className="row-evidence">{match.evidence}</div>}
      </div>
      <span className="badge-slot">
        <span className={`badge badge-${tone}`}>
          {label} <span className="sim">{match.similarity.toFixed(2)}</span>
        </span>
      </span>
    </li>
  );
}

/**
 * Analyze screen: paste a job description, run it against the selected
 * resume, and see the fit score with evidenced strengths and gaps plus
 * relevant prep notes.
 *
 * Calls `POST /analyze-jd` and `POST /prep-list` together (both take the same
 * JD). Both run LLM inference server-side, so the results pane shows a live
 * "Thinking" indicator; on CPU-only hardware this takes a minute or two.
 *
 * @param {{
 *   resumeState: ReturnType<import("../useResumes.js").default>,
 *   goToResumes: () => void,
 * }} props - Shared resume state, and a callback to switch to the Resumes screen.
 */
export default function AnalyzeJD({ resumeState, goToResumes }) {
  const { active, resumes, loading: resumesLoading } = resumeState;
  const [jdText, setJdText] = useState("");
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [analysis, setAnalysis] = useState(null);
  const [notes, setNotes] = useState([]);
  const [error, setError] = useState(null);
  const [analyzedWith, setAnalyzedWith] = useState(null);
  const [tookSeconds, setTookSeconds] = useState(null);

  const wordCount = jdText.trim() ? jdText.trim().split(/\s+/).length : 0;
  const noResume = !resumesLoading && resumes.length === 0;
  const canAnalyze = wordCount > 0 && status !== "loading" && !!active;

  async function handleSubmit(event) {
    event.preventDefault();
    if (!canAnalyze) return;

    setStatus("loading");
    setError(null);
    setAnalysis(null);
    setNotes([]);
    const started = Date.now();
    const resume = active;

    try {
      const [analysisResult, notesResult] = await Promise.all([analyzeJD(jdText, resume.id), prepList(jdText)]);
      setAnalysis(analysisResult);
      setNotes(notesResult.notes);
      setAnalyzedWith(resume.name);
      setTookSeconds(Math.round((Date.now() - started) / 1000));
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  function clearAll() {
    setJdText("");
    setStatus("idle");
    setAnalysis(null);
    setNotes([]);
    setError(null);
  }

  const total = analysis ? analysis.strengths.length + analysis.gaps.length : 0;
  const summary = analysis ? describeScore(analysis.fit_score, analysis.gaps, total, analysis.strengths.length) : null;

  return (
    <div className="workspace">
      <form className="card card-pad" onSubmit={handleSubmit}>
        <div className="card-head">
          <h2>Job description</h2>
        </div>

        <div className="field">
          <label htmlFor="jd" className="sr">
            Job description text
          </label>
          <textarea
            id="jd"
            className="textarea"
            rows={15}
            placeholder="Paste the full job description here…"
            value={jdText}
            onChange={(event) => setJdText(event.target.value)}
          />
        </div>

        {noResume && (
          <p className="alert alert-warn">
            You haven't added a resume yet.{" "}
            <button type="button" className="link-button" onClick={goToResumes}>
              Upload one
            </button>{" "}
            to run an analysis.
          </p>
        )}

        <div className="jd-foot">
          <span className="meta">{wordCount} words</span>
          <div className="btn-row">
            <button type="button" className="btn btn-ghost" onClick={clearAll} disabled={status === "loading"}>
              Clear
            </button>
            <button type="submit" className="btn btn-primary" disabled={!canAnalyze}>
              <Icon name="spark" size={18} />
              {status === "loading" ? "Analyzing…" : "Analyze"}
            </button>
          </div>
        </div>
      </form>

      <div className="card card-pad" aria-live="polite">
        {status === "idle" && (
          <div className="empty">
            <div className="ic">
              <Icon name="analyze" size={24} />
            </div>
            <h3>Results appear here</h3>
            <p className="muted">
              Your fit score, strengths, gaps and relevant prep notes will show up next to the job description.
            </p>
          </div>
        )}

        {status === "loading" && (
          <div className="empty">
            <Thinking label="Analyzing your fit" />
            <p className="muted">
              Extracting requirements and matching them against {active ? `“${active.name}”` : "your resume"}. This can
              take a minute or two.
            </p>
          </div>
        )}

        {status === "error" && (
          <div className="alert alert-bad">
            <strong>Analysis failed.</strong> {error}
          </div>
        )}

        {status === "done" && analysis && (
          <>
            <div className="result-top">
              <ScoreRing score={analysis.fit_score} />
              <div className="result-summary">
                <h2>{summary.title}</h2>
                <p className="muted">{summary.detail}</p>
                <div className="chips">
                  <span className="badge badge-accent">{analyzedWith}</span>
                  <span className="badge badge-neutral">{total} requirements</span>
                  {tookSeconds != null && <span className="badge badge-neutral">Analyzed in {tookSeconds}s</span>}
                </div>
              </div>
            </div>

            <div className="group">
              <div className="group-title">
                <h3>Strengths</h3>
                <span className="count">{analysis.strengths.length}</span>
              </div>
              {analysis.strengths.length === 0 ? (
                <p className="muted">No clear matches found.</p>
              ) : (
                <ul className="rows">
                  {analysis.strengths.map((match, i) => (
                    <MatchRow key={i} match={match} />
                  ))}
                </ul>
              )}
            </div>

            <div className="group">
              <div className="group-title">
                <h3>Gaps</h3>
                <span className="count">{analysis.gaps.length}</span>
              </div>
              {analysis.gaps.length === 0 ? (
                <p className="muted">No gaps found — this resume covers every extracted requirement.</p>
              ) : (
                <ul className="rows">
                  {analysis.gaps.map((match, i) => (
                    <MatchRow key={i} match={match} />
                  ))}
                </ul>
              )}
            </div>

            {notes.length > 0 && (
              <div className="group">
                <div className="group-title">
                  <h3>Prep notes for this role</h3>
                  <span className="count">{notes.length}</span>
                </div>
                <div>
                  {notes.map((note) => (
                    <div key={note.chunk_id} className="note-item">
                      <strong>{note.title}</strong>
                      <p>{note.snippet}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
