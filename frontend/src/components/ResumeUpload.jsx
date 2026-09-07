// =============================================================================
// Aptly — AI Job Search Co-Pilot
// Author: Irfan Mohammed
// License: MIT — see LICENSE file in the project root.
// =============================================================================
import { useState } from "react";

import { uploadResume } from "../api.js";

/**
 * Resume upload panel: lets the visitor pick a PDF resume and send it to the
 * backend's `POST /upload-resume` endpoint, which extracts text, splits it
 * into per-bullet chunks via an LLM call, and indexes them — see
 * `aptly.api.routes_resume.upload_resume` on the backend.
 *
 * Because this triggers a blocking LLM call on CPU-only inference, this can
 * take anywhere from tens of seconds to several minutes depending on resume
 * length and hardware — the loading state is deliberately explicit about
 * that rather than implying something is broken.
 */
export default function ResumeUpload() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file) return;

    setStatus("loading");
    setError(null);
    setResult(null);

    try {
      const response = await uploadResume(file);
      setResult(response);
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  return (
    <section className="panel">
      <h2>Upload your resume</h2>
      <p className="hint">
        Upload a PDF resume. It's parsed, split into per-bullet chunks by a local LLM call, and
        indexed automatically — no manual JSON editing required. This talks to your locally
        running Aptly backend, not a hosted service.
      </p>

      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept="application/pdf"
          onChange={(event) => setFile(event.target.files[0] || null)}
        />
        <button type="submit" disabled={!file || status === "loading"}>
          {status === "loading" ? "Processing… (this can take a few minutes)" : "Upload"}
        </button>
      </form>

      {status === "error" && <p className="error">Failed: {error}</p>}

      {status === "done" && result && (
        <p className="success">
          Done — {result.chunks_created} resume chunk{result.chunks_created === 1 ? "" : "s"}{" "}
          created and indexed.
        </p>
      )}
    </section>
  );
}
