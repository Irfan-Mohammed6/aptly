// =============================================================================
// Aptly — AI Job Search Co-Pilot
// Author: Irfan Mohammed
// License: MIT — see LICENSE file in the project root.
// =============================================================================
/**
 * Thin fetch wrapper around the Aptly FastAPI backend.
 *
 * Every function here calls a backend that is expected to be running locally
 * on the *visitor's own machine* at `API_BASE_URL` (http://localhost:8000 by
 * default) — this frontend may be hosted publicly, but the backend, Ollama,
 * and the Chroma index are not; see the root README.md for why. Requests
 * that trigger an LLM call server-side (analyzeJD, prepList, uploadResume)
 * can take anywhere from ~20 seconds to several minutes on CPU-only
 * inference — callers should show a clear loading state rather than assuming
 * a fast response.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/**
 * Parse a fetch Response as JSON, raising a descriptive Error on a non-2xx status.
 *
 * @param {Response} response - The fetch Response to parse.
 * @returns {Promise<any>} The parsed JSON body.
 * @throws {Error} If the response status is not ok; the error message
 *   includes the backend's `detail` field when present (FastAPI's standard
 *   error shape), falling back to the raw status text otherwise.
 */
async function parseJsonOrThrow(response) {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // Response body wasn't JSON — fall back to statusText, already set above.
    }
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json();
}

/**
 * Score how well the candidate's resume matches a job description.
 *
 * Calls `POST /analyze-jd`. See `aptly.api.routes_jd.analyze_jd` on the
 * backend for the full pipeline this triggers (requirement extraction,
 * per-requirement retrieval, deterministic scoring).
 *
 * @param {string} jdText - The full raw job description text.
 * @returns {Promise<{fit_score: number, strengths: object[], gaps: object[]}>}
 */
export async function analyzeJD(jdText) {
  const response = await fetch(`${API_BASE_URL}/analyze-jd`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jd_text: jdText }),
  });
  return parseJsonOrThrow(response);
}

/**
 * Retrieve the most relevant concept notes for a job description's stack.
 *
 * Calls `POST /prep-list`.
 *
 * @param {string} jdText - The full raw job description text.
 * @returns {Promise<{notes: object[]}>}
 */
export async function prepList(jdText) {
  const response = await fetch(`${API_BASE_URL}/prep-list`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jd_text: jdText }),
  });
  return parseJsonOrThrow(response);
}

/**
 * Add a new note to the personal knowledge base.
 *
 * Calls `POST /add-note`. The note is written to disk and immediately
 * searchable on the backend — no separate reindex step needed.
 *
 * @param {string} title - Human-readable title for the note.
 * @param {string[]} tags - A list of freeform tag strings.
 * @param {string} body - The Markdown body content of the note.
 * @returns {Promise<{note_id: string}>}
 */
export async function addNote(title, tags, body) {
  const response = await fetch(`${API_BASE_URL}/add-note`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, tags, body }),
  });
  return parseJsonOrThrow(response);
}

/**
 * Upload a PDF resume and automatically populate resume chunks from it.
 *
 * Calls `POST /upload-resume` with a multipart/form-data body. See
 * `aptly.api.routes_resume.upload_resume` on the backend for the
 * extract-then-chunk-then-index pipeline this triggers.
 *
 * @param {File} file - The resume file selected by the user, expected to be a PDF.
 * @returns {Promise<{chunks_created: number}>}
 */
export async function uploadResume(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE_URL}/upload-resume`, {
    method: "POST",
    body: formData,
  });
  return parseJsonOrThrow(response);
}
