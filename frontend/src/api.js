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
    throw new Error(typeof detail === "string" ? detail : `${response.status}: request failed`);
  }
  if (response.status === 204) return null;
  return response.json();
}

/**
 * fetch() wrapper that turns a network failure (backend not running, CORS
 * block) into an error message a person can act on, instead of the browser's
 * bare "Failed to fetch".
 *
 * @param {string} path - Path on the API, e.g. "/resumes".
 * @param {RequestInit} [options] - Passed through to fetch.
 * @returns {Promise<Response>}
 */
async function request(path, options) {
  try {
    return await fetch(`${API_BASE_URL}${path}`, options);
  } catch {
    throw new Error(`Can't reach the Aptly backend at ${API_BASE_URL}. Is it running?`);
  }
}

/** @returns {Promise<object[]>} Every uploaded resume with its chunk count. */
export async function listResumes() {
  return parseJsonOrThrow(await request("/resumes"));
}

/**
 * @param {string} id - Resume id.
 * @returns {Promise<object[]>} That resume's chunks (id, company, role, text, tags).
 */
export async function getResumeChunks(id) {
  return parseJsonOrThrow(await request(`/resumes/${encodeURIComponent(id)}/chunks`));
}

/**
 * @param {string} id - Resume id.
 * @param {string} name - New display name.
 * @returns {Promise<object>} The updated resume.
 */
export async function renameResume(id, name) {
  return parseJsonOrThrow(
    await request(`/resumes/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    })
  );
}

/**
 * Delete a resume along with its chunks.
 *
 * Idempotent: deleting a resume that's already gone (a 404) counts as success,
 * since the end state is what the caller wanted — this matters when a slow
 * request gets clicked twice and the second one finds nothing left to delete.
 *
 * @param {string} id - Resume id.
 * @returns {Promise<null>}
 */
export async function deleteResume(id) {
  const response = await request(`/resumes/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (response.status === 404) return null;
  return parseJsonOrThrow(response);
}

/** @returns {Promise<{id: string, title: string, tags: string[], body: string}[]>} Every concept note. */
export async function listNotes() {
  return parseJsonOrThrow(await request("/notes"));
}

/**
 * Score how well the candidate's resume matches a job description.
 *
 * Calls `POST /analyze-jd`. See `aptly.api.routes_jd.analyze_jd` on the
 * backend for the full pipeline this triggers (requirement extraction,
 * per-requirement retrieval, deterministic scoring).
 *
 * @param {string} jdText - The full raw job description text.
 * @param {string|null} resumeId - Which resume to analyze against (an id from
 *   `listResumes`); null matches against every resume's chunks.
 * @returns {Promise<{fit_score: number, strengths: object[], gaps: object[]}>}
 */
export async function analyzeJD(jdText, resumeId = null) {
  const response = await request("/analyze-jd", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jd_text: jdText, resume_id: resumeId }),
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
  const response = await request("/prep-list", {
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
  const response = await request("/add-note", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, tags, body }),
  });
  return parseJsonOrThrow(response);
}

/**
 * Chat with the local model — a plain conversation, not part of the analysis
 * pipeline, used by the "Model Chat" tab to sanity-check the model.
 *
 * Calls `POST /chat`. The backend is stateless, so the full conversation so
 * far is sent every time.
 *
 * @param {{role: "user"|"assistant"|"system", content: string}[]} messages
 * @returns {Promise<{reply: string, model: string}>}
 */
export async function chat(messages) {
  const response = await request("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  return parseJsonOrThrow(response);
}

/**
 * Upload a PDF resume as a new, named resume and populate its chunks.
 *
 * Calls `POST /upload-resume` with a multipart/form-data body. See
 * `aptly.api.routes_resume.upload_resume` on the backend for the
 * extract-then-chunk-then-index pipeline this triggers. Takes minutes on
 * CPU-only hardware.
 *
 * @param {File} file - The resume file selected by the user, expected to be a PDF.
 * @param {string} name - Display name for the new resume.
 * @returns {Promise<{chunks_created: number, resume_id: string, name: string}>}
 */
export async function uploadResume(file, name) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("name", name);
  const response = await request("/upload-resume", {
    method: "POST",
    body: formData,
  });
  return parseJsonOrThrow(response);
}
