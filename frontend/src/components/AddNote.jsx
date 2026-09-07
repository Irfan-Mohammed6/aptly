// =============================================================================
// Aptly — AI Job Search Co-Pilot
// Author: Irfan Mohammed
// License: MIT — see LICENSE file in the project root.
// =============================================================================
import { useState } from "react";

import { addNote } from "../api.js";

/**
 * Add-note panel: lets the visitor grow their personal knowledge base by
 * writing a new concept note directly from the browser, calling
 * `POST /add-note`. Unlike resume analysis, this does not trigger an LLM
 * call — it's a fast, synchronous write-to-disk-then-embed operation. See
 * `aptly.api.routes_notes.add_note` on the backend.
 */
export default function AddNote() {
  const [title, setTitle] = useState("");
  const [tags, setTags] = useState("");
  const [body, setBody] = useState("");
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!title.trim() || !body.trim()) return;

    setStatus("loading");
    setError(null);
    setResult(null);

    const tagList = tags
      .split(",")
      .map((tag) => tag.trim())
      .filter(Boolean);

    try {
      const response = await addNote(title, tagList, body);
      setResult(response);
      setStatus("done");
      setTitle("");
      setTags("");
      setBody("");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  return (
    <section className="panel">
      <h2>Add a prep note</h2>
      <p className="hint">
        Add a note to your personal knowledge base — it's written to disk and immediately
        searchable for future job description analyses.
      </p>

      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Title, e.g. &quot;RAG vs Fine-tuning&quot;"
          value={title}
          onChange={(event) => setTitle(event.target.value)}
        />
        <input
          type="text"
          placeholder="Tags, comma-separated, e.g. GenAI, LLM, RAG"
          value={tags}
          onChange={(event) => setTags(event.target.value)}
        />
        <textarea
          rows={8}
          placeholder="Note content (Markdown)…"
          value={body}
          onChange={(event) => setBody(event.target.value)}
        />
        <button type="submit" disabled={!title.trim() || !body.trim() || status === "loading"}>
          {status === "loading" ? "Saving…" : "Add note"}
        </button>
      </form>

      {status === "error" && <p className="error">Failed: {error}</p>}
      {status === "done" && result && <p className="success">Saved as "{result.note_id}".</p>}
    </section>
  );
}
