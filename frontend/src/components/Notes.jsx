// =============================================================================
// Aptly — AI Job Search Co-Pilot
// Author: Irfan Mohammed
// License: MIT — see LICENSE file in the project root.
// =============================================================================
import { useCallback, useEffect, useState } from "react";

import { addNote, listNotes } from "../api.js";

/**
 * First `max` characters of a note body, cut at a word boundary, for the
 * list view.
 *
 * @param {string} body
 * @param {number} [max]
 * @returns {string}
 */
function snippet(body, max = 140) {
  const flat = body.replace(/\s+/g, " ").trim();
  if (flat.length <= max) return flat;
  return `${flat.slice(0, max).replace(/\s+\S*$/, "")}…`;
}

/**
 * Prep notes screen: write new notes and browse the ones you have.
 *
 * Saving calls `POST /add-note`, which writes the Markdown file and indexes it
 * immediately (no LLM call, so it's instant). The list comes from
 * `GET /notes`, which reads the files on disk — so it also shows notes you
 * added by hand outside the app.
 */
export default function Notes() {
  const [title, setTitle] = useState("");
  const [tags, setTags] = useState("");
  const [body, setBody] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(null);
  const [error, setError] = useState(null);
  const [notes, setNotes] = useState([]);
  const [listError, setListError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      setNotes(await listNotes());
      setListError(null);
    } catch (err) {
      setListError(err.message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!title.trim() || !body.trim() || saving) return;

    setSaving(true);
    setError(null);
    setSaved(null);
    const tagList = tags
      .split(",")
      .map((tag) => tag.trim())
      .filter(Boolean);

    try {
      const result = await addNote(title.trim(), tagList, body);
      setSaved(result.note_id);
      setTitle("");
      setTags("");
      setBody("");
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="two-col">
      <form className="card card-pad" onSubmit={handleSubmit}>
        <div className="card-head">
          <h2>New note</h2>
        </div>
        <div className="stack">
          <div className="field">
            <label htmlFor="n-title">Title</label>
            <input
              id="n-title"
              className="input"
              placeholder="e.g. RAG vs Fine-tuning"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="n-tags">Tags</label>
            <input
              id="n-tags"
              className="input"
              placeholder="GenAI, LLM, RAG"
              value={tags}
              onChange={(event) => setTags(event.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="n-body">Note</label>
            <textarea
              id="n-body"
              className="textarea"
              rows={9}
              placeholder="Write it in your own words. Markdown works."
              value={body}
              onChange={(event) => setBody(event.target.value)}
            />
          </div>
          <div className="btn-row">
            <button type="submit" className="btn btn-primary" disabled={!title.trim() || !body.trim() || saving}>
              {saving ? "Saving…" : "Save note"}
            </button>
          </div>
          {error && <p className="alert alert-bad">{error}</p>}
          {saved && <p className="alert alert-good">Saved as “{saved}”.</p>}
        </div>
      </form>

      <div className="card card-pad">
        <div className="card-head">
          <h2>Your notes</h2>
          <span className="badge badge-neutral">
            {notes.length} {notes.length === 1 ? "note" : "notes"}
          </span>
        </div>
        {listError && <p className="alert alert-bad">{listError}</p>}
        <div>
          {notes.map((note) => (
            <div key={note.id} className="note-item">
              <strong>{note.title}</strong>
              <p>{snippet(note.body)}</p>
              {note.tags.length > 0 && (
                <div className="tags">
                  {note.tags.map((tag) => (
                    <span key={tag} className="badge badge-neutral">
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
