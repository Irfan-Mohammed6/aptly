// =============================================================================
// Aptly — AI Job Search Co-Pilot
// Author: Irfan Mohammed
// License: MIT — see LICENSE file in the project root.
// =============================================================================
import { useEffect, useRef, useState } from "react";

import { deleteResume, getResumeChunks, renameResume, uploadResume } from "../api.js";
import Icon from "./Icon.jsx";
import Thinking from "./Thinking.jsx";

/**
 * Format an ISO timestamp as a short "12 Sep" style date.
 *
 * @param {string} iso - ISO-8601 timestamp from the backend.
 * @returns {string}
 */
function shortDate(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

/**
 * Resumes screen: keep several resumes side by side and choose which one
 * analyses run against.
 *
 * - The radio button on each row sets the "in use" resume (shared app state,
 *   remembered across reloads — see `useResumes`).
 * - Uploading takes a PDF plus a name, and creates a *new* resume; existing
 *   ones are untouched. Extraction runs a local LLM call, so it shows a live
 *   "Thinking" indicator and can take a few minutes.
 * - "View" previews the bullet-level chunks extracted from a resume — what
 *   analyses actually match against.
 * - Rename and delete are inline (delete asks for confirmation on the row).
 *
 * @param {{ resumeState: ReturnType<import("../useResumes.js").default> }} props
 */
export default function Resumes({ resumeState }) {
  const { resumes, loading, error, activeId, setActiveId, refresh } = resumeState;

  const fileInput = useRef(null);
  const [previewId, setPreviewId] = useState(null);
  const [chunks, setChunks] = useState([]);
  const [chunksError, setChunksError] = useState(null);
  const [pending, setPending] = useState(null); // { file, name } chosen but not yet uploaded
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState("");
  const [confirmId, setConfirmId] = useState(null);
  const [actionError, setActionError] = useState(null);
  const [busyId, setBusyId] = useState(null); // resume with a rename/delete in flight
  const [chunksLoading, setChunksLoading] = useState(false);

  const shownId = resumes.some((r) => r.id === previewId) ? previewId : activeId;
  const shown = resumes.find((r) => r.id === shownId) ?? null;
  const shownCount = shown?.chunk_count;

  // A change to the resume list means whatever went wrong before is stale.
  useEffect(() => {
    setActionError(null);
  }, [resumes]);

  useEffect(() => {
    if (!shownId) {
      setChunks([]);
      setChunksLoading(false);
      return undefined;
    }
    let cancelled = false;
    setChunksError(null);
    setChunksLoading(true);
    getResumeChunks(shownId)
      .then((result) => !cancelled && setChunks(result))
      .catch((err) => !cancelled && setChunksError(err.message))
      .finally(() => !cancelled && setChunksLoading(false));
    return () => {
      cancelled = true;
    };
  }, [shownId, shownCount]);

  function chooseFile(file) {
    if (!file) return;
    setUploadError(null);
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Only PDF resumes are supported.");
      return;
    }
    setPending({ file, name: file.name.replace(/\.pdf$/i, "") });
  }

  async function handleUpload(event) {
    event.preventDefault();
    if (!pending || !pending.name.trim() || uploading) return;
    setUploading(true);
    setUploadError(null);
    try {
      const result = await uploadResume(pending.file, pending.name.trim());
      await refresh();
      setActiveId(result.resume_id);
      setPreviewId(result.resume_id);
      setPending(null);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function saveRename(id) {
    if (!editName.trim() || busyId) return;
    setActionError(null);
    setBusyId(id);
    try {
      await renameResume(id, editName.trim());
      await refresh();
      setEditingId(null);
    } catch (err) {
      setActionError(err.message);
    } finally {
      setBusyId(null);
    }
  }

  async function confirmDelete(id) {
    if (busyId) return; // ignore repeat clicks while a delete is already running
    setActionError(null);
    setBusyId(id);
    try {
      await deleteResume(id);
      setConfirmId(null);
      await refresh();
    } catch (err) {
      setActionError(err.message);
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="two-col">
      <div className="card card-pad">
        <div className="card-head">
          <h2>Your resumes</h2>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => fileInput.current?.click()}
            disabled={uploading}
          >
            <Icon name="upload" size={18} />
            Upload resume
          </button>
          <input
            ref={fileInput}
            type="file"
            accept="application/pdf"
            className="sr"
            tabIndex={-1}
            onChange={(event) => {
              chooseFile(event.target.files[0]);
              event.target.value = "";
            }}
          />
        </div>

        {error && <p className="alert alert-bad">{error}</p>}
        {actionError && <p className="alert alert-bad">{actionError}</p>}

        {!loading && !error && resumes.length === 0 && (
          <div className="empty">
            <div className="ic">
              <Icon name="resume" size={24} />
            </div>
            <h3>No resumes yet</h3>
            <p className="muted">Upload a PDF to get started. You can keep several versions and pick one per analysis.</p>
          </div>
        )}

        <div className="resume-list" role="radiogroup" aria-label="Resume in use">
          {resumes.map((resume) => {
            const isActive = resume.id === activeId;
            return (
              <div key={resume.id} className="resume" data-active={isActive}>
                <input
                  type="radio"
                  name="active-resume"
                  checked={isActive}
                  onChange={() => setActiveId(resume.id)}
                  aria-label={`Use ${resume.name} for analysis`}
                />
                <div className="resume-main">
                  {editingId === resume.id ? (
                    <form
                      className="inline-edit"
                      onSubmit={(event) => {
                        event.preventDefault();
                        saveRename(resume.id);
                      }}
                    >
                      <input
                        className="input"
                        value={editName}
                        onChange={(event) => setEditName(event.target.value)}
                        aria-label="Resume name"
                        autoFocus
                      />
                      <button type="submit" className="btn btn-primary btn-sm">
                        Save
                      </button>
                      <button type="button" className="btn btn-ghost btn-sm" onClick={() => setEditingId(null)}>
                        Cancel
                      </button>
                    </form>
                  ) : (
                    <div className="resume-name">
                      {resume.name}
                      {isActive && <span className="badge badge-accent">In use</span>}
                    </div>
                  )}
                  <div className="meta">
                    {resume.filename ? `${resume.filename} · ` : ""}
                    {resume.chunk_count} chunks · Updated {shortDate(resume.updated_at)}
                  </div>
                </div>
                <div className="resume-actions">
                  {confirmId === resume.id ? (
                    <>
                      <span className="meta confirm-text">
                        {busyId === resume.id ? "Deleting…" : "Delete this resume?"}
                      </span>
                      <button
                        type="button"
                        className="btn btn-quiet btn-danger"
                        onClick={() => confirmDelete(resume.id)}
                        disabled={busyId !== null}
                      >
                        Delete
                      </button>
                      <button
                        type="button"
                        className="btn btn-quiet"
                        onClick={() => setConfirmId(null)}
                        disabled={busyId !== null}
                      >
                        Keep
                      </button>
                    </>
                  ) : (
                    <>
                      <button type="button" className="btn btn-quiet" onClick={() => setPreviewId(resume.id)}>
                        View
                      </button>
                      <button
                        type="button"
                        className="btn btn-quiet"
                        onClick={() => {
                          setEditingId(resume.id);
                          setEditName(resume.name);
                        }}
                      >
                        Rename
                      </button>
                      <button type="button" className="btn btn-quiet btn-danger" onClick={() => setConfirmId(resume.id)}>
                        Delete
                      </button>
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {pending && (
          <form className="pending" onSubmit={handleUpload}>
            <div className="field">
              <label htmlFor="resume-name">Name this resume</label>
              <input
                id="resume-name"
                className="input"
                value={pending.name}
                onChange={(event) => setPending({ ...pending, name: event.target.value })}
                disabled={uploading}
              />
              <span className="meta">{pending.file.name}</span>
            </div>
            {uploading ? (
              <div className="uploading">
                <Thinking label="Extracting your bullets" />
                <p className="meta">A local model reads the PDF and splits it into chunks. This can take a few minutes.</p>
              </div>
            ) : (
              <div className="btn-row">
                <button type="submit" className="btn btn-primary" disabled={!pending.name.trim()}>
                  Add resume
                </button>
                <button type="button" className="btn btn-ghost" onClick={() => setPending(null)}>
                  Cancel
                </button>
              </div>
            )}
          </form>
        )}

        {uploadError && <p className="alert alert-bad">{uploadError}</p>}

        {!pending && (
          <button
            type="button"
            className={dragging ? "drop dragging" : "drop"}
            onClick={() => fileInput.current?.click()}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              chooseFile(event.dataTransfer.files[0]);
            }}
          >
            <Icon name="upload" size={26} />
            <strong>Drop a PDF here, or click to choose</strong>
            <span className="meta">
              Text is extracted and split into bullet-level chunks on your machine. Takes a few minutes on CPU.
            </span>
          </button>
        )}
      </div>

      <div className="card card-pad">
        <div className="card-head">
          <h2>{shown ? shown.name : "Resume preview"}</h2>
          {shown && <span className="badge badge-neutral">{shown.chunk_count} chunks</span>}
        </div>
        {!shown && <p className="muted">Select a resume to see the chunks extracted from it.</p>}
        {chunksError && <p className="alert alert-bad">{chunksError}</p>}
        {chunksLoading && <p className="muted">Loading chunks…</p>}
        <div hidden={chunksLoading}>
          {chunks.map((chunk) => (
            <div key={chunk.id} className="chunk">
              {(chunk.company || chunk.role) && (
                <div className="chunk-top">
                  {chunk.company && <span className="badge badge-neutral">{chunk.company}</span>}
                  {chunk.role && <span className="badge badge-neutral">{chunk.role}</span>}
                </div>
              )}
              {chunk.text}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
