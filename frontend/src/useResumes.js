// =============================================================================
// Aptly — AI Job Search Co-Pilot
// Author: Irfan Mohammed
// License: MIT — see LICENSE file in the project root.
// =============================================================================
import { useCallback, useEffect, useState } from "react";

import { listResumes } from "./api.js";

const STORAGE_KEY = "aptly.activeResumeId";

function readStoredId() {
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null; // storage can be blocked (private windows); the app works without it
  }
}

function storeId(id) {
  try {
    if (id) window.localStorage.setItem(STORAGE_KEY, id);
  } catch {
    // ignore — see readStoredId
  }
}

/**
 * Shared resume state for the whole app: the list of uploaded resumes and
 * which one is "in use" for analysis.
 *
 * The in-use choice is remembered in localStorage so it survives a reload; if
 * the remembered resume no longer exists (deleted, or a different backend),
 * it falls back to the first available one.
 *
 * @returns {{
 *   resumes: object[],
 *   loading: boolean,
 *   error: string|null,
 *   activeId: string|null,
 *   active: object|null,
 *   setActiveId: (id: string) => void,
 *   refresh: () => Promise<object[]>,
 * }}
 */
export default function useResumes() {
  const [resumes, setResumes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeId, setActiveIdState] = useState(readStoredId);

  const refresh = useCallback(async () => {
    try {
      const list = await listResumes();
      setResumes(list);
      setError(null);
      return list;
    } catch (err) {
      setError(err.message);
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (resumes.length > 0 && !resumes.some((r) => r.id === activeId)) {
      setActiveIdState(resumes[0].id);
    }
  }, [resumes, activeId]);

  const setActiveId = useCallback((id) => {
    setActiveIdState(id);
    storeId(id);
  }, []);

  const active = resumes.find((r) => r.id === activeId) ?? null;
  return { resumes, loading, error, activeId: active?.id ?? null, active, setActiveId, refresh };
}
