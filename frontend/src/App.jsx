// =============================================================================
// Aptly — AI Job Search Co-Pilot
// Author: Irfan Mohammed
// License: MIT — see LICENSE file in the project root.
// =============================================================================
import { useState } from "react";

import AnalyzeJD from "./components/AnalyzeJD.jsx";
import Chat from "./components/Chat.jsx";
import Icon from "./components/Icon.jsx";
import Notes from "./components/Notes.jsx";
import Resumes from "./components/Resumes.jsx";
import Select from "./components/Select.jsx";
import useResumes from "./useResumes.js";

/**
 * The app's screens, in navigation order. `short` is the label used in the
 * cramped mobile bottom bar.
 */
const VIEWS = [
  {
    id: "analyze",
    label: "Analyze a job",
    short: "Analyze",
    icon: "analyze",
    title: "Analyze a job",
    sub: "See how your resume stacks up against a job description, requirement by requirement.",
  },
  {
    id: "resumes",
    label: "Resumes",
    short: "Resumes",
    icon: "resume",
    title: "Resumes",
    sub: "Keep several versions and choose which one each analysis runs against.",
  },
  {
    id: "notes",
    label: "Prep notes",
    short: "Notes",
    icon: "note",
    title: "Prep notes",
    sub: "Your own study notes. The most relevant ones surface with each analysis.",
  },
  {
    id: "chat",
    label: "Model chat",
    short: "Chat",
    icon: "chat",
    title: "Model chat",
    sub: "A quick way to check the local model is up and see how it behaves.",
  },
];

/**
 * Root component: the app shell (sidebar on desktop, bottom tab bar on
 * mobile), the page header with the resume picker, and the four screens.
 *
 * Every screen stays mounted and the inactive ones are just hidden, so
 * in-progress work — analysis results, a chat, an upload — isn't lost when
 * you switch screens.
 *
 * This app expects the Aptly FastAPI backend (see the root README.md) to be
 * running locally on the visitor's own machine at the URL configured via
 * `VITE_API_BASE_URL` (defaulting to http://localhost:8000) — it will not
 * function for a visitor who hasn't set that up, by design (see
 * docs/ARCHITECTURE.md for why Ollama/Chroma are never hosted remotely).
 */
export default function App() {
  const [activeView, setActiveView] = useState(VIEWS[0].id);
  const resumeState = useResumes();
  const view = VIEWS.find((v) => v.id === activeView);
  const backendUp = !resumeState.error;

  const screens = {
    analyze: <AnalyzeJD resumeState={resumeState} goToResumes={() => setActiveView("resumes")} />,
    resumes: <Resumes resumeState={resumeState} />,
    notes: <Notes />,
    chat: <Chat />,
  };

  return (
    <div className="app">
      <aside className="side">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            A
          </div>
          <div>
            <div className="brand-name">Aptly</div>
            <div className="brand-sub">Job search co-pilot</div>
          </div>
        </div>

        <nav aria-label="Main">
          <div className="nav-label">Workspace</div>
          <div className="nav">
            {VIEWS.map((v) => (
              <button
                key={v.id}
                type="button"
                className="nav-btn"
                aria-current={v.id === activeView ? "page" : undefined}
                onClick={() => setActiveView(v.id)}
              >
                <Icon name={v.icon} />
                {v.label}
              </button>
            ))}
          </div>
        </nav>

        <div className="side-foot">
          <div className="status">
            <span className={backendUp ? "dot" : "dot dot-down"} />
            {backendUp ? "Backend connected" : "Backend unreachable"}
          </div>
          <div className="meta">Runs locally · no API keys</div>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div>
            <h1 className="page-title">{view.title}</h1>
            <p className="page-sub">{view.sub}</p>
          </div>
          {activeView === "analyze" && resumeState.resumes.length > 0 && (
            <Select
              label="Analyze with"
              value={resumeState.activeId ?? ""}
              options={resumeState.resumes.map((r) => ({ value: r.id, label: r.name }))}
              onChange={resumeState.setActiveId}
            />
          )}
        </header>

        <main>
          {VIEWS.map((v) => (
            <div key={v.id} hidden={v.id !== activeView}>
              {screens[v.id]}
            </div>
          ))}
        </main>
      </div>

      <nav className="bottom-nav" aria-label="Main">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            type="button"
            aria-current={v.id === activeView ? "page" : undefined}
            onClick={() => setActiveView(v.id)}
          >
            <Icon name={v.icon} size={22} />
            {v.short}
          </button>
        ))}
      </nav>
    </div>
  );
}
