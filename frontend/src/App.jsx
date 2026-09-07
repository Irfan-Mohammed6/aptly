// =============================================================================
// Aptly — AI Job Search Co-Pilot
// Author: Irfan Mohammed
// License: MIT — see LICENSE file in the project root.
// =============================================================================
import { useState } from "react";

import AddNote from "./components/AddNote.jsx";
import AnalyzeJD from "./components/AnalyzeJD.jsx";
import ResumeUpload from "./components/ResumeUpload.jsx";

const TABS = [
  { id: "analyze", label: "Analyze JD", Component: AnalyzeJD },
  { id: "upload", label: "Upload Resume", Component: ResumeUpload },
  { id: "note", label: "Add Note", Component: AddNote },
];

/**
 * Root application component: simple tab navigation between the three
 * feature panels, each a thin UI over one or more backend endpoints.
 *
 * This app expects the Aptly FastAPI backend (see the root README.md) to be
 * running locally on the visitor's own machine at the URL configured via
 * `VITE_API_BASE_URL` (defaulting to http://localhost:8000) — it will not
 * function for a visitor who hasn't set that up, by design (see
 * docs/ARCHITECTURE.md for why Ollama/Chroma are never hosted remotely).
 */
export default function App() {
  const [activeTab, setActiveTab] = useState(TABS[0].id);
  const ActivePanel = TABS.find((tab) => tab.id === activeTab).Component;

  return (
    <div className="app">
      <header>
        <h1>Aptly</h1>
        <p className="tagline">AI Job Search Co-Pilot — runs locally, no API keys</p>
      </header>

      <nav className="tabs">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={tab.id === activeTab ? "tab active" : "tab"}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main>
        <ActivePanel />
      </main>

      <footer>
        <p>
          Backend must be running locally — see{" "}
          <a href="https://github.com/Irfan-Mohammed6/aptly" target="_blank" rel="noreferrer">
            the README
          </a>{" "}
          for setup.
        </p>
      </footer>
    </div>
  );
}
