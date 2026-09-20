// =============================================================================
// Aptly — AI Job Search Co-Pilot
// Author: Irfan Mohammed
// License: MIT — see LICENSE file in the project root.
// =============================================================================
import { useEffect, useState } from "react";

import Icon from "./Icon.jsx";

/**
 * The animated "the model is working" indicator: a pulsing spark, shimmering
 * label, and a live seconds counter. The counter matters — a CPU-only local
 * model can take minutes, and a ticking timer makes it obvious the app is
 * working rather than stuck. Used for chat replies, JD analysis, and resume
 * uploads.
 *
 * The timer starts when the component mounts and stops when it unmounts, so
 * render it only while the work is in progress.
 *
 * @param {{ label?: string }} props - The text shown next to the spark.
 */
export default function Thinking({ label = "Thinking" }) {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <span className="thinking" role="status" aria-live="polite">
      <span className="spark">
        <Icon name="spark" size={18} />
      </span>
      <span className="shimmer">{label}</span>
      <span className="t">{seconds}s</span>
    </span>
  );
}
