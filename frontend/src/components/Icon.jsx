// =============================================================================
// Aptly — AI Job Search Co-Pilot
// Author: Irfan Mohammed
// License: MIT — see LICENSE file in the project root.
// =============================================================================

const ICONS = {
  analyze: {
    sw: 1.8,
    d: ["M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z", "M14 3v5h5", "M9 13h6M9 17h4"],
  },
  resume: { sw: 1.8, d: ["M12 4.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7z", "M5 20c.9-3.8 3.7-6 7-6s6.1 2.2 7 6"] },
  note: { sw: 1.8, d: ["M6 3h11a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H6z", "M9 8h6M9 12h6M9 16h3"] },
  chat: { sw: 1.8, d: ["M4 5h16v11H10l-5 4v-4H4z"] },
  check: { sw: 2.2, d: ["M5 12.5l4.5 4.5L19 7.5"] },
  chevron: { sw: 2, d: ["M6 9l6 6 6-6"] },
  x: { sw: 2.2, d: ["M6 6l12 12M18 6L6 18"] },
  upload: { sw: 1.8, d: ["M12 16V4M7 9l5-5 5 5", "M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"] },
  send: { sw: 2, d: ["M5 12h14M13 6l6 6-6 6"] },
  spark: { fill: true, d: ["M12 2l2.2 7.8L22 12l-7.8 2.2L12 22l-2.2-7.8L2 12l7.8-2.2z"] },
};

/**
 * Small inline SVG icon set (stroke icons, one filled "spark"), so the UI
 * needs no icon font or image files and icons inherit the surrounding text
 * color via `currentColor`.
 *
 * @param {{ name: keyof typeof ICONS, size?: number }} props
 */
export default function Icon({ name, size = 20 }) {
  const icon = ICONS[name];
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={icon.fill ? "currentColor" : "none"}
      stroke={icon.fill ? "none" : "currentColor"}
      strokeWidth={icon.sw}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {icon.d.map((path) => (
        <path key={path} d={path} />
      ))}
    </svg>
  );
}
