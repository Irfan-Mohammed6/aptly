// =============================================================================
// Aptly — AI Job Search Co-Pilot
// Author: Irfan Mohammed
// License: MIT — see LICENSE file in the project root.
// =============================================================================
import { useEffect, useId, useRef, useState } from "react";

import Icon from "./Icon.jsx";

/**
 * A themed dropdown, used instead of the native `<select>`.
 *
 * The native control's option list is drawn by the browser/OS (system fonts,
 * a blue highlight) and can't be reliably restyled with CSS, so it clashed
 * with the rest of the UI. This one is built from a button and a listbox
 * so it can match the theme, while keeping the behavior people expect:
 *
 * - Click the button (or press Enter, Space, or an arrow key) to open.
 * - Arrow Up/Down, Home and End move the highlight; Enter or Space picks it;
 *   Escape closes without changing anything; Tab closes and moves on.
 * - Clicking outside closes it.
 * - Follows the WAI-ARIA listbox pattern (`aria-haspopup`, `aria-expanded`,
 *   `aria-activedescendant`, `role="option"`), so screen readers announce it
 *   like a normal select.
 *
 * @param {{
 *   label: string,
 *   value: string,
 *   options: { value: string, label: string }[],
 *   onChange: (value: string) => void,
 * }} props
 */
export default function Select({ label, value, options, onChange }) {
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const rootRef = useRef(null);
  const buttonRef = useRef(null);
  const labelId = useId();
  const listId = useId();

  const selectedIndex = Math.max(
    0,
    options.findIndex((o) => o.value === value)
  );
  const selected = options[selectedIndex];

  useEffect(() => {
    if (!open) return undefined;
    function handleOutside(event) {
      if (!rootRef.current?.contains(event.target)) setOpen(false);
    }
    document.addEventListener("mousedown", handleOutside);
    return () => document.removeEventListener("mousedown", handleOutside);
  }, [open]);

  function openList() {
    setHighlight(selectedIndex);
    setOpen(true);
  }

  function choose(index) {
    onChange(options[index].value);
    setOpen(false);
    buttonRef.current?.focus();
  }

  function handleKeyDown(event) {
    if (!open) {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
        event.preventDefault();
        openList();
      }
      return;
    }
    switch (event.key) {
      case "Escape":
        event.preventDefault();
        setOpen(false);
        break;
      case "ArrowDown":
        event.preventDefault();
        setHighlight((h) => Math.min(h + 1, options.length - 1));
        break;
      case "ArrowUp":
        event.preventDefault();
        setHighlight((h) => Math.max(h - 1, 0));
        break;
      case "Home":
        event.preventDefault();
        setHighlight(0);
        break;
      case "End":
        event.preventDefault();
        setHighlight(options.length - 1);
        break;
      case "Enter":
      case " ":
        event.preventDefault();
        choose(highlight);
        break;
      case "Tab":
        setOpen(false);
        break;
      default:
    }
  }

  return (
    <div className="select" ref={rootRef}>
      <span id={labelId} className="select-label">
        {label}
      </span>
      <button
        ref={buttonRef}
        type="button"
        className="select-btn"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listId : undefined}
        aria-labelledby={labelId}
        aria-activedescendant={open ? `${listId}-${highlight}` : undefined}
        onClick={() => (open ? setOpen(false) : openList())}
        onKeyDown={handleKeyDown}
      >
        <span className="select-value">{selected?.label}</span>
        <span className="select-chevron">
          <Icon name="chevron" size={16} />
        </span>
      </button>

      {open && (
        <ul id={listId} className="select-list" role="listbox" aria-labelledby={labelId}>
          {options.map((option, index) => (
            <li
              key={option.value}
              id={`${listId}-${index}`}
              role="option"
              aria-selected={option.value === value}
              className={`select-option${index === highlight ? " highlighted" : ""}${
                option.value === value ? " selected" : ""
              }`}
              onMouseEnter={() => setHighlight(index)}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => choose(index)}
            >
              <span>{option.label}</span>
              {option.value === value && <Icon name="check" size={16} />}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
