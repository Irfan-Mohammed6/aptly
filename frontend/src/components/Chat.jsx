// =============================================================================
// Aptly — AI Job Search Co-Pilot
// Author: Irfan Mohammed
// License: MIT — see LICENSE file in the project root.
// =============================================================================
import { useEffect, useRef, useState } from "react";

import { chat } from "../api.js";
import Icon from "./Icon.jsx";
import Thinking from "./Thinking.jsx";

const SUGGESTIONS = [
  "Say hello in five words.",
  "Explain RAG in two sentences.",
  "Write a haiku about job hunting.",
];

/**
 * Model chat screen: a plain conversation with the local Ollama model, for
 * trying it out and checking that it's running — separate from the
 * resume/JD analysis features. Calls `POST /chat` (see
 * `aptly.api.routes_chat.chat_with_model` on the backend), which is
 * stateless, so the full history is resent with every message.
 *
 * Enter sends, Shift+Enter inserts a newline. While a reply is generating,
 * the shared `Thinking` indicator (spark, shimmer, seconds counter) shows in
 * the conversation.
 */
export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  async function send(text) {
    const content = text.trim();
    if (!content || loading) return;

    const history = [...messages, { role: "user", content }];
    setMessages(history);
    setInput("");
    setLoading(true);

    try {
      const payload = history.filter((m) => !m.error).map(({ role, content: c }) => ({ role, content: c }));
      const response = await chat(payload);
      setModel(response.model);
      setMessages([...history, { role: "assistant", content: response.reply }]);
    } catch (err) {
      setMessages([...history, { role: "assistant", content: err.message, error: true }]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send(input);
    }
  }

  return (
    <section className="card card-pad chat-card">
      <div className="card-head">
        <div>
          <h2>Model chat</h2>
          <p className="meta">Talk to the local model directly. Not connected to your resume or notes.</p>
        </div>
        <div className="chat-head-side">
          {model && <span className="badge badge-accent">{model}</span>}
          {messages.length > 0 && (
            <button type="button" className="btn btn-quiet" onClick={() => setMessages([])} disabled={loading}>
              Clear
            </button>
          )}
        </div>
      </div>

      <div className="chat-log">
        {messages.length === 0 && !loading && (
          <div className="empty">
            <div className="ic">
              <Icon name="spark" size={24} />
            </div>
            <h3>Ask the model anything</h3>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} type="button" className="chip" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role === "user" ? "user" : "bot"}${m.error ? " error-msg" : ""}`}>
            {m.content}
          </div>
        ))}

        {loading && (
          <div className="msg bot">
            <Thinking />
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          send(input);
        }}
      >
        <label htmlFor="chat-input" className="sr">
          Message
        </label>
        <textarea
          id="chat-input"
          className="textarea"
          rows={1}
          placeholder="Message the model…"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button type="submit" className="btn btn-primary" disabled={!input.trim() || loading}>
          <Icon name="send" size={18} />
          Send
        </button>
      </form>
    </section>
  );
}
