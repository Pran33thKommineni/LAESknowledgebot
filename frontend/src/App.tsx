import { FormEvent, useState } from "react";
import "./App.css";

interface AnswerSource {
  documentId: string;
  documentName: string;
  webUrl: string;
  snippet: string;
}

interface Answer {
  text: string;
  sources: AnswerSource[];
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  answer?: Answer;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:4000";

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [syncStatus, setSyncStatus] = useState<string | null>(null);
  const [syncLoading, setSyncLoading] = useState(false);

  const submitQuestion = async (e: FormEvent) => {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) return;

    setError(null);
    setLoading(true);

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
    };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");

    try {
      const res = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: trimmed }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || "Request failed");
      }

      const answer = (await res.json()) as Answer;

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: answer.text,
        answer,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      setError(err.message ?? "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const triggerSync = async () => {
    setSyncLoading(true);
    setSyncStatus(null);
    try {
      const res = await fetch(`${API_BASE}/admin/sync`, {
        method: "POST",
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(body.error || "Sync failed");
      }
      setSyncStatus(`Synced ${body.documentsSynced ?? 0} documents.`);
    } catch (err: any) {
      setSyncStatus(`Sync error: ${err.message ?? "Unknown error"}`);
    } finally {
      setSyncLoading(false);
    }
  };

  return (
    <div className="app-root">
      <header className="app-header">
        <div>
          <h1>LAES Knowledge Bot</h1>
          <p className="subtitle">
            Ask questions about your LAES enhanced technical studio materials without digging through OneDrive.
          </p>
        </div>
        <div className="admin-controls">
          <button type="button" onClick={triggerSync} disabled={syncLoading}>
            {syncLoading ? "Syncing…" : "Sync OneDrive"}
          </button>
          {syncStatus && <span className="sync-status">{syncStatus}</span>}
        </div>
      </header>

      <main className="chat-layout">
        <section className="chat-window">
          {messages.length === 0 && (
            <div className="empty-state">
              <h2>Try asking:</h2>
              <ul>
                <li>“What are the key requirements for the current LAES technical studio project?”</li>
                <li>“Summarize the latest assignment instructions.”</li>
                <li>“What readings are suggested for week 2?”</li>
              </ul>
            </div>
          )}
          {messages.map((m) => (
            <div key={m.id} className={`message ${m.role}`}>
              <div className="message-role">{m.role === "user" ? "You" : "Bot"}</div>
              <div className="message-content">{m.content}</div>
              {m.role === "assistant" && m.answer && m.answer.sources.length > 0 && (
                <div className="sources">
                  <div className="sources-title">Sources</div>
                  <ul>
                    {m.answer.sources.map((s, idx) => (
                      <li key={`${m.id}-${idx}`}>
                        <a href={s.webUrl} target="_blank" rel="noreferrer">
                          {s.documentName}
                        </a>
                        <div className="source-snippet">{s.snippet.slice(0, 220)}{s.snippet.length > 220 ? "…" : ""}</div>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
        </section>

        <section className="input-panel">
          <form onSubmit={submitQuestion} className="question-form">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about your LAES technical studio materials…"
              rows={3}
            />
            <div className="form-footer">
              {error && <span className="error-text">{error}</span>}
              <button type="submit" disabled={loading}>
                {loading ? "Thinking…" : "Ask"}
              </button>
            </div>
          </form>
        </section>
      </main>
    </div>
  );
}

export default App;
