import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  getChats,
  createChat,
  getFiles,
  getMessages,
  saveMessage,
  uploadFiles,
  queryChat,
  downloadSummaryReport,
} from "./api";
import "./App.css";

function App() {
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [files, setFiles] = useState([]);
  const [newChatName, setNewChatName] = useState("");
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [intelligence, setIntelligence] = useState("auto");
  const [selectedMode, setSelectedMode] = useState("generic"); // "generic" | "document"
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [summaryLoading, setSummaryLoading] = useState(false);

  const activeChat = chats.find((c) => c.chat_id === activeChatId);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    loadChats();
  }, []);

  useEffect(() => {
    if (activeChatId) {
      loadChatData(activeChatId);
    }
  }, [activeChatId]);

  async function loadChats() {
    const res = await getChats();
    setChats(res.data);

    if (res.data.length > 0 && !activeChatId) {
      setActiveChatId(res.data[0].chat_id);
    }
  }

  async function loadChatData(chatId) {
    const [messagesRes, filesRes] = await Promise.all([
      getMessages(chatId),
      getFiles(chatId),
    ]);

    setMessages(messagesRes.data || []);
    setFiles(filesRes.data || []);
  }

  async function handleCreateChat() {
    if (!newChatName.trim()) return;

    try {
      const res = await createChat(newChatName.trim());
      setNewChatName("");
      await loadChats();
      setActiveChatId(res.data.chat_id);
    } catch (err) {
      alert(err.response?.data?.detail || "Could not create chat");
    }
  }

  async function handleUpload() {
    if (!activeChatId || selectedFiles.length === 0) return;

    setUploading(true);

    try {
      await uploadFiles(activeChatId, selectedFiles);
      setSelectedFiles([]);
      await loadChatData(activeChatId);
      alert("Indexing completed");
    } catch (err) {
      alert(err.response?.data?.detail || "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function handleSend() {
    if (!prompt.trim() || !activeChatId) return;

    const userMessage = {
      role: "user",
      content: prompt,
      citations: [],
    };

    setMessages((prev) => [...prev, userMessage]);
    setPrompt("");
    setLoading(true);

    try {
      await saveMessage(activeChatId, userMessage);

      const res = await queryChat(activeChatId, prompt, intelligence, selectedMode);

      const assistantMessage = {
        role: "assistant",
        content: res.data.answer,
        citations: res.data.citations || [],
        intelligence: res.data.intelligence,
        model_used: res.data.model_used,
        tokens_used: res.data.tokens_used,
        mode: selectedMode,
        cache_metadata: res.data.cache_metadata,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      await loadChats();
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            err.response?.data?.detail ||
            "Could not connect to FastAPI backend.",
          citations: [],
        },
      ]);
    } finally {
      setLoading(false);
    }
  }
  async function handleChatSummary() {
    if (!activeChatId) return;
    setSummaryLoading(true);
    try {
      await downloadSummaryReport(activeChatId);
    } catch (err) {
      let detail = "Could not generate summary report. Please try again.";
      if (err.response?.data instanceof Blob) {
        try {
          const text = await err.response.data.text();
          const parsed = JSON.parse(text);
          detail = parsed.detail || detail;
        } catch (_) {
          // ignore parse errors — use the default message
        }
      } else {
        detail = err.response?.data?.detail || err.message || detail;
      }
      alert(detail);
    } finally {
      setSummaryLoading(false);
    }
  }

  return (
    <div className="app">
      {/* ── Sidebar ─────────────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <span className="logo-icon">⬡</span>
          <span className="logo-text">RAG Portal</span>
        </div>

        <div className="new-chat-row">
          <input
            className="chat-name-input"
            placeholder="New chat name..."
            value={newChatName}
            onChange={(e) => setNewChatName(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") handleCreateChat(); }}
          />
          <button className="new-chat-btn" onClick={handleCreateChat} title="Create chat">
            +
          </button>
        </div>

        <div className="section-label">Chats</div>

        <div className="chat-list">
          {chats.length === 0 && <p className="empty-hint">No chats yet.</p>}

          {chats.map((chat) => (
            <button
              key={chat.chat_id}
              className={chat.chat_id === activeChatId ? "chat-btn active" : "chat-btn"}
              onClick={() => setActiveChatId(chat.chat_id)}
            >
              <span className="chat-btn-icon"></span>
              <span className="chat-btn-name">{chat.chat_name}</span>
            </button>
          ))}
        </div>

        {activeChat && (
          <div className="token-pill">
            <span className="token-label">Tokens</span>
            <span className="token-value">{(activeChat.total_tokens_used || 0).toLocaleString()}</span>
          </div>
        )}

        <div className="sidebar-divider" />

        <div className="section-label">Document Upload</div>

        <label className="file-drop-zone">
          <input
            type="file"
            multiple
            accept=".pdf,.docx,.txt,.png,.jpg,.jpeg,.webp"
            onChange={(e) => setSelectedFiles([...e.target.files])}
            style={{ display: "none" }}
          />
          <span className="file-drop-icon">📂</span>
          <span className="file-drop-text">
            {selectedFiles.length > 0
              ? `${selectedFiles.length} file(s) selected`
              : "Click to select files"}
          </span>
        </label>

        <button
          className="index-btn"
          onClick={handleUpload}
          disabled={uploading || !activeChatId || selectedFiles.length === 0}
        >
          {uploading ? (
            <><span className="spinner" />Indexing...</>
          ) : (
            " Index Documents"
          )}
        </button>

        <div className="sidebar-divider" />

        <div className="section-label">Intelligence</div>
        <select
          className="intelligence-select"
          value={intelligence}
          onChange={(e) => setIntelligence(e.target.value)}
        >
          <option value="auto">Auto</option>
          <option value="instant">Instant</option>
          <option value="medium">Medium</option>
          <option value="high">High</option>
        </select>

        <div className="sidebar-divider" />

        <div className="section-label">Uploaded Files</div>

        <div className="files">
          {files.length === 0 && <p className="empty-hint">No files uploaded.</p>}
          {files.map((file, index) => (
            <span className="file-badge" key={index}>
               {file.file_name}
            </span>
          ))}
        </div>
      </aside>

      {/* ── Main Area ────────────────────────────────────────────────────── */}
      <main className="main">
        {/* Header */}
        <div className="main-header">
          <div className="main-title-block">
            <h1 className="main-title">
              {selectedMode === "generic" ? "Generic" : "Document"} RAG Portal
            </h1>
            <p className="subtitle">
              {activeChatId
                ? selectedMode === "generic"
                  ? "Hi Sanjana! Ask anything — I'll search your documents or use general knowledge."
                  : "Hi Sanjana! Ask a question — I'll answer strictly from your uploaded documents."
                : "Create or select a chat session from the sidebar to begin."}
            </p>
          </div>

          {/* ── Mode Toggle ─────────────────────────────────────────────── */}
          <div className="mode-toggle" role="group" aria-label="Query mode">
            <button
              id="mode-generic"
              className={`mode-btn ${selectedMode === "generic" ? "active" : ""}`}
              onClick={() => setSelectedMode("generic")}
            >
              <span className="mode-icon"></span> Generic
            </button>
            <button
              id="mode-document"
              className={`mode-btn ${selectedMode === "document" ? "active" : ""}`}
              onClick={() => setSelectedMode("document")}
            >
              <span className="mode-icon"></span> Document
            </button>
          </div>
        </div>

        {/* Mode hint banner */}
        <div className={`mode-banner ${selectedMode}`}>
          {selectedMode === "generic" ? (
            <>
              <strong>Generic mode</strong> — Answers from documents when available, otherwise uses general knowledge.
            </>
          ) : (
            <>
              <strong>Document mode</strong> — Answers strictly from your uploaded documents only.
            </>
          )}
        </div>

        {/* Messages */}
        <div className="messages">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={msg.role === "user" ? "message user" : "message assistant"}
            >
              <div className="message-avatar">
                {msg.role === "user" ? "S" : "A"}
              </div>

              <div className="message-body">
                <ReactMarkdown>{msg.content}</ReactMarkdown>

                {msg.citations?.length > 0 && (
                  <details className="citations-details">
                    <summary>📎 {msg.citations.length} source chunk{msg.citations.length > 1 ? "s" : ""}</summary>

                    {msg.citations.map((cite, i) => (
                      <div className="citation-card" key={i}>
                        <div className="citation-header">
                          <b>Source #{i + 1}: {cite.file_name} · Page {cite.page_number}</b>
                          <span className={`elem-badge ${cite.element_type?.includes("table") ? "table" : "text"}`}>
                            {cite.element_type?.toUpperCase()}
                          </span>
                        </div>
                        <pre>{cite.content}</pre>
                      </div>
                    ))}
                  </details>
                )}

                {msg.role === "assistant" && (
                  <>
                    <div className="cache-source-note">
                      <div className="divider-line">────────────────────────────</div>
                      {msg.cache_metadata && msg.cache_metadata.cache_hit ? (
                        <div className="source-label hit">
                          Source: Semantic Cache
                          {msg.cache_metadata.similarity_score !== undefined && msg.cache_metadata.similarity_score !== null && (
                            <span> | Similarity: {(msg.cache_metadata.similarity_score * 100).toFixed(1)}%</span>
                          )}
                        </div>
                      ) : (
                        <div className="source-label miss">
                          Source: LLM (Fresh Response)
                        </div>
                      )}
                    </div>
                    <div className="message-meta">
                      {msg.mode && (
                        <span className={`mode-pill ${msg.mode}`}>
                          {msg.mode === "generic" ? "✦ Generic" : " Document"}
                        </span>
                      )}
                      {msg.model_used && (
                        <span className="meta-item">
                          {msg.intelligence && <>{msg.intelligence} · </>}
                          {msg.model_used}
                          {msg.tokens_used ? ` · ${(msg.tokens_used).toLocaleString()} tokens` : ""}
                        </span>
                      )}
                    </div>
                  </>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message assistant">
              <div className="message-avatar">A</div>
              <div className="message-body loading-dots">
                <span /><span /><span />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input bar */}
        {activeChatId && (
          <div className="chat-box">
            <textarea
              className="chat-textarea"
              value={prompt}
              placeholder={
                selectedMode === "document"
                  ? "Ask a question about your uploaded documents..."
                  : "Ask anything — documents or general knowledge..."
              }
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              rows={1}
            />

            <button
              className="summary-btn"
              onClick={handleChatSummary}
              title="Chat summary report"
              disabled={summaryLoading}
            >
              {summaryLoading ? "…" : "📄"}
            </button>

            <button
              className="send-btn"
              onClick={handleSend}
              disabled={loading || !prompt.trim()}
            >
              {loading ? <span className="spinner white" /> : "↑"}
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;