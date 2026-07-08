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

      const res = await queryChat(activeChatId, prompt, intelligence);

      const assistantMessage = {
        role: "assistant",
        content: res.data.answer,
        citations: res.data.citations || [],
        intelligence: res.data.intelligence,
        model_used: res.data.model_used,
        tokens_used: res.data.tokens_used,
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
      // downloadSummaryReport triggers a browser file-save dialog directly.
      // No return value — the PDF is streamed to disk by the browser.
      await downloadSummaryReport(activeChatId);
    } catch (err) {
      // When the response is a blob, axios wraps the error differently.
      // Try to parse the blob as text to surface the backend error message.
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
      <aside className="sidebar">
        <button className="new-chat-btn" onClick={handleCreateChat}>
          + New Chat
        </button>

        <input
          className="chat-input"
          placeholder="Chat name*"
          value={newChatName}
          onChange={(e) => setNewChatName(e.target.value)}
        />

        <hr />

        <h3>Chats</h3>

        <div className="chat-list">
          {chats.length === 0 && <p>No chats yet.</p>}

          {chats.map((chat) => (
            <button
              key={chat.chat_id}
              className={
                chat.chat_id === activeChatId
                  ? "chat-btn active"
                  : "chat-btn"
              }
              onClick={() => setActiveChatId(chat.chat_id)}
            >
              {chat.chat_name}
            </button>
          ))}
        </div>

        {activeChat && (
          <p className="token-count">
            Total tokens used:{" "}
            <b>{activeChat.total_tokens_used || 0}</b>
          </p>
        )}

        <hr />

        <h3>Document Upload</h3>

        <input
          type="file"
          multiple
          accept=".pdf,.docx,.txt,.png,.jpg,.jpeg,.webp"
          onChange={(e) => setSelectedFiles([...e.target.files])}
        />

        <button
          className="index-btn"
          onClick={handleUpload}
          disabled={uploading || !activeChatId}
        >
          {uploading ? "Indexing..." : "Index Documents"}
        </button>

        <hr />

        <label>Intelligence</label>
        <select
          value={intelligence}
          onChange={(e) => setIntelligence(e.target.value)}
        >
          <option value="auto">Auto</option>
          <option value="instant">Instant</option>
          <option value="medium">Medium</option>
          <option value="high">High</option>
        </select>

        <hr />

        <h3>Uploaded Files</h3>

        <div className="files">
          {files.length === 0 && <p>No files uploaded.</p>}

          {files.map((file, index) => (
            <span className="file-badge" key={index}>
              {file.file_name}
            </span>
          ))}
        </div>
      </aside>

      <main className="main">
        <h1>RAG Portal</h1>

        <p className="subtitle">
          {activeChatId
            ? "Hi Sanjana! Welcome to the RAG Portal."
            : "Create or select a chat session to begin."}
        </p>

        <div className="messages">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={
                msg.role === "user"
                  ? "message user"
                  : "message assistant"
              }
            >
              <ReactMarkdown>{msg.content}</ReactMarkdown>

              {msg.citations?.length > 0 && (
                <details>
                  <summary>Chunks</summary>

                  {msg.citations.map((cite, i) => (
                    <div className="citation-card" key={i}>
                      <div className="citation-header">
                        <b>
                          Source #{i + 1}: {cite.file_name}{" "}
                          Page {cite.page_number}
                        </b>
                        <span>{cite.element_type}</span>
                      </div>

                      <pre>{cite.content}</pre>
                    </div>
                  ))}
                </details>
              )}

              {msg.role === "assistant" && msg.model_used && (
                <small>
                  Routed to {msg.intelligence} · model: {msg.model_used} ·
                  tokens: {msg.tokens_used || 0}
                </small>
              )}
            </div>
          ))}

          {loading && <div className="message assistant">Retrieving answer...</div>}
          <div ref={messagesEndRef} />
        </div>

        {activeChatId && (
          <div className="chat-box">
            <input
                value={prompt}
                placeholder="Ask a question based on your uploaded documents..."
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                if (e.key === "Enter") handleSend();
                }}
            />

            <button
                className="summary-btn"
                onClick={handleChatSummary}
                title="Chat summary report"
                disabled={summaryLoading}
            >
                {summaryLoading ? "..." : "📄"}
            </button>

            <button onClick={handleSend} disabled={loading}>
                Send
            </button>
            </div>
        )}
      </main>
    </div>
  );
}

export default App;