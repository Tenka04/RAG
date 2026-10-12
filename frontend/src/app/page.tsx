"use client";

import React, { useState, useEffect, useRef } from "react";

interface SourceItem {
  content: string;
  metadata?: {
    source?: string;
    chunk_id?: number | string;
    [key: string]: any;
  };
  score?: number | string;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceItem[];
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
}

interface KBStats {
  total_chunks: number;
  unique_documents: string[];
  ollama_online: boolean;
  default_model?: string;
}

const MODELS = [
  { id: "llama3.2:3b", name: "ChatGPT 4o (llama3.2:3b)", desc: "Flagship model • Fits 4GB VRAM GPU" },
  { id: "llama3.2:1b", name: "ChatGPT 4o mini (llama3.2:1b)", desc: "Ultra-fast for lightweight queries" },
  { id: "direct_synthesis", name: "Direct Grounded Mode", desc: "No Ollama required • Instant local synthesis" },
  { id: "gemma4:cloud", name: "Cloud Engine (gemma4:cloud)", desc: "Ollama cloud datacenter tier" },
];

export default function Home() {
  // Chat Sessions (like ChatGPT history)
  const [sessions, setSessions] = useState<ChatSession[]>([
    { id: "default_chat", title: "New chat", messages: [], createdAt: Date.now() },
  ]);
  const [activeSessionId, setActiveSessionId] = useState<string>("default_chat");

  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeModel, setActiveModel] = useState("llama3.2:3b");
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({});
  const [copiedMsgId, setCopiedMsgId] = useState<string | null>(null);

  // Knowledge Base Telemetry
  const [stats, setStats] = useState<KBStats>({
    total_chunks: 0,
    unique_documents: [],
    ollama_online: false,
  });
  const [isIngestingSamples, setIsIngestingSamples] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const currentSession =
    sessions.find((s) => s.id === activeSessionId) || sessions[0];
  const messages = currentSession?.messages || [];

  // Scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isStreaming]);

  // Fetch KB stats
  const fetchStats = async () => {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch {
      // Backend starting up
    }
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 6000);
    return () => clearInterval(interval);
  }, []);

  // Textarea auto-resize
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [input]);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3500);
  };

  // Start a new chat
  const handleNewChat = () => {
    const newId = `chat_${Date.now()}`;
    const newSession: ChatSession = {
      id: newId,
      title: "New chat",
      messages: [],
      createdAt: Date.now(),
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newId);
    setInput("");
    if (textareaRef.current) textareaRef.current.focus();
  };

  const handleDeleteSession = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (sessions.length === 1) {
      setSessions([{ id: `chat_${Date.now()}`, title: "New chat", messages: [], createdAt: Date.now() }]);
      return;
    }
    const updated = sessions.filter((s) => s.id !== id);
    setSessions(updated);
    if (activeSessionId === id) {
      setActiveSessionId(updated[0].id);
    }
  };

  // Sample Documents Ingestion
  const handleIngestSamples = async () => {
    setIsIngestingSamples(true);
    try {
      const res = await fetch("/api/ingest-sample", { method: "POST" });
      const data = await res.json();
      if (res.ok) {
        showToast(`Indexed ${data.total_chunks_added} chunks from sample documents!`);
        await fetchStats();
      } else {
        showToast(data.detail || "Error ingesting samples");
      }
    } catch {
      showToast("Unable to reach backend server.");
    } finally {
      setIsIngestingSamples(false);
    }
  };

  // Clear Database
  const handleClearKB = async () => {
    if (!confirm("Clear all indexed documents from the ChromaDB vector database?")) return;
    try {
      const res = await fetch("/api/clear", { method: "POST" });
      if (res.ok) {
        showToast("Knowledge base cleared!");
        await fetchStats();
      }
    } catch {
      showToast("Error clearing index.");
    }
  };

  // Upload Documents
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        showToast(`Indexed ${data.total_chunks_added} chunks from ${files.length} file(s)!`);
        await fetchStats();
        setUploadModalOpen(false);
      } else {
        showToast(data.detail || "Upload error");
      }
    } catch {
      showToast("Error uploading documents to server.");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  // Send message and stream response
  const handleSendMessage = async (textToSend?: string) => {
    const query = (textToSend || input).trim();
    if (!query || isStreaming) return;

    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    const userMessage: Message = {
      id: `usr_${Date.now()}`,
      role: "user",
      content: query,
    };

    const assistantId = `ast_${Date.now()}`;
    const assistantMessage: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      sources: [],
    };

    // Update session title on first message
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id === activeSessionId) {
          const isFirst = s.messages.length === 0;
          return {
            ...s,
            title: isFirst ? query.slice(0, 32) + (query.length > 32 ? "..." : "") : s.title,
            messages: [...s.messages, userMessage, assistantMessage],
          };
        }
        return s;
      })
    );

    setIsStreaming(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          model: activeModel,
          top_k: 3,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data: ")) {
            const rawJson = trimmed.slice(6);
            try {
              const chunk = JSON.parse(rawJson);
              if (chunk.type === "sources") {
                setSessions((prev) =>
                  prev.map((s) =>
                    s.id === activeSessionId
                      ? {
                          ...s,
                          messages: s.messages.map((m) =>
                            m.id === assistantId ? { ...m, sources: chunk.sources } : m
                          ),
                        }
                      : s
                  )
                );
              } else if (chunk.type === "token") {
                setSessions((prev) =>
                  prev.map((s) =>
                    s.id === activeSessionId
                      ? {
                          ...s,
                          messages: s.messages.map((m) =>
                            m.id === assistantId
                              ? { ...m, content: m.content + chunk.token }
                              : m
                          ),
                        }
                      : s
                  )
                );
              } else if (chunk.type === "done") {
                setIsStreaming(false);
              }
            } catch {
              // Parse error
            }
          }
        }
      }
    } catch {
      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeSessionId
            ? {
                ...s,
                messages: s.messages.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        content:
                          m.content ||
                          "⚠️ Could not reach the backend. Please start the server using `python server.py`.",
                      }
                    : m
                ),
              }
            : s
        )
      );
    } finally {
      setIsStreaming(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const copyToClipboard = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedMsgId(id);
    setTimeout(() => setCopiedMsgId(null), 2000);
  };

  const toggleSourceView = (id: string) => {
    setExpandedSources((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const selectedModelObj =
    MODELS.find((m) => m.id === activeModel) || MODELS[0];

  const quickPills = [
    { label: "📄 Summarize documents", prompt: "Summarize the key takeaways and topics across all uploaded documents." },
    { label: "🧠 Explain RAG pipeline", prompt: "Explain the 5 core stages of Retrieval-Augmented Generation described in the files." },
    { label: "🔍 Check indexed files", prompt: "List the documents currently indexed in ChromaDB and their main topics." },
    { label: "⚡ Recommended GPU models", prompt: "What Ollama models are recommended for local PCs with 4GB VRAM?" },
  ];

  return (
    <div className="gpt-app">
      {/* =====================================================================
          CHATGPT SIDEBAR (#171717)
          ===================================================================== */}
      <aside className={`gpt-sidebar ${sidebarOpen ? "" : "closed"}`}>
        <div className="sidebar-top-bar">
          <button
            className="sidebar-icon-btn"
            onClick={() => setSidebarOpen(false)}
            title="Close sidebar"
            aria-label="Close sidebar"
          >
            {/* Sidebar close icon */}
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <line x1="9" y1="3" x2="9" y2="21" />
            </svg>
          </button>

          <button
            className="sidebar-icon-btn"
            onClick={handleNewChat}
            title="New chat"
            aria-label="New chat"
          >
            {/* Edit / New chat icon */}
            <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 20h9" />
              <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
            </svg>
          </button>
        </div>

        {/* New Chat Row */}
        <div className="new-chat-row">
          <button className="new-chat-button" onClick={handleNewChat}>
            <div className="new-chat-left">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 8v8" />
                <path d="M8 12h8" />
              </svg>
              <span>New chat</span>
            </div>
            <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>Ctrl+K</span>
          </button>
        </div>

        {/* Scrollable Chat History & Knowledge Base */}
        <div className="sidebar-scroll-area">
          {/* History */}
          <div>
            <div className="sidebar-group-title">
              <span>Today</span>
            </div>
            {sessions.map((sess) => (
              <div
                key={sess.id}
                className={`chat-history-item ${sess.id === activeSessionId ? "active" : ""}`}
                onClick={() => setActiveSessionId(sess.id)}
              >
                <span className="chat-history-title">{sess.title}</span>
                <button
                  className="chat-delete-btn"
                  onClick={(e) => handleDeleteSession(e, sess.id)}
                  title="Delete chat"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>

          {/* Knowledge Base Documents */}
          <div>
            <div className="sidebar-group-title">
              <span>Knowledge Base</span>
              <span className="kb-badge">{stats.total_chunks} chunks</span>
            </div>

            <div className="kb-card-sidebar">
              <button
                className="kb-action-btn"
                onClick={handleIngestSamples}
                disabled={isIngestingSamples}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                </svg>
                {isIngestingSamples ? "Loading..." : "Load Sample Docs"}
              </button>

              <button
                className="kb-action-btn"
                onClick={() => setUploadModalOpen(true)}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                Upload Files (.pdf, .md)
              </button>

              {stats.unique_documents && stats.unique_documents.length > 0 && (
                <div style={{ display: "flex", flexDirection: "column", gap: "2px", marginTop: "4px" }}>
                  {stats.unique_documents.map((doc, idx) => (
                    <div key={idx} className="kb-doc-pill" title={doc}>
                      📄 <span>{doc}</span>
                    </div>
                  ))}
                </div>
              )}

              {stats.total_chunks > 0 && (
                <button
                  className="kb-action-btn"
                  onClick={handleClearKB}
                  style={{ color: "#ef4444", marginTop: "4px" }}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="3 6 5 6 21 6" />
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
                  </svg>
                  Clear Vector DB
                </button>
              )}
            </div>
          </div>
        </div>

        {/* User Profile at bottom (ChatGPT style) */}
        <div className="sidebar-footer-profile">
          <div className="user-profile-btn">
            <div className="user-avatar-circle">V</div>
            <div className="user-profile-info">
              <div className="user-profile-name">Vansh</div>
              <div className="user-profile-sub">
                {stats.ollama_online ? "🟢 Ollama Online" : "🟡 Grounded Mode"}
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* =====================================================================
          MAIN CHAT VIEWPORT (#212121)
          ===================================================================== */}
      <main className="gpt-main">
        {/* Top Header */}
        <header className="gpt-header">
          <div className="header-left">
            {!sidebarOpen && (
              <button
                className="sidebar-icon-btn"
                onClick={() => setSidebarOpen(true)}
                title="Open sidebar"
                aria-label="Open sidebar"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                  <line x1="9" y1="3" x2="9" y2="21" />
                </svg>
              </button>
            )}

            {/* ChatGPT Model Picker */}
            <div style={{ position: "relative" }}>
              <div
                className="model-picker-btn"
                onClick={() => setModelDropdownOpen(!modelDropdownOpen)}
              >
                <span>{selectedModelObj.name.split(" ")[0]}</span>
                <span className="model-picker-tag">{selectedModelObj.id}</span>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <polyline points="6 9 12 15 18 9" />
                </svg>
              </div>

              {modelDropdownOpen && (
                <div className="model-dropdown-menu">
                  {MODELS.map((m) => (
                    <div
                      key={m.id}
                      className={`model-menu-item ${m.id === activeModel ? "selected" : ""}`}
                      onClick={() => {
                        setActiveModel(m.id);
                        setModelDropdownOpen(false);
                      }}
                    >
                      <div>
                        <div className="model-item-title">{m.name}</div>
                        <div className="model-item-desc">{m.desc}</div>
                      </div>
                      {m.id === activeModel && (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                          <polyline points="20 6 9 17 4 12" />
                        </svg>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="header-right">
            <button
              className="sidebar-icon-btn"
              onClick={() => setUploadModalOpen(true)}
              title="Add Documents"
            >
              <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </button>
          </div>
        </header>

        {/* Toast Alert */}
        {toastMessage && (
          <div
            style={{
              position: "absolute",
              top: "60px",
              left: "50%",
              transform: "translateX(-50%)",
              background: "#333333",
              border: "1px solid rgba(255,255,255,0.15)",
              color: "#fff",
              padding: "7px 16px",
              borderRadius: "9999px",
              fontSize: "13px",
              zIndex: 60,
              boxShadow: "0 6px 20px rgba(0,0,0,0.4)",
            }}
          >
            {toastMessage}
          </div>
        )}

        {/* Chat Scroll Viewport */}
        <div className="chat-viewport">
          {messages.length === 0 ? (
            /* ===============================================================
               CHATGPT EMPTY STATE ("What can I help with?")
               =============================================================== */
            <div className="empty-hero-container">
              <h1 className="empty-hero-title">What can I help with?</h1>

              {/* Centered Input Capsule in Empty State */}
              <div className="empty-input-capsule-wrapper">
                <div className="gpt-input-capsule">
                  <button
                    className="capsule-plus-btn"
                    onClick={() => setUploadModalOpen(true)}
                    title="Attach files"
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <line x1="12" y1="5" x2="12" y2="19" />
                      <line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                  </button>

                  <textarea
                    ref={textareaRef}
                    rows={1}
                    className="gpt-textarea"
                    placeholder="Ask anything about your documents..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                  />

                  <button
                    className="gpt-send-btn"
                    onClick={() => handleSendMessage()}
                    disabled={!input.trim() || isStreaming}
                    title="Send prompt"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <line x1="12" y1="19" x2="12" y2="5" />
                      <polyline points="5 12 12 5 19 12" />
                    </svg>
                  </button>
                </div>
              </div>

              {/* Quick Suggestion Pills */}
              <div className="hero-pills-row">
                {quickPills.map((pill, idx) => (
                  <button
                    key={idx}
                    className="hero-pill-btn"
                    onClick={() => handleSendMessage(pill.prompt)}
                  >
                    {pill.label}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* ===============================================================
               MESSAGE STREAM (ChatGPT Exact Style)
               =============================================================== */
            <div className="chat-scroll-content">
              {messages.map((msg) => (
                <div key={msg.id} className={`msg-row ${msg.role}`}>
                  {msg.role === "assistant" && (
                    <div className="assistant-spark-avatar">
                      {/* Authentic OpenAI flower emblem SVG */}
                      <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M22.28 9.82a6 6 0 0 0-.52-4.91 6.05 6.05 0 0 0-6.51-2.9A6.06 6.06 0 0 0 4.98 4.18a6 6 0 0 0-4 2.9 6.05 6.05 0 0 0 .74 7.1 6 6 0 0 0 .51 4.91 6.05 6.05 0 0 0 6.52 2.9A6 6 0 0 0 13.26 24a6.05 6.05 0 0 0 5.77-4.2 6 6 0 0 0 4-2.9 6.05 6.05 0 0 0-.75-7.08zM13.26 22.43a4.48 4.48 0 0 1-2.88-1.04l.14-.08 4.78-2.76a.8.8 0 0 0 .4-.68v-6.74l2.02 1.17a.07.07 0 0 1 .04.05v5.58a4.5 4.5 0 0 1-4.5 4.5zm-9.66-4.13a4.47 4.47 0 0 1-.53-3.01l.14.08 4.78 2.76a.77.77 0 0 0 .78 0l5.85-3.37v2.33a.08.08 0 0 1-.03.06L9.74 19.95a4.5 4.5 0 0 1-6.14-1.65zM2.34 7.9a4.48 4.48 0 0 1 2.37-1.97v5.67a.77.77 0 0 0 .39.68l5.81 3.35-2.02 1.17a.08.08 0 0 1-.07 0l-4.83-2.79A4.5 4.5 0 0 1 2.34 7.9zm16.6 3.85L13.1 8.38l2.02-1.16a.08.08 0 0 1 .07 0l4.83 2.79a4.49 4.49 0 0 1-.68 8.1v-5.67a.79.79 0 0 0-.4-.59zm2.01-3.02l-.14-.08-4.78-2.79a.78.78 0 0 0-.78 0L9.41 9.23V6.9a.07.07 0 0 1 .03-.06l4.83-2.79a4.5 4.5 0 0 1 6.68 4.67zm-12.64 4.14l-2.02-1.17a.08.08 0 0 1-.04-.05V6.08a4.5 4.5 0 0 1 7.37-3.46l-.14.08-4.78 2.76a.8.8 0 0 0-.4.68zm1.1-2.37l2.6-1.5 2.61 1.5v3l-2.61 1.5-2.6-1.5z" />
                      </svg>
                    </div>
                  )}

                  {msg.role === "user" ? (
                    <div className="user-bubble">{msg.content}</div>
                  ) : (
                    <div className="assistant-content-wrapper">
                      <div className="assistant-text-content">
                        {msg.content}
                        {isStreaming && !msg.content && (
                          <span style={{ color: "var(--text-muted)" }}>Thinking...</span>
                        )}
                        {isStreaming && <span className="gpt-typing-cursor" />}
                      </div>

                      {/* Sources Cards (ChatGPT Search Style) */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="sources-carousel">
                          <div
                            className="sources-header-pill"
                            onClick={() => toggleSourceView(msg.id)}
                          >
                            <span>📚</span>
                            <span>{msg.sources.length} sources referenced</span>
                            <span>{expandedSources[msg.id] ? "▲" : "▼"}</span>
                          </div>

                          {expandedSources[msg.id] && (
                            <div className="sources-cards-row">
                              {msg.sources.map((src, i) => (
                                <div key={i} className="source-citation-card">
                                  <div className="source-card-top">
                                    <span className="source-doc-name" title={src.metadata?.source}>
                                      📄 {src.metadata?.source || `Source ${i + 1}`}
                                    </span>
                                    <span className="source-score">
                                      {src.score ? `${Math.round(Number(src.score) * 100)}%` : ""}
                                    </span>
                                  </div>
                                  <div className="source-card-preview">{src.content}</div>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Action Bar (Copy, Like, Dislike) */}
                      {msg.content && (
                        <div className="msg-action-bar">
                          <button
                            className="action-bar-btn"
                            onClick={() => copyToClipboard(msg.id, msg.content)}
                            title="Copy response"
                          >
                            {copiedMsgId === msg.id ? (
                              <span style={{ fontSize: "11px", color: "var(--status-green)" }}>✓</span>
                            ) : (
                              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                              </svg>
                            )}
                          </button>

                          <button className="action-bar-btn" title="Good response">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
                            </svg>
                          </button>

                          <button className="action-bar-btn" title="Bad response">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3" />
                            </svg>
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Floating Input Capsule at Bottom (when chat is active) */}
        {messages.length > 0 && (
          <div className="gpt-input-fixed-bar">
            <div className="gpt-input-capsule">
              <button
                className="capsule-plus-btn"
                onClick={() => setUploadModalOpen(true)}
                title="Attach files"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <line x1="12" y1="5" x2="12" y2="19" />
                  <line x1="5" y1="12" x2="19" y2="12" />
                </svg>
              </button>

              <textarea
                ref={textareaRef}
                rows={1}
                className="gpt-textarea"
                placeholder="Message ChatGPT..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
              />

              <button
                className="gpt-send-btn"
                onClick={() => handleSendMessage()}
                disabled={!input.trim() || isStreaming}
                title="Send message"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <line x1="12" y1="19" x2="12" y2="5" />
                  <polyline points="5 12 12 5 19 12" />
                </svg>
              </button>
            </div>

            <div className="gpt-disclaimer-text">
              ChatGPT can make mistakes. Verify important facts from source documents.
            </div>
          </div>
        )}
      </main>

      {/* =====================================================================
          DOCUMENT UPLOAD MODAL
          ===================================================================== */}
      {uploadModalOpen && (
        <div className="gpt-modal-overlay" onClick={() => setUploadModalOpen(false)}>
          <div className="gpt-modal-box" onClick={(e) => e.stopPropagation()}>
            <div className="modal-title-row">
              <h2 className="modal-main-title">Add to Knowledge Base</h2>
              <button className="sidebar-icon-btn" onClick={() => setUploadModalOpen(false)}>
                ✕
              </button>
            </div>

            <p style={{ fontSize: "13.5px", color: "var(--text-secondary)", lineHeight: 1.5 }}>
              Upload files to index them directly into ChromaDB. Supported formats: PDF (.pdf), Word (.docx), Markdown (.md), plain text (.txt), CSV, JSON, and code.
            </p>

            <label className="gpt-upload-zone">
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.docx,.txt,.md,.csv,.json,.py,.js,.html"
                onChange={handleFileUpload}
                style={{ display: "none" }}
              />
              <div style={{ fontSize: "32px" }}>📁</div>
              <div style={{ fontSize: "14px", fontWeight: 500, color: "var(--text-primary)" }}>
                {isUploading ? "Chunking & Vector Indexing..." : "Click to select or drop files here"}
              </div>
              <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                Files will be indexed with ChromaDB vector embeddings
              </div>
            </label>

            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              <button
                className="kb-action-btn"
                style={{ width: "auto", padding: "8px 16px" }}
                onClick={() => setUploadModalOpen(false)}
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
