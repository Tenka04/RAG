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
  timestamp: string;
}

interface KBStats {
  status?: string;
  total_chunks: number;
  unique_documents: string[];
  ollama_online: boolean;
  default_model?: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeModel, setActiveModel] = useState("llama3.2:3b");
  const [topK, setTopK] = useState(3);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({});
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // System & KB status
  const [stats, setStats] = useState<KBStats>({
    total_chunks: 0,
    unique_documents: [],
    ollama_online: false,
  });
  const [isIngestingSamples, setIsIngestingSamples] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll on new tokens
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
      // API might be spinning up
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
    setStatusMessage(msg);
    setTimeout(() => setStatusMessage(null), 4000);
  };

  // Ingest sample documents
  const handleIngestSamples = async () => {
    setIsIngestingSamples(true);
    try {
      const res = await fetch("/api/ingest-sample", { method: "POST" });
      const data = await res.json();
      if (res.ok) {
        showToast(`Indexed ${data.total_chunks_added} chunks from sample documents!`);
        await fetchStats();
      } else {
        showToast(data.detail || "Error loading samples");
      }
    } catch {
      showToast("Could not reach backend API. Check server status.");
    } finally {
      setIsIngestingSamples(false);
    }
  };

  // Clear knowledge base
  const handleClearKB = async () => {
    if (!confirm("Are you sure you want to clear all indexed documents from ChromaDB?")) return;
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

  // Handle file upload
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

  // Send query & stream SSE response
  const handleSendMessage = async (textToSend?: string) => {
    const query = (textToSend || input).trim();
    if (!query || isStreaming) return;

    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    const userMessage: Message = {
      id: `usr_${Date.now()}`,
      role: "user",
      content: query,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    const assistantId = `ast_${Date.now()}`;
    const assistantMessage: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      sources: [],
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsStreaming(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          model: activeModel,
          top_k: topK,
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
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantId ? { ...msg, sources: chunk.sources } : msg
                  )
                );
              } else if (chunk.type === "token") {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantId
                      ? { ...msg, content: msg.content + chunk.token }
                      : msg
                  )
                );
              } else if (chunk.type === "done") {
                setIsStreaming(false);
              }
            } catch {
              // Handle parse errors
            }
          }
        }
      }
    } catch {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? {
                ...msg,
                content:
                  msg.content ||
                  "⚠️ Unable to reach the backend API server. Make sure `python server.py` is running on port 8000.",
              }
            : msg
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
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const toggleSourceView = (id: string) => {
    setExpandedSources((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleNewChat = () => {
    setMessages([]);
    if (textareaRef.current) textareaRef.current.focus();
  };

  const promptSuggestions = [
    {
      title: "What is RAG?",
      desc: "Understand Retrieval-Augmented Generation architecture",
      prompt: "What is Retrieval-Augmented Generation (RAG) and why is it important?",
    },
    {
      title: "5 Stages of RAG",
      desc: "Detailed breakdown of the RAG pipeline phases",
      prompt: "Explain the five core stages of the RAG pipeline described in the documentation.",
    },
    {
      title: "Ollama & VRAM",
      desc: "Recommended models for 4GB GPUs like RTX 3050",
      prompt: "Which Ollama models fit within 4GB VRAM GPUs like an RTX 3050?",
    },
    {
      title: "Summarize Knowledge Base",
      desc: "Get an executive summary of all indexed topics",
      prompt: "Provide an executive summary of key facts across all currently indexed documents.",
    },
  ];

  return (
    <div className="app-container">
      {/* =====================================================================
          SIDEBAR
          ===================================================================== */}
      <aside className={`sidebar ${sidebarOpen ? "" : "collapsed"}`}>
        <div className="sidebar-header">
          <div className="brand-badge">
            <div className="brand-icon">🧠</div>
            <div>
              <div className="brand-title">Deep RAG</div>
            </div>
            <span className="brand-tag">v2.0</span>
          </div>
          <button
            className="icon-btn"
            onClick={() => setSidebarOpen(false)}
            title="Collapse Sidebar"
            aria-label="Collapse Sidebar"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        <button className="new-chat-btn" onClick={handleNewChat}>
          <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            New Chat
          </span>
          <span className="kbd-shortcut">Clear</span>
        </button>

        <div className="sidebar-content">
          <div>
            <div className="sidebar-section-title">
              <span>Knowledge Base</span>
              <span className="brand-tag">{stats.total_chunks} chunks</span>
            </div>

            <div className="action-row">
              <button
                className="secondary-action-btn"
                onClick={handleIngestSamples}
                disabled={isIngestingSamples}
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                </svg>
                {isIngestingSamples ? "Indexing Samples..." : "Ingest Sample Docs"}
              </button>

              <button
                className="secondary-action-btn"
                onClick={() => setUploadModalOpen(true)}
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                Upload Documents
              </button>
            </div>
          </div>

          <div>
            <div className="sidebar-section-title">
              <span>Indexed Files ({stats.unique_documents?.length || 0})</span>
            </div>

            <div className="doc-list">
              {stats.unique_documents && stats.unique_documents.length > 0 ? (
                stats.unique_documents.map((doc, idx) => (
                  <div key={idx} className="doc-item">
                    <span className="doc-name" title={doc}>
                      📄 {doc}
                    </span>
                    <span className="doc-badge">active</span>
                  </div>
                ))
              ) : (
                <div style={{ fontSize: "12px", color: "var(--text-muted)", padding: "6px 0" }}>
                  No documents indexed yet. Ingest sample docs or upload files above.
                </div>
              )}
            </div>
          </div>

          {stats.total_chunks > 0 && (
            <div style={{ marginTop: "auto" }}>
              <button
                className="secondary-action-btn"
                onClick={handleClearKB}
                style={{ color: "#ef4444", borderColor: "rgba(239, 68, 68, 0.2)" }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="3 6 5 6 21 6" />
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
                Clear Database
              </button>
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          <div className="telemetry-card">
            <div className="telemetry-row">
              <span style={{ color: "var(--text-muted)" }}>Ollama Engine:</span>
              <span className="status-indicator">
                <span className={stats.ollama_online ? "dot-green" : "dot-amber"} />
                {stats.ollama_online ? "Online" : "Grounded Mode"}
              </span>
            </div>
            <div className="telemetry-row">
              <span style={{ color: "var(--text-muted)" }}>Vector Store:</span>
              <span style={{ color: "var(--text-primary)", fontWeight: 500 }}>ChromaDB</span>
            </div>
          </div>
        </div>
      </aside>

      {/* =====================================================================
          MAIN VIEWPORT
          ===================================================================== */}
      <main className="main-viewport">
        {/* Top Navbar */}
        <header className="top-navbar">
          <div className="nav-left">
            {!sidebarOpen && (
              <button
                className="icon-btn"
                onClick={() => setSidebarOpen(true)}
                title="Open Sidebar"
                aria-label="Open Sidebar"
              >
                <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="3" y1="12" x2="21" y2="12" />
                  <line x1="3" y1="6" x2="21" y2="6" />
                  <line x1="3" y1="18" x2="21" y2="18" />
                </svg>
              </button>
            )}

            {/* Model Selector Dropdown */}
            <div style={{ position: "relative" }}>
              <select
                className="model-dropdown-btn"
                value={activeModel}
                onChange={(e) => setActiveModel(e.target.value)}
              >
                <option value="llama3.2:3b">🤖 llama3.2:3b (Recommended 4GB)</option>
                <option value="llama3.2:1b">⚡ llama3.2:1b (Ultra Fast)</option>
                <option value="direct_synthesis">📄 Grounded Synthesis (No Ollama)</option>
                <option value="gemma4:cloud">☁️ gemma4:cloud (Ollama Cloud)</option>
              </select>
            </div>
          </div>

          <div className="nav-right">
            <div className="top-k-pill">
              <span>Top-K:</span>
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                style={{ background: "transparent", color: "var(--text-primary)", fontWeight: 600 }}
              >
                <option value="1">1 Chunk</option>
                <option value="2">2 Chunks</option>
                <option value="3">3 Chunks</option>
                <option value="5">5 Chunks</option>
                <option value="8">8 Chunks</option>
              </select>
            </div>

            <button
              className="icon-btn"
              onClick={() => setUploadModalOpen(true)}
              title="Upload Documents"
            >
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </button>
          </div>
        </header>

        {/* Status Toast */}
        {statusMessage && (
          <div
            style={{
              position: "absolute",
              top: "68px",
              left: "50%",
              transform: "translateX(-50%)",
              background: "var(--bg-surface-elevated)",
              border: "1px solid var(--border-medium)",
              padding: "8px 18px",
              borderRadius: "9999px",
              fontSize: "13px",
              color: "var(--text-primary)",
              boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
              zIndex: 50,
            }}
          >
            {statusMessage}
          </div>
        )}

        {/* Message Thread */}
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-logo-glow">🧠</div>
              <h1 className="empty-title">Deep RAG Intelligence</h1>
              <p className="empty-subtitle">
                Ask questions grounded strictly in your private documents. Powered by ChromaDB vector
                similarity and high-precision neural retrieval.
              </p>

              <div className="prompt-suggestions-grid">
                {promptSuggestions.map((item, idx) => (
                  <button
                    key={idx}
                    className="suggestion-card"
                    onClick={() => handleSendMessage(item.prompt)}
                  >
                    <div className="suggestion-header">{item.title}</div>
                    <div className="suggestion-desc">{item.desc}</div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages-wrapper">
              {messages.map((msg) => (
                <div key={msg.id} className="message-row">
                  <div className={`message-avatar ${msg.role === "user" ? "user-avatar" : "assistant-avatar"}`}>
                    {msg.role === "user" ? (
                      "V"
                    ) : (
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2a1 1 0 0 1 1 1v2.07A8.002 8.002 0 0 1 20.93 11H23a1 1 0 1 1 0 2h-2.07A8.002 8.002 0 0 1 13 20.93V23a1 1 0 1 1-2 0v-2.07A8.002 8.002 0 0 1 3.07 13H1a1 1 0 1 1 0-2h2.07A8.002 8.002 0 0 1 11 5.07V3a1 1 0 0 1 1-1zm0 5a6 6 0 1 0 0 12 6 6 0 0 0 0-12z" />
                      </svg>
                    )}
                  </div>

                  <div className="message-body">
                    <div className="message-sender">
                      <span>{msg.role === "user" ? "You" : "RAG Assistant"}</span>
                      <span className="message-time">{msg.timestamp}</span>
                    </div>

                    <div className="message-content">
                      {msg.content}
                      {isStreaming && msg.role === "assistant" && !msg.content && (
                        <span style={{ color: "var(--text-muted)" }}>Retrieving sources & analyzing...</span>
                      )}
                      {isStreaming && msg.role === "assistant" && (
                        <span className="typing-cursor" />
                      )}
                    </div>

                    {/* Sources Section */}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="sources-card">
                        <button
                          className="sources-toggle"
                          onClick={() => toggleSourceView(msg.id)}
                        >
                          <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                            <span>📚</span>
                            <span>
                              {msg.sources.length} Context Sources Referenced
                            </span>
                          </span>
                          <span>{expandedSources[msg.id] ? "Hide ▲" : "Inspect ▼"}</span>
                        </button>

                        {expandedSources[msg.id] && (
                          <div className="sources-list">
                            {msg.sources.map((src, i) => (
                              <div key={i} className="source-item">
                                <div className="source-header">
                                  <span>
                                    📄 {src.metadata?.source || `Source ${i + 1}`}{" "}
                                    <span style={{ color: "var(--text-muted)", fontSize: "11px" }}>
                                      (Chunk {src.metadata?.chunk_id ?? i})
                                    </span>
                                  </span>
                                  <span className="source-score-pill">
                                    Score: {src.score}
                                  </span>
                                </div>
                                <div className="source-snippet">{src.content}</div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Message Actions */}
                    {msg.role === "assistant" && msg.content && (
                      <div className="message-actions">
                        <button
                          className="action-icon-btn"
                          onClick={() => copyToClipboard(msg.id, msg.content)}
                        >
                          {copiedId === msg.id ? "✓ Copied" : "Copy"}
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Bottom Floating Prompt Capsule */}
        <div className="input-fixed-container">
          <div className="input-capsule">
            <button
              className="capsule-btn attach-btn"
              onClick={() => setUploadModalOpen(true)}
              title="Attach Document"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
              </svg>
            </button>

            <textarea
              ref={textareaRef}
              rows={1}
              className="chat-textarea"
              placeholder="Ask a question about your documents..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
            />

            <button
              className="capsule-btn send-btn"
              onClick={() => handleSendMessage()}
              disabled={!input.trim() || isStreaming}
              title="Send Prompt"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="12" y1="19" x2="12" y2="5" />
                <polyline points="5 12 12 5 19 12" />
              </svg>
            </button>
          </div>

          <div className="input-disclaimer">
            Deep RAG answers strictly with verified facts extracted from your documents.
          </div>
        </div>
      </main>

      {/* =====================================================================
          UPLOAD MODAL
          ===================================================================== */}
      {uploadModalOpen && (
        <div className="modal-backdrop" onClick={() => setUploadModalOpen(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">Upload Documents</h2>
              <button className="icon-btn" onClick={() => setUploadModalOpen(false)}>
                ✕
              </button>
            </div>

            <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
              Upload files to index them into ChromaDB. Supports PDF, Word documents (.docx), Markdown (.md),
              plain text, CSV, JSON, and source code.
            </p>

            <label className="dropzone">
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.docx,.txt,.md,.csv,.json,.py,.js,.html"
                onChange={handleFileUpload}
                style={{ display: "none" }}
              />
              <div className="dropzone-icon">📁</div>
              <div className="dropzone-text">
                {isUploading ? "Processing & Embedding Chunks..." : "Click or Drag Files Here to Upload"}
              </div>
              <div className="dropzone-hint">
                PDF, DOCX, TXT, MD, CSV, JSON, Code up to 50MB
              </div>
            </label>

            <div style={{ display: "flex", justifyContent: "flex-end", gap: "10px" }}>
              <button
                className="secondary-action-btn"
                style={{ width: "auto" }}
                onClick={() => setUploadModalOpen(false)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
