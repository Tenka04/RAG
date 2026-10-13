"use client";

import React, { useState, useEffect } from "react";
import { DevLogEntry, EmbeddingTestResult, SystemDiagnostics } from "@/types";

interface DevConsoleDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  onLockDevMode: () => void;
}

export const DevConsoleDrawer: React.FC<DevConsoleDrawerProps> = ({
  isOpen,
  onClose,
  onLockDevMode,
}) => {
  const [activeTab, setActiveTab] = useState<"logs" | "embed" | "chroma" | "system">("logs");
  const [logs, setLogs] = useState<DevLogEntry[]>([]);
  const [logFilter, setLogFilter] = useState<string>("ALL");
  const [isLoadingLogs, setIsLoadingLogs] = useState(false);
  const [expandedLogId, setExpandedLogId] = useState<string | null>(null);

  // Embedding Playground
  const [testText, setTestText] = useState("Retrieval-Augmented Generation combines search with neural language models.");
  const [embedResult, setEmbedResult] = useState<EmbeddingTestResult | null>(null);
  const [isTestingEmbed, setIsTestingEmbed] = useState(false);
  const [embedError, setEmbedError] = useState<string | null>(null);

  // ChromaDB chunks
  const [chunks, setChunks] = useState<any[]>([]);
  const [isLoadingChunks, setIsLoadingChunks] = useState(false);

  // System Diagnostics
  const [diagnostics, setDiagnostics] = useState<SystemDiagnostics | null>(null);
  const [pingLatency, setPingLatency] = useState<number | null>(null);

  // Fetch logs
  const fetchLogs = async () => {
    setIsLoadingLogs(true);
    try {
      const url = logFilter === "ALL" ? "/api/dev/logs" : `/api/dev/logs?level=${logFilter}`;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setLogs(data.logs || []);
      }
    } catch {
      // Backend starting or offline
    } finally {
      setIsLoadingLogs(false);
    }
  };

  // Fetch ChromaDB chunks
  const fetchChunks = async () => {
    setIsLoadingChunks(true);
    try {
      const res = await fetch("/api/dev/collection-chunks?limit=50");
      if (res.ok) {
        const data = await res.json();
        setChunks(data.chunks || []);
      }
    } catch {
      // Ignore
    } finally {
      setIsLoadingChunks(false);
    }
  };

  // Fetch diagnostics
  const fetchDiagnostics = async () => {
    const t0 = performance.now();
    try {
      const res = await fetch("/api/dev/diagnostics");
      const latency = Math.round(performance.now() - t0);
      setPingLatency(latency);
      if (res.ok) {
        const data = await res.json();
        setDiagnostics(data);
      }
    } catch {
      setPingLatency(null);
    }
  };

  // Run embedding test
  const handleRunEmbeddingTest = async () => {
    if (!testText.trim()) return;
    setIsTestingEmbed(true);
    setEmbedError(null);
    try {
      const res = await fetch("/api/dev/embed-test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: testText.trim() }),
      });
      const data = await res.json();
      if (res.ok && data.success) {
        setEmbedResult(data);
      } else {
        setEmbedError(data.error || "Failed to generate embedding");
      }
    } catch {
      setEmbedError("Unable to communicate with backend server.");
    } finally {
      setIsTestingEmbed(false);
    }
  };

  // Simulate an error
  const handleSimulateError = async () => {
    try {
      await fetch("/api/dev/logs/simulate-error", { method: "POST" });
      await fetchLogs();
    } catch {
      // Ignore
    }
  };

  // Clear logs
  const handleClearLogs = async () => {
    try {
      await fetch("/api/dev/logs/clear", { method: "POST" });
      setLogs([]);
    } catch {
      // Ignore
    }
  };

  useEffect(() => {
    if (isOpen) {
      if (activeTab === "logs") fetchLogs();
      if (activeTab === "chroma") fetchChunks();
      if (activeTab === "system") fetchDiagnostics();
    }
  }, [isOpen, activeTab, logFilter]);

  if (!isOpen) return null;

  return (
    <div className="dev-drawer-backdrop" onClick={onClose}>
      <div className="dev-drawer" onClick={(e) => e.stopPropagation()}>
        {/* Drawer Header */}
        <div className="dev-drawer-header">
          <div className="dev-header-left">
            <span className="dev-header-icon">🛠️</span>
            <div>
              <div className="dev-header-title">Developer Console</div>
              <div className="dev-header-sub">Bugs details, vector embeddings & system telemetry</div>
            </div>
          </div>

          <div className="dev-header-right">
            <button
              className="dev-lock-btn"
              onClick={onLockDevMode}
              title="Lock Developer Mode"
              type="button"
            >
              🔒 Lock Dev Mode
            </button>
            <button
              className="sidebar-icon-btn"
              onClick={onClose}
              title="Close drawer"
              type="button"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Drawer Tabs */}
        <div className="dev-drawer-tabs">
          <button
            className={`dev-main-tab ${activeTab === "logs" ? "active" : ""}`}
            onClick={() => setActiveTab("logs")}
            type="button"
          >
            🐞 Bug & Error Log
            {logs.filter((l) => l.level === "ERROR").length > 0 && (
              <span className="dev-tab-err-badge">
                {logs.filter((l) => l.level === "ERROR").length}
              </span>
            )}
          </button>
          <button
            className={`dev-main-tab ${activeTab === "embed" ? "active" : ""}`}
            onClick={() => setActiveTab("embed")}
            type="button"
          >
            🧬 Embedding Explorer
          </button>
          <button
            className={`dev-main-tab ${activeTab === "chroma" ? "active" : ""}`}
            onClick={() => setActiveTab("chroma")}
            type="button"
          >
            🗄️ ChromaDB Internals
          </button>
          <button
            className={`dev-main-tab ${activeTab === "system" ? "active" : ""}`}
            onClick={() => setActiveTab("system")}
            type="button"
          >
            ⚙️ System Health
          </button>
        </div>

        {/* Tab 1: Bug & Error Log */}
        {activeTab === "logs" && (
          <div className="dev-drawer-body">
            <div className="dev-toolbar">
              <div className="dev-filter-group">
                {["ALL", "ERROR", "WARN", "INFO", "DEBUG"].map((lvl) => (
                  <button
                    key={lvl}
                    className={`dev-filter-pill ${logFilter === lvl ? "active" : ""} ${lvl.toLowerCase()}`}
                    onClick={() => setLogFilter(lvl)}
                    type="button"
                  >
                    {lvl}
                  </button>
                ))}
              </div>

              <div className="dev-actions-group">
                <button className="dev-action-btn" onClick={fetchLogs} disabled={isLoadingLogs} type="button">
                  ↻ Refresh
                </button>
                <button className="dev-action-btn warn" onClick={handleSimulateError} type="button">
                  ⚠️ Simulate Error
                </button>
                <button className="dev-action-btn danger" onClick={handleClearLogs} type="button">
                  🗑️ Clear
                </button>
              </div>
            </div>

            {/* Log Feed */}
            <div className="dev-logs-feed">
              {logs.length === 0 ? (
                <div className="dev-empty-feed">
                  <div style={{ fontSize: "24px" }}>📋</div>
                  <div>No logs found matching filter &quot;{logFilter}&quot;.</div>
                  <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                    Interact with the app or click &quot;Simulate Error&quot; to test stack trace logging.
                  </div>
                </div>
              ) : (
                logs.map((item) => (
                  <div
                    key={item.id}
                    className={`dev-log-card ${item.level.toLowerCase()}`}
                    onClick={() => setExpandedLogId(expandedLogId === item.id ? null : item.id)}
                  >
                    <div className="dev-log-top">
                      <span className={`dev-log-tag ${item.level.toLowerCase()}`}>{item.level}</span>
                      <span className="dev-log-time">{item.timestamp}</span>
                      {item.endpoint && <span className="dev-log-ep">{item.endpoint}</span>}
                      {item.duration_ms !== undefined && (
                        <span className="dev-log-dur">{item.duration_ms}ms</span>
                      )}
                    </div>
                    <div className="dev-log-msg">{item.message}</div>

                    {/* Expandable Traceback */}
                    {item.traceback && (
                      <div className="dev-trace-section">
                        <div className="dev-trace-label">
                          {expandedLogId === item.id ? "Hide Stack Trace ▲" : "View Exception Traceback ▼"}
                        </div>
                        {expandedLogId === item.id && (
                          <pre className="dev-trace-pre">
                            <code>{item.traceback}</code>
                          </pre>
                        )}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* Tab 2: Embedding Playground */}
        {activeTab === "embed" && (
          <div className="dev-drawer-body">
            <div className="dev-embed-playground">
              <label className="dev-label">Test Text for Embedding Vector Analysis:</label>
              <textarea
                className="dev-embed-textarea"
                rows={3}
                value={testText}
                onChange={(e) => setTestText(e.target.value)}
                placeholder="Type or paste any text to test vector generation..."
              />

              <div style={{ display: "flex", justifyContent: "flex-end", marginTop: "10px" }}>
                <button
                  className="dev-submit-btn"
                  onClick={handleRunEmbeddingTest}
                  disabled={!testText.trim() || isTestingEmbed}
                  type="button"
                >
                  {isTestingEmbed ? "Computing Vector..." : "🧬 Compute Vector Embedding"}
                </button>
              </div>

              {embedError && <div className="dev-auth-error">{embedError}</div>}

              {embedResult && (
                <div className="dev-embed-result-card">
                  <div className="dev-metrics-grid">
                    <div className="dev-metric-card">
                      <div className="dev-metric-lbl">Vector Model</div>
                      <div className="dev-metric-val code" style={{ fontSize: "13.5px" }}>
                        {embedResult.model}
                      </div>
                    </div>
                    <div className="dev-metric-card">
                      <div className="dev-metric-lbl">Dimensions</div>
                      <div className="dev-metric-val highlight">{embedResult.dimensions} dims</div>
                    </div>
                    <div className="dev-metric-card">
                      <div className="dev-metric-lbl">L2 Norm</div>
                      <div className="dev-metric-val">{embedResult.norm}</div>
                    </div>
                    <div className="dev-metric-card">
                      <div className="dev-metric-lbl">Compute Latency</div>
                      <div className="dev-metric-val">{embedResult.latency_ms} ms</div>
                    </div>
                  </div>

                  <div className="dev-vector-preview-box" style={{ marginTop: "16px" }}>
                    <div className="dev-preview-heading">Sample Vector Floats (First 24 values):</div>
                    <div className="dev-vector-pills">
                      {embedResult.sample_vector.map((val, idx) => (
                        <span key={idx} className={`dev-float-pill ${val < 0 ? "neg" : "pos"}`}>
                          [{idx}]: {val.toFixed(5)}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab 3: ChromaDB Internals */}
        {activeTab === "chroma" && (
          <div className="dev-drawer-body">
            <div className="dev-toolbar">
              <div style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
                Displaying indexed chunks from ChromaDB collection
              </div>
              <button className="dev-action-btn" onClick={fetchChunks} disabled={isLoadingChunks} type="button">
                ↻ Reload Chunks
              </button>
            </div>

            <div className="dev-chunks-feed">
              {chunks.length === 0 ? (
                <div className="dev-empty-feed">
                  <div style={{ fontSize: "24px" }}>📂</div>
                  <div>No documents indexed in ChromaDB yet.</div>
                  <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                    Upload files or click &quot;Load Sample Docs&quot; in the sidebar to populate chunks.
                  </div>
                </div>
              ) : (
                chunks.map((chk, i) => (
                  <div key={i} className="dev-chunk-card">
                    <div className="dev-chunk-header">
                      <div className="dev-chunk-title">
                        <span className="dev-chunk-idx">#{i + 1}</span>
                        <span className="dev-chunk-name">{chk.metadata?.source || "Document"}</span>
                        <span className="dev-chunk-id code">{chk.id}</span>
                      </div>
                      <span className="dev-len-badge">{chk.char_count} chars</span>
                    </div>
                    <div className="dev-chunk-preview">{chk.content}</div>
                    {chk.metadata && (
                      <div className="dev-chunk-meta-chips">
                        {Object.entries(chk.metadata).map(([k, v]) => (
                          <span key={k} className="dev-meta-chip">
                            <strong>{k}:</strong> {String(v)}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* Tab 4: System Health & Diagnostics */}
        {activeTab === "system" && (
          <div className="dev-drawer-body">
            <div className="dev-toolbar">
              <div style={{ fontSize: "13.5px", fontWeight: 500 }}>System Health &amp; Runtime Telemetry</div>
              <button className="dev-action-btn" onClick={fetchDiagnostics} type="button">
                ↻ Ping Diagnostics
              </button>
            </div>

            {diagnostics ? (
              <div className="dev-diagnostics-container">
                <div className="dev-diag-section">
                  <div className="dev-diag-heading">RAG Pipeline Health</div>
                  <div className="dev-diag-row">
                    <span>Ollama LLM Engine:</span>
                    <span className={diagnostics.ollama_online ? "status-ok" : "status-warn"}>
                      {diagnostics.ollama_online ? "🟢 Online" : "🟡 Offline (Fallback Grounded Synthesis Active)"}
                    </span>
                  </div>
                  <div className="dev-diag-row">
                    <span>Ollama Host:</span>
                    <span className="code">
                      {diagnostics.ollama_host || "http://localhost:11434"}
                      {diagnostics.is_cloud && " (☁️ Ollama Cloud)"}
                    </span>
                  </div>
                  {diagnostics.api_key_configured && (
                    <div className="dev-diag-row">
                      <span>Cloud API Key:</span>
                      <span className="code" style={{ color: "#10a37f" }}>
                        🔑 {diagnostics.api_key_masked} (Authenticated)
                      </span>
                    </div>
                  )}
                  <div className="dev-diag-row">
                    <span>Default Model:</span>
                    <span className="code">{diagnostics.default_llm_model}</span>
                  </div>
                  <div className="dev-diag-row">
                    <span>Embedding Engine:</span>
                    <span className="code">{diagnostics.embedding_model} (ONNX Fallback Active)</span>
                  </div>
                  <div className="dev-diag-row">
                    <span>Backend Ping Latency:</span>
                    <span className="code">{pingLatency !== null ? `${pingLatency} ms` : "N/A"}</span>
                  </div>
                </div>

                <div className="dev-diag-section">
                  <div className="dev-diag-heading">Vector Database (ChromaDB)</div>
                  <div className="dev-diag-row">
                    <span>ChromaDB Version:</span>
                    <span className="code">{diagnostics.chromadb_version}</span>
                  </div>
                  <div className="dev-diag-row">
                    <span>Collection:</span>
                    <span className="code">{diagnostics.collection_name}</span>
                  </div>
                  <div className="dev-diag-row">
                    <span>Total Indexed Chunks:</span>
                    <span className="highlight">{diagnostics.total_chunks}</span>
                  </div>
                  <div className="dev-diag-row">
                    <span>Persist Directory:</span>
                    <span className="code" style={{ wordBreak: "break-all" }}>{diagnostics.persist_dir}</span>
                  </div>
                </div>

                <div className="dev-diag-section">
                  <div className="dev-diag-heading">Environment &amp; Models</div>
                  <div className="dev-diag-row">
                    <span>Python Runtime:</span>
                    <span className="code">{diagnostics.python_version}</span>
                  </div>
                  <div className="dev-diag-row">
                    <span>Server Timestamp:</span>
                    <span>{diagnostics.server_time}</span>
                  </div>
                  <div className="dev-diag-row">
                    <span>Recorded Errors:</span>
                    <span className={diagnostics.error_count > 0 ? "status-warn" : "status-ok"}>
                      {diagnostics.error_count}
                    </span>
                  </div>
                  {diagnostics.available_ollama_models && diagnostics.available_ollama_models.length > 0 && (
                    <div style={{ marginTop: "8px" }}>
                      <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>Installed Ollama Models:</span>
                      <div className="dev-models-tags" style={{ marginTop: "4px" }}>
                        {diagnostics.available_ollama_models.map((m, idx) => (
                          <span key={idx} className="dev-model-pill code">{m}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="dev-empty-feed">Checking diagnostics...</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
