"use client";

import React, { useState } from "react";
import { DevTelemetry } from "@/types";

interface DevInspectorProps {
  telemetry: DevTelemetry;
}

export const DevInspector: React.FC<DevInspectorProps> = ({ telemetry }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<"timing" | "embeddings" | "chunks" | "prompt" | "bugs">("timing");
  const [copiedPrompt, setCopiedPrompt] = useState(false);

  const handleCopyPrompt = () => {
    if (telemetry.prompt_preview) {
      navigator.clipboard.writeText(telemetry.prompt_preview);
      setCopiedPrompt(true);
      setTimeout(() => setCopiedPrompt(false), 2000);
    }
  };

  const hasBugOrFallback = telemetry.used_fallback || Boolean(telemetry.fallback_reason);

  return (
    <div className="dev-inspector-container">
      {/* Dev Toggle Pill */}
      <button
        className={`dev-inspector-toggle-btn ${isOpen ? "open" : ""} ${hasBugOrFallback ? "has-warn" : ""}`}
        onClick={() => setIsOpen(!isOpen)}
        type="button"
      >
        <div className="dev-toggle-left">
          <span className="dev-badge-chip">DEV</span>
          <span className="dev-metric-summary">
            ⏱️ {telemetry.total_time_ms ? `${telemetry.total_time_ms}ms` : "telemetry"}
            {telemetry.query_embedding?.dimensions ? ` • 🧬 ${telemetry.query_embedding.dimensions}d` : ""}
            {telemetry.chunks_retrieved_count !== undefined ? ` • 📄 ${telemetry.chunks_retrieved_count} chunks` : ""}
          </span>
          {hasBugOrFallback && <span className="dev-warn-indicator">⚠️ Fallback Used</span>}
        </div>
        <span className="dev-toggle-arrow">{isOpen ? "Hide ▲" : "Inspect ▼"}</span>
      </button>

      {/* Expanded Dev Panel */}
      {isOpen && (
        <div className="dev-inspector-panel">
          {/* Subtabs */}
          <div className="dev-panel-tabs">
            <button
              className={`dev-subtab-btn ${activeTab === "timing" ? "active" : ""}`}
              onClick={() => setActiveTab("timing")}
              type="button"
            >
              ⏱️ Latency & Timing
            </button>
            <button
              className={`dev-subtab-btn ${activeTab === "embeddings" ? "active" : ""}`}
              onClick={() => setActiveTab("embeddings")}
              type="button"
            >
              🧬 Embeddings
            </button>
            <button
              className={`dev-subtab-btn ${activeTab === "chunks" ? "active" : ""}`}
              onClick={() => setActiveTab("chunks")}
              type="button"
            >
              🔍 Chunks ({telemetry.raw_chunks?.length || 0})
            </button>
            <button
              className={`dev-subtab-btn ${activeTab === "prompt" ? "active" : ""}`}
              onClick={() => setActiveTab("prompt")}
              type="button"
            >
              📜 Raw Prompt
            </button>
            {hasBugOrFallback && (
              <button
                className={`dev-subtab-btn warn ${activeTab === "bugs" ? "active" : ""}`}
                onClick={() => setActiveTab("bugs")}
                type="button"
              >
                ⚠️ Bug Trace
              </button>
            )}
          </div>

          {/* Tab 1: Timing & Latency */}
          {activeTab === "timing" && (
            <div className="dev-tab-body">
              <div className="dev-metrics-grid">
                <div className="dev-metric-card">
                  <div className="dev-metric-lbl">Total Time</div>
                  <div className="dev-metric-val highlight">{telemetry.total_time_ms ?? "N/A"} ms</div>
                  <div className="dev-metric-sub">End-to-end execution</div>
                </div>
                <div className="dev-metric-card">
                  <div className="dev-metric-lbl">Vector Retrieval</div>
                  <div className="dev-metric-val">{telemetry.retrieval_time_ms ?? "N/A"} ms</div>
                  <div className="dev-metric-sub">ChromaDB query</div>
                </div>
                <div className="dev-metric-card">
                  <div className="dev-metric-lbl">Embedding Latency</div>
                  <div className="dev-metric-val">{telemetry.embedding_time_ms ?? "N/A"} ms</div>
                  <div className="dev-metric-sub">Vector generation</div>
                </div>
                <div className="dev-metric-card">
                  <div className="dev-metric-lbl">LLM Generation</div>
                  <div className="dev-metric-val">{telemetry.llm_time_ms ?? "N/A"} ms</div>
                  <div className="dev-metric-sub">{telemetry.model_used || "Active model"}</div>
                </div>
              </div>

              {/* Pipeline Progress Bar */}
              {telemetry.total_time_ms && telemetry.total_time_ms > 0 && (
                <div className="dev-latency-bar-wrap">
                  <div className="dev-bar-labels">
                    <span>Retrieval ({Math.round(((telemetry.retrieval_time_ms || 0) / telemetry.total_time_ms) * 100)}%)</span>
                    <span>Generation ({Math.round(((telemetry.llm_time_ms || 0) / telemetry.total_time_ms) * 100)}%)</span>
                  </div>
                  <div className="dev-latency-bar">
                    <div
                      className="dev-bar-segment retrieval"
                      style={{ width: `${Math.max(5, Math.min(95, ((telemetry.retrieval_time_ms || 0) / telemetry.total_time_ms) * 100))}%` }}
                      title={`Retrieval: ${telemetry.retrieval_time_ms}ms`}
                    />
                    <div
                      className="dev-bar-segment generation"
                      style={{ width: `${Math.max(5, Math.min(95, ((telemetry.llm_time_ms || 0) / telemetry.total_time_ms) * 100))}%` }}
                      title={`Generation: ${telemetry.llm_time_ms}ms`}
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Tab 2: Embeddings Details */}
          {activeTab === "embeddings" && (
            <div className="dev-tab-body">
              <div className="dev-embedding-summary">
                <div className="dev-keyval-row">
                  <span className="dev-key">Embedding Model:</span>
                  <span className="dev-val code">{telemetry.query_embedding?.model || "chroma-onnx-default"}</span>
                </div>
                <div className="dev-keyval-row">
                  <span className="dev-key">Vector Dimensions:</span>
                  <span className="dev-val code">{telemetry.query_embedding?.dimensions || 0} dims</span>
                </div>
                <div className="dev-keyval-row">
                  <span className="dev-key">L2 Norm:</span>
                  <span className="dev-val code">{telemetry.query_embedding?.norm ?? "1.0000"}</span>
                </div>
              </div>

              {telemetry.query_embedding?.sample_values && telemetry.query_embedding.sample_values.length > 0 && (
                <div className="dev-vector-preview-box">
                  <div className="dev-preview-heading">Vector Values Preview (First 16 floats):</div>
                  <div className="dev-vector-pills">
                    {telemetry.query_embedding.sample_values.map((v, i) => (
                      <span key={i} className={`dev-float-pill ${v < 0 ? "neg" : "pos"}`}>
                        [{i}]: {v.toFixed(4)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Tab 3: Retrieved Chunks with Raw Distances */}
          {activeTab === "chunks" && (
            <div className="dev-tab-body">
              {telemetry.raw_chunks && telemetry.raw_chunks.length > 0 ? (
                <div className="dev-chunks-list">
                  {telemetry.raw_chunks.map((chk, i) => (
                    <div key={i} className="dev-chunk-card">
                      <div className="dev-chunk-header">
                        <div className="dev-chunk-title">
                          <span className="dev-chunk-idx">#{i + 1}</span>
                          <span className="dev-chunk-name">{chk.source}</span>
                          <span className="dev-chunk-id code">ID: {chk.id}</span>
                        </div>
                        <div className="dev-chunk-metrics">
                          <span className="dev-dist-badge">Distance: {chk.raw_distance ?? "N/A"}</span>
                          <span className="dev-score-badge">Score: {chk.score ?? "N/A"}</span>
                          <span className="dev-len-badge">{chk.char_count} chars</span>
                        </div>
                      </div>
                      <div className="dev-chunk-preview">{chk.content_preview}</div>
                      {chk.metadata && Object.keys(chk.metadata).length > 0 && (
                        <div className="dev-chunk-meta-chips">
                          {Object.entries(chk.metadata).map(([k, v]) => (
                            <span key={k} className="dev-meta-chip">
                              <strong>{k}:</strong> {String(v)}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="dev-empty-msg">No chunks retrieved for this query.</div>
              )}
            </div>
          )}

          {/* Tab 4: Raw Prompt */}
          {activeTab === "prompt" && (
            <div className="dev-tab-body">
              <div className="dev-prompt-actions">
                <span className="dev-prompt-sub">Exact system instructions and context sent to the LLM:</span>
                <button className="dev-copy-btn" onClick={handleCopyPrompt} type="button">
                  {copiedPrompt ? "✓ Copied" : "Copy Full Prompt"}
                </button>
              </div>
              <pre className="dev-prompt-pre">
                <code>{telemetry.prompt_preview || "No prompt recorded"}</code>
              </pre>
            </div>
          )}

          {/* Tab 5: Bugs & Fallback Tracing */}
          {activeTab === "bugs" && hasBugOrFallback && (
            <div className="dev-tab-body">
              <div className="dev-error-alert-box">
                <div className="dev-error-head">⚠️ Fallback Synthesis Triggered</div>
                <div className="dev-error-msg">
                  {telemetry.fallback_reason || "Ollama service was unreachable. Query was synthesized via grounded document extraction."}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
