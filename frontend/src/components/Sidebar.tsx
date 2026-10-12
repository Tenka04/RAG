"use client";

import React from "react";
import { ChatSession, KBStats } from "@/types";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onNewChat: () => void;
  sessions: ChatSession[];
  activeSessionId: string;
  onSelectSession: (id: string) => void;
  onDeleteSession: (e: React.MouseEvent, id: string) => void;
  stats: KBStats;
  onIngestSamples: () => void;
  isIngestingSamples: boolean;
  onOpenUpload: () => void;
  onClearKB: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onClose,
  onNewChat,
  sessions,
  activeSessionId,
  onSelectSession,
  onDeleteSession,
  stats,
  onIngestSamples,
  isIngestingSamples,
  onOpenUpload,
  onClearKB,
}) => {
  return (
    <aside className={`gpt-sidebar ${isOpen ? "" : "closed"}`}>
      <div className="sidebar-top-bar">
        <button
          className="sidebar-icon-btn"
          onClick={onClose}
          title="Close sidebar"
          aria-label="Close sidebar"
          type="button"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
            <line x1="9" y1="3" x2="9" y2="21" />
          </svg>
        </button>

        <button
          className="sidebar-icon-btn"
          onClick={onNewChat}
          title="New chat"
          aria-label="New chat"
          type="button"
        >
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 20h9" />
            <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
          </svg>
        </button>
      </div>

      {/* New Chat Button */}
      <div className="new-chat-row">
        <button className="new-chat-button" onClick={onNewChat} type="button">
          <div className="new-chat-left">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 8v8" />
              <path d="M8 12h8" />
            </svg>
            <span className="mono-text">New chat</span>
          </div>
          <span className="mono-text" style={{ fontSize: "11px", color: "var(--text-muted)" }}>Ctrl+K</span>
        </button>
      </div>

      {/* Scrollable History & Documents */}
      <div className="sidebar-scroll-area">
        {/* Chat History */}
        <div>
          <div className="sidebar-group-title mono-text">
            <span>// CHAT SESSIONS</span>
          </div>
          {sessions.map((sess) => (
            <div
              key={sess.id}
              className={`chat-history-item ${sess.id === activeSessionId ? "active" : ""}`}
              onClick={() => onSelectSession(sess.id)}
            >
              <span className="chat-history-title mono-text">{sess.title}</span>
              <button
                className="chat-delete-btn"
                onClick={(e) => onDeleteSession(e, sess.id)}
                title="Delete chat"
                type="button"
              >
                ✕
              </button>
            </div>
          ))}
        </div>

        {/* Knowledge Base */}
        <div>
          <div className="sidebar-group-title mono-text">
            <span>// KNOWLEDGE BASE</span>
            <span className="kb-badge mono-text">{stats.total_chunks} CHUNKS</span>
          </div>

          <div className="kb-card-sidebar">
            <button
              className="kb-action-btn mono-text"
              onClick={onIngestSamples}
              disabled={isIngestingSamples}
              type="button"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
              {isIngestingSamples ? "INDEXING..." : "LOAD SAMPLE DOCS"}
            </button>

            <button
              className="kb-action-btn mono-text"
              onClick={onOpenUpload}
              type="button"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              UPLOAD FILES
            </button>

            {stats.unique_documents && stats.unique_documents.length > 0 && (
              <div style={{ display: "flex", flexDirection: "column", gap: "3px", marginTop: "4px" }}>
                {stats.unique_documents.map((doc, idx) => (
                  <div key={idx} className="kb-doc-pill mono-text" title={doc}>
                    <span>::</span> <span>{doc}</span>
                  </div>
                ))}
              </div>
            )}

            {stats.total_chunks > 0 && (
              <button
                className="kb-action-btn mono-text"
                onClick={onClearKB}
                style={{ color: "#ef4444", marginTop: "4px" }}
                type="button"
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="3 6 5 6 21 6" />
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
                </svg>
                PURGE DATABASE
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Profile / Status Footer */}
      <div className="sidebar-footer-profile">
        <div className="user-profile-btn">
          <div className="user-avatar-circle mono-text">AM</div>
          <div className="user-profile-info">
            <div className="user-profile-name mono-text">Assist Me</div>
            <div className="user-profile-sub mono-text">
              {stats.ollama_online ? "🟢 OLLAMA ACTIVE" : "🟡 GROUNDED MODE"}
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
};
