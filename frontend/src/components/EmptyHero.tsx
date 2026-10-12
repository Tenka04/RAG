"use client";

import React, { useRef, useEffect } from "react";

interface EmptyHeroProps {
  input: string;
  setInput: (val: string) => void;
  onSend: (text?: string) => void;
  isStreaming: boolean;
  onAttachClick: () => void;
}

const QUICK_PILLS = [
  { label: "📄 Summarize documents", prompt: "Summarize the key takeaways and topics across all uploaded documents." },
  { label: "🧠 Explain RAG pipeline", prompt: "Explain the 5 core stages of Retrieval-Augmented Generation described in the files." },
  { label: "🔍 Check indexed files", prompt: "List the documents currently indexed in ChromaDB and their main topics." },
  { label: "⚡ Recommended GPU models", prompt: "What Ollama models are recommended for local PCs with 4GB VRAM?" },
];

export const EmptyHero: React.FC<EmptyHeroProps> = ({
  input,
  setInput,
  onSend,
  isStreaming,
  onAttachClick,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [input]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <div className="empty-hero-container">
      <div className="assist-me-badge mono-text">ASSIST ME // v2.0</div>
      <h1 className="empty-hero-title">What can I help with?</h1>

      {/* Centered Input Capsule in Empty State */}
      <div className="empty-input-capsule-wrapper">
        <div className="gpt-input-capsule">
          <button
            className="capsule-plus-btn"
            onClick={onAttachClick}
            title="Attach files"
            type="button"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
          </button>

          <textarea
            ref={textareaRef}
            rows={1}
            className="gpt-textarea mono-input"
            placeholder="Ask Assist Me anything about your documents..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />

          <button
            className="gpt-send-btn"
            onClick={() => onSend()}
            disabled={!input.trim() || isStreaming}
            title="Send prompt"
            type="button"
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
        {QUICK_PILLS.map((pill, idx) => (
          <button
            key={idx}
            className="hero-pill-btn mono-text"
            onClick={() => onSend(pill.prompt)}
            type="button"
          >
            {pill.label}
          </button>
        ))}
      </div>
    </div>
  );
};
