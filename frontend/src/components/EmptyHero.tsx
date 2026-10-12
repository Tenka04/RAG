"use client";

import React, { useRef, useEffect } from "react";

interface EmptyHeroProps {
  input: string;
  setInput: (val: string) => void;
  onSend: (text?: string) => void;
  isStreaming: boolean;
  onAttachClick: () => void;
}

const SUGGESTIONS = [
  {
    icon: (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="16" y1="13" x2="8" y2="13" />
        <line x1="16" y1="17" x2="8" y2="17" />
        <polyline points="10 9 9 9 8 9" />
      </svg>
    ),
    label: "Summarize documents",
    prompt: "Summarize the key takeaways and topics across all uploaded documents.",
  },
  {
    icon: (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="16" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12.01" y2="8" />
      </svg>
    ),
    label: "Explain RAG pipeline",
    prompt: "Explain the 5 core stages of Retrieval-Augmented Generation described in the files.",
  },
  {
    icon: (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="11" cy="11" r="8" />
        <line x1="21" y1="21" x2="16.65" y2="16.65" />
      </svg>
    ),
    label: "Inspect indexed files",
    prompt: "List the documents currently indexed in ChromaDB and their main topics.",
  },
  {
    icon: (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
      </svg>
    ),
    label: "Local 4GB GPU models",
    prompt: "What Ollama models are recommended for local PCs with 4GB VRAM?",
  },
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
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [input]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isStreaming && input.trim()) {
        onSend();
      }
    }
  };

  return (
    <div className="empty-hero-container">
      <h1 className="empty-hero-title">What can I help with?</h1>

      {/* Centered Large Input Capsule in Empty State */}
      <div className="empty-input-capsule-wrapper">
        <div className="gpt-input-capsule">
          <button
            className="capsule-plus-btn"
            onClick={onAttachClick}
            title="Attach documents"
            type="button"
            aria-label="Attach documents"
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
            placeholder="Message Assist Me..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            autoFocus
          />

          <button
            className="gpt-send-btn"
            onClick={() => onSend()}
            disabled={!input.trim() || isStreaming}
            title="Send prompt"
            type="button"
            aria-label="Send prompt"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="19" x2="12" y2="5" />
              <polyline points="5 12 12 5 19 12" />
            </svg>
          </button>
        </div>
      </div>

      {/* Suggestion Pills */}
      <div className="hero-pills-row">
        {SUGGESTIONS.map((item, idx) => (
          <button
            key={idx}
            className="hero-pill-btn"
            onClick={() => onSend(item.prompt)}
            type="button"
          >
            <span className="hero-pill-icon">{item.icon}</span>
            <span>{item.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
};
