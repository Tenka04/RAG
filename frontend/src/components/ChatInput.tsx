"use client";

import React, { useRef, useEffect } from "react";

interface ChatInputProps {
  input: string;
  setInput: (val: string) => void;
  onSend: () => void;
  isStreaming: boolean;
  onAttachClick: () => void;
  placeholder?: string;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  input,
  setInput,
  onSend,
  isStreaming,
  onAttachClick,
  placeholder = "Ask Assist Me anything about your documents...",
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
    <div className="gpt-input-fixed-bar">
      <div className="gpt-input-capsule">
        <button
          className="capsule-plus-btn"
          onClick={onAttachClick}
          title="Add documents"
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
          placeholder={placeholder}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
        />

        <button
          className="gpt-send-btn"
          onClick={onSend}
          disabled={!input.trim() || isStreaming}
          title="Send message"
          type="button"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="12" y1="19" x2="12" y2="5" />
            <polyline points="5 12 12 5 19 12" />
          </svg>
        </button>
      </div>

      <div className="gpt-disclaimer-text mono-text">
        Assist Me strictly verifies responses with indexed knowledge base chunks.
      </div>
    </div>
  );
};
