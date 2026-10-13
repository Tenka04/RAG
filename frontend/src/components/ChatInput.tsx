"use client";

import React, { useRef, useEffect } from "react";

interface ChatInputProps {
  input: string;
  setInput: (val: string) => void;
  onSend: () => void;
  onStop?: () => void;
  isStreaming: boolean;
  onAttachClick: () => void;
  placeholder?: string;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  input,
  setInput,
  onSend,
  onStop,
  isStreaming,
  onAttachClick,
  placeholder = "Message Assist Me...",
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
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
    <div className="gpt-input-fixed-bar">
      <div className="gpt-input-capsule">
        <button
          className="capsule-plus-btn"
          onClick={onAttachClick}
          title="Add files or documents"
          type="button"
          aria-label="Add files"
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
          placeholder={placeholder}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
        />

        {isStreaming ? (
          <button
            className="gpt-stop-btn"
            onClick={onStop}
            title="Stop generating"
            type="button"
            aria-label="Stop generating"
          >
            <div className="gpt-stop-square" />
          </button>
        ) : (
          <button
            className="gpt-send-btn"
            onClick={onSend}
            disabled={!input.trim()}
            title="Send prompt"
            type="button"
            aria-label="Send prompt"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="19" x2="12" y2="5" />
              <polyline points="5 12 12 5 19 12" />
            </svg>
          </button>
        )}
      </div>

      <div className="gpt-disclaimer-text">
        Assist Me can make mistakes. Verify important info from source documents.
      </div>
    </div>
  );
};
