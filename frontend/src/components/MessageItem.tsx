"use client";

import React, { useState } from "react";
import { Message } from "@/types";
import { SourcesView } from "./SourcesView";
import { MarkdownRenderer } from "./MarkdownRenderer";

interface MessageItemProps {
  message: Message;
  isStreaming?: boolean;
  onRegenerate?: () => void;
}

export const MessageItem: React.FC<MessageItemProps> = ({
  message,
  isStreaming,
  onRegenerate,
}) => {
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (message.role === "user") {
    return (
      <div className="msg-row user">
        <div className="user-bubble">{message.content}</div>
      </div>
    );
  }

  return (
    <div className="msg-row assistant">
      <div className="assistant-spark-avatar" title="Assist Me">
        {/* OpenAI Sparkle / Whirl emblem */}
        <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor">
          <path d="M22.28 9.82a6 6 0 0 0-.52-4.91 6.05 6.05 0 0 0-6.51-2.9A6.06 6.06 0 0 0 4.98 4.18a6 6 0 0 0-4 2.9 6.05 6.05 0 0 0 .74 7.1 6 6 0 0 0 .51 4.91 6.05 6.05 0 0 0 6.52 2.9A6 6 0 0 0 13.26 24a6.05 6.05 0 0 0 5.77-4.2 6 6 0 0 0 4-2.9 6.05 6.05 0 0 0-.75-7.08zM13.26 22.43a4.48 4.48 0 0 1-2.88-1.04l.14-.08 4.78-2.76a.8.8 0 0 0 .4-.68v-6.74l2.02 1.17a.07.07 0 0 1 .04.05v5.58a4.5 4.5 0 0 1-4.5 4.5zm-9.66-4.13a4.47 4.47 0 0 1-.53-3.01l.14.08 4.78 2.76a.77.77 0 0 0 .78 0l5.85-3.37v2.33a.08.08 0 0 1-.03.06L9.74 19.95a4.5 4.5 0 0 1-6.14-1.65zM2.34 7.9a4.48 4.48 0 0 1 2.37-1.97v5.67a.77.77 0 0 0 .39.68l5.81 3.35-2.02 1.17a.08.08 0 0 1-.07 0l-4.83-2.79A4.5 4.5 0 0 1 2.34 7.9zm16.6 3.85L13.1 8.38l2.02-1.16a.08.08 0 0 1 .07 0l4.83 2.79a4.49 4.49 0 0 1-.68 8.1v-5.67a.79.79 0 0 0-.4-.59zm2.01-3.02l-.14-.08-4.78-2.79a.78.78 0 0 0-.78 0L9.41 9.23V6.9a.07.07 0 0 1 .03-.06l4.83-2.79a4.5 4.5 0 0 1 6.68 4.67zm-12.64 4.14l-2.02-1.17a.08.08 0 0 1-.04-.05V6.08a4.5 4.5 0 0 1 7.37-3.46l-.14.08-4.78 2.76a.8.8 0 0 0-.4.68zm1.1-2.37l2.6-1.5 2.61 1.5v3l-2.61 1.5-2.6-1.5z" />
        </svg>
      </div>

      <div className="assistant-content-wrapper">
        <MarkdownRenderer content={message.content} isStreaming={isStreaming} />

        {/* Source Citations */}
        {message.sources && message.sources.length > 0 && (
          <SourcesView sources={message.sources} />
        )}

        {/* Action Bar */}
        {message.content && !isStreaming && (
          <div className="msg-action-bar">
            <button
              className="action-bar-btn"
              onClick={handleCopy}
              title={copied ? "Copied!" : "Copy"}
              type="button"
            >
              {copied ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10a37f" strokeWidth="2.5">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              ) : (
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                </svg>
              )}
            </button>

            <button
              className={`action-bar-btn ${feedback === "up" ? "active-green" : ""}`}
              onClick={() => setFeedback(feedback === "up" ? null : "up")}
              title="Good response"
              type="button"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
              </svg>
            </button>

            <button
              className={`action-bar-btn ${feedback === "down" ? "active-red" : ""}`}
              onClick={() => setFeedback(feedback === "down" ? null : "down")}
              title="Bad response"
              type="button"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3" />
              </svg>
            </button>

            {onRegenerate && (
              <button
                className="action-bar-btn"
                onClick={onRegenerate}
                title="Regenerate response"
                type="button"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2" />
                </svg>
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
