"use client";

import React, { useRef } from "react";

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  isUploading: boolean;
}

export const UploadModal: React.FC<UploadModalProps> = ({
  isOpen,
  onClose,
  onUpload,
  isUploading,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  return (
    <div className="gpt-modal-overlay" onClick={onClose}>
      <div className="gpt-modal-box" onClick={(e) => e.stopPropagation()}>
        <div className="modal-title-row">
          <div className="modal-main-title">Add to Knowledge Base</div>
          <button className="sidebar-icon-btn" onClick={onClose} type="button">
            ✕
          </button>
        </div>

        <p style={{ fontSize: "13.5px", color: "var(--text-secondary)", lineHeight: 1.5 }}>
          Upload local documents to index into the vector store. Supports PDF (.pdf), Word (.docx), Markdown (.md), plain text (.txt), CSV, JSON, and code files.
        </p>

        <label className="gpt-upload-zone">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.txt,.md,.csv,.json,.py,.js,.html"
            onChange={onUpload}
            style={{ display: "none" }}
          />
          <div style={{ fontSize: "28px" }}>📁</div>
          <div style={{ fontSize: "14px", fontWeight: 500, color: "var(--text-primary)" }}>
            {isUploading ? "Chunking & vector indexing..." : "Click to select or drop files here"}
          </div>
          <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>
            Embedded and indexed for instant retrieval
          </div>
        </label>

        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button
            className="kb-action-btn"
            style={{ width: "auto", padding: "8px 16px" }}
            onClick={onClose}
            type="button"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
