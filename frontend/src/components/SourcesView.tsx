"use client";

import React, { useState } from "react";
import { SourceItem } from "@/types";

interface SourcesViewProps {
  sources: SourceItem[];
}

export const SourcesView: React.FC<SourcesViewProps> = ({ sources }) => {
  const [expanded, setExpanded] = useState(false);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="sources-carousel">
      <div
        className="sources-header-pill"
        onClick={() => setExpanded(!expanded)}
        title="Toggle retrieved sources"
      >
        <span>📚</span>
        <span>{sources.length} sources referenced</span>
        <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
          {expanded ? "Hide ▲" : "Inspect ▼"}
        </span>
      </div>

      {expanded && (
        <div className="sources-cards-row">
          {sources.map((src, idx) => (
            <div key={idx} className="source-citation-card">
              <div className="source-card-top">
                <span className="source-doc-name" title={src.metadata?.source}>
                  📄 {src.metadata?.source || `Source ${idx + 1}`}
                </span>
                <span className="source-score">
                  {src.score ? `${Math.round(Number(src.score) * 100)}%` : ""}
                </span>
              </div>
              <div className="source-card-preview">{src.content}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
