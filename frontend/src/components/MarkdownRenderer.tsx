"use client";

import React, { useState } from "react";
import { MermaidViewer } from "./widgets/MermaidViewer";
import { ComparisonWidget } from "./widgets/ComparisonWidget";
import { VisualCardWidget } from "./widgets/VisualCardWidget";

interface MarkdownRendererProps {
  content: string;
  isStreaming?: boolean;
}

function normalizeContent(raw: string): string {
  if (!raw) return "";

  // 1. If LLM outputs ```mermaid flowchart TD, split into ```mermaid\nflowchart TD
  let text = raw.replace(/```+mermaid\s+([a-zA-Z0-9_-]+[^\n\r]*)/gi, "```mermaid\n$1");

  // 2. Wrap un-fenced widget:comparison [ ... ] or widget:cards [ ... ] if LLM omitted code backticks
  text = text.replace(/(?:^|\n)(widget:comparison|widget:progress)\s*\n\s*(\[[\s\S]*?\])(?=\n\n|\n[A-Z0-9#]|$)/gi, "\n```$1\n$2\n```\n");
  text = text.replace(/(?:^|\n)(widget:cards|widget:card-list)\s*\n\s*(\[[\s\S]*?\])(?=\n\n|\n[A-Z0-9#]|$)/gi, "\n```$1\n$2\n```\n");

  return text;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, isStreaming }) => {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const handleCopyCode = (code: string, idx: number) => {
    navigator.clipboard.writeText(code);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  if (!content && isStreaming) {
    return (
      <div className="gpt-thinking-state">
        <span className="gpt-pulse-dot" />
        <span className="gpt-pulse-dot" />
        <span className="gpt-pulse-dot" />
      </div>
    );
  }

  // Pre-normalize common LLM formatting edge-cases (e.g. ```mermaid flowchart TD or un-fenced widgets)
  const normalizedContent = normalizeContent(content);

  // Split by code fences: 3+ backticks, flexible info line, CRLF/LF support
  const parts: React.ReactNode[] = [];
  const codeBlockRegex = /(`{3,})([^\n\r]*)\r?\n([\s\S]*?)\1/g;
  let lastIndex = 0;
  let blockIdx = 0;
  let match;

  while ((match = codeBlockRegex.exec(normalizedContent)) !== null) {
    // Text before code block
    if (match.index > lastIndex) {
      const textChunk = normalizedContent.slice(lastIndex, match.index);
      parts.push(renderFormattedText(textChunk, `text_${lastIndex}`, isStreaming));
    }

    const infoLine = (match[2] || "").trim();
    const [firstToken = "", ...restTokens] = infoLine.split(/\s+/);
    const rawLang = firstToken.toLowerCase();
    let codeSnippet = match[3].trimEnd();

    // If info line was "```mermaid flowchart TD", prepend "flowchart TD" to snippet
    if (rawLang === "mermaid" && restTokens.length > 0) {
      codeSnippet = restTokens.join(" ") + "\n" + codeSnippet;
    }

    const normLang = rawLang.replace(/^json:/, "").replace(/[-_]/g, ":");
    const currentBlockIdx = blockIdx++;

    const isMermaid =
      normLang === "mermaid" ||
      normLang === "diagram" ||
      normLang === "flowchart" ||
      /^(flowchart|graph|sequenceDiagram|classDiagram|stateDiagram|erDiagram|journey|gantt|pie|gitGraph)\b/i.test(codeSnippet.trim());

    const isComparison =
      normLang.includes("comparison") ||
      normLang.includes("progress") ||
      normLang.includes("timeline");

    const isCards =
      normLang.includes("card") ||
      normLang.includes("feature");

    if (isMermaid) {
      parts.push(
        <MermaidViewer key={`mermaid_${match.index}`} chart={codeSnippet} />
      );
    } else if (isComparison) {
      try {
        const cleanedJson = codeSnippet.replace(/^```json/i, "").replace(/```$/, "").trim();
        const parsed = JSON.parse(cleanedJson);
        parts.push(
          <ComparisonWidget key={`comp_${match.index}`} items={Array.isArray(parsed) ? parsed : [parsed]} />
        );
      } catch {
        parts.push(
          <div key={`comp_err_${match.index}`} className="gpt-diagram-card">
            <pre className="gpt-code-pre"><code>{codeSnippet}</code></pre>
          </div>
        );
      }
    } else if (isCards) {
      try {
        const cleanedJson = codeSnippet.replace(/^```json/i, "").replace(/```$/, "").trim();
        const parsed = JSON.parse(cleanedJson);
        parts.push(
          <VisualCardWidget key={`cards_${match.index}`} items={Array.isArray(parsed) ? parsed : [parsed]} />
        );
      } catch {
        parts.push(
          <div key={`cards_err_${match.index}`} className="gpt-diagram-card">
            <pre className="gpt-code-pre"><code>{codeSnippet}</code></pre>
          </div>
        );
      }
    } else {
      parts.push(
        <div key={`code_${match.index}`} className="gpt-code-block">
          <div className="gpt-code-header">
            <span className="gpt-code-lang">{rawLang || "code"}</span>
            <button
              type="button"
              className="gpt-code-copy-btn"
              onClick={() => handleCopyCode(codeSnippet, currentBlockIdx)}
            >
              {copiedIndex === currentBlockIdx ? (
                <>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10a37f" strokeWidth="2.5">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  <span style={{ color: "#10a37f" }}>Copied!</span>
                </>
              ) : (
                <>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                  </svg>
                  <span>Copy code</span>
                </>
              )}
            </button>
          </div>
          <pre className="gpt-code-pre">
            <code>{codeSnippet}</code>
          </pre>
        </div>
      );
    }

    lastIndex = match.index + match[0].length;
  }

  // Trailing text and safe streaming handling for unclosed blocks
  if (lastIndex < normalizedContent.length) {
    const trailing = normalizedContent.slice(lastIndex);
    const unclosedMatch = trailing.match(/(`{3,})([^\n\r]*)\r?\n([\s\S]*)$/);

    if (unclosedMatch && isStreaming) {
      const textBefore = trailing.slice(0, unclosedMatch.index);
      if (textBefore) {
        parts.push(renderFormattedText(textBefore, `text_${lastIndex}`, isStreaming));
      }
      const unclosedInfo = (unclosedMatch[2] || "").toLowerCase();
      if (unclosedInfo.includes("mermaid") || unclosedInfo.includes("flowchart") || unclosedInfo.includes("diagram")) {
        parts.push(
          <div key="unclosed_mermaid" className="gpt-diagram-card gpt-diagram-fallback">
            <div className="gpt-diagram-header">
              <span className="gpt-diagram-badge">Drawing Diagram...</span>
            </div>
            <div className="gpt-diagram-loading">
              <span className="gpt-pulse-dot" />
              <span className="gpt-pulse-dot" />
              <span className="gpt-pulse-dot" />
            </div>
          </div>
        );
      } else if (unclosedInfo.includes("widget")) {
        parts.push(
          <div key="unclosed_widget" className="gpt-diagram-card gpt-diagram-fallback">
            <div className="gpt-diagram-header">
              <span className="gpt-diagram-badge">Loading Visual...</span>
            </div>
            <div className="gpt-diagram-loading">
              <span className="gpt-pulse-dot" />
              <span className="gpt-pulse-dot" />
              <span className="gpt-pulse-dot" />
            </div>
          </div>
        );
      } else {
        parts.push(renderFormattedText(trailing, `text_${lastIndex}`, isStreaming));
      }
    } else {
      parts.push(renderFormattedText(trailing, `text_${lastIndex}`, isStreaming));
    }
  }

  return (
    <div className="gpt-markdown-body">
      {parts}
      {isStreaming && <span className="gpt-typing-cursor" />}
    </div>
  );
};

// Helper to render markdown text paragraphs, lists, headers, bold, italics, inline code
function renderFormattedText(text: string, keyPrefix: string, isStreaming?: boolean): React.ReactNode {
  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];
  let inList = false;
  let listItems: React.ReactNode[] = [];
  let isNumberedList = false;

  const flushList = () => {
    if (inList) {
      if (isNumberedList) {
        elements.push(
          <ol key={`${keyPrefix}_ol_${elements.length}`} className="gpt-ol">
            {listItems}
          </ol>
        );
      } else {
        elements.push(
          <ul key={`${keyPrefix}_ul_${elements.length}`} className="gpt-ul">
            {listItems}
          </ul>
        );
      }
      listItems = [];
      inList = false;
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Headers
    if (line.startsWith("### ")) {
      flushList();
      elements.push(
        <h3 key={`${keyPrefix}_h3_${i}`} className="gpt-h3">
          {formatInline(line.slice(4))}
        </h3>
      );
      continue;
    }
    if (line.startsWith("## ")) {
      flushList();
      elements.push(
        <h2 key={`${keyPrefix}_h2_${i}`} className="gpt-h2">
          {formatInline(line.slice(3))}
        </h2>
      );
      continue;
    }
    if (line.startsWith("# ")) {
      flushList();
      elements.push(
        <h1 key={`${keyPrefix}_h1_${i}`} className="gpt-h1">
          {formatInline(line.slice(2))}
        </h1>
      );
      continue;
    }

    // Blockquote (accumulate consecutive blockquote lines)
    if (line.startsWith("> ") || line === ">") {
      flushList();
      const quoteLines: string[] = [line.startsWith("> ") ? line.slice(2) : ""];
      while (i + 1 < lines.length && (lines[i + 1].startsWith("> ") || lines[i + 1] === ">")) {
        i++;
        quoteLines.push(lines[i].startsWith("> ") ? lines[i].slice(2) : "");
      }

      const isThinking = quoteLines.length > 0 && quoteLines[0].toLowerCase().includes("thinking process");
      if (isThinking) {
        const bodyLines = quoteLines.slice(1);
        elements.push(
          <details key={`${keyPrefix}_think_${i}`} className="gpt-thinking-box" open={isStreaming}>
            <summary className="gpt-thinking-summary">
              <span className="gpt-thinking-icon">💭</span>
              <span className="gpt-thinking-title">Thought Process</span>
              {isStreaming && <span className="gpt-thinking-live-badge">thinking...</span>}
            </summary>
            <div className="gpt-thinking-content">
              {bodyLines.map((ql, qIdx) => (
                <div key={qIdx} className="gpt-thinking-line">
                  {ql ? formatInline(ql) : <br />}
                </div>
              ))}
            </div>
          </details>
        );
      } else {
        elements.push(
          <blockquote key={`${keyPrefix}_quote_${i}`} className="gpt-blockquote">
            {quoteLines.map((ql, qIdx) => (
              <div key={qIdx}>{ql ? formatInline(ql) : <br />}</div>
            ))}
          </blockquote>
        );
      }
      continue;
    }

    // Bullet list
    const bulletMatch = line.match(/^(\s*)[-*+]\s+(.*)$/);
    if (bulletMatch) {
      if (!inList || isNumberedList) {
        flushList();
        inList = true;
        isNumberedList = false;
      }
      listItems.push(
        <li key={`${keyPrefix}_li_${i}`}>{formatInline(bulletMatch[2])}</li>
      );
      continue;
    }

    // Numbered list
    const numberMatch = line.match(/^(\s*)\d+\.\s+(.*)$/);
    if (numberMatch) {
      if (!inList || !isNumberedList) {
        flushList();
        inList = true;
        isNumberedList = true;
      }
      listItems.push(
        <li key={`${keyPrefix}_nli_${i}`}>{formatInline(numberMatch[2])}</li>
      );
      continue;
    }

    // Blank line
    if (!line.trim()) {
      flushList();
      continue;
    }

    // Regular paragraph
    flushList();
    elements.push(
      <p key={`${keyPrefix}_p_${i}`} className="gpt-p">
        {formatInline(line)}
      </p>
    );
  }

  flushList();
  return <React.Fragment key={keyPrefix}>{elements}</React.Fragment>;
}

// Inline formatting: bold (**), italics (*), inline code (`)
function formatInline(str: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  // Tokenize by `code` or **bold** or *italic*
  const tokenRegex = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let lastIdx = 0;
  let match;

  while ((match = tokenRegex.exec(str)) !== null) {
    if (match.index > lastIdx) {
      parts.push(str.slice(lastIdx, match.index));
    }

    const token = match[0];
    if (token.startsWith("`") && token.endsWith("`")) {
      parts.push(
        <code key={`code_${match.index}`} className="gpt-inline-code">
          {token.slice(1, -1)}
        </code>
      );
    } else if (token.startsWith("**") && token.endsWith("**")) {
      parts.push(
        <strong key={`b_${match.index}`} className="gpt-strong">
          {token.slice(2, -2)}
        </strong>
      );
    } else if (token.startsWith("*") && token.endsWith("*")) {
      parts.push(
        <em key={`i_${match.index}`} className="gpt-em">
          {token.slice(1, -1)}
        </em>
      );
    }

    lastIdx = match.index + token.length;
  }

  if (lastIdx < str.length) {
    parts.push(str.slice(lastIdx));
  }

  return parts;
}
