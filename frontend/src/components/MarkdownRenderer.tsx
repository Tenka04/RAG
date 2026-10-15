"use client";

import React, { useState } from "react";

interface MarkdownRendererProps {
  content: string;
  isStreaming?: boolean;
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

  // Split by code fences: ```lang\ncode\n```
  const parts: React.ReactNode[] = [];
  const codeBlockRegex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g;
  let lastIndex = 0;
  let blockIdx = 0;
  let match;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    // Text before code block
    if (match.index > lastIndex) {
      const textChunk = content.slice(lastIndex, match.index);
      parts.push(renderFormattedText(textChunk, `text_${lastIndex}`, isStreaming));
    }

    const language = match[1] || "code";
    const codeSnippet = match[2].trimEnd();
    const currentBlockIdx = blockIdx++;

    parts.push(
      <div key={`code_${match.index}`} className="gpt-code-block">
        <div className="gpt-code-header">
          <span className="gpt-code-lang">{language}</span>
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

    lastIndex = match.index + match[0].length;
  }

  // Trailing text
  if (lastIndex < content.length) {
    const trailing = content.slice(lastIndex);
    parts.push(renderFormattedText(trailing, `text_${lastIndex}`, isStreaming));
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
