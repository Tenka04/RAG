"use client";

import React, { useEffect, useId, useState } from "react";
import { Maximize2, ZoomIn, ZoomOut, RotateCcw, X, Layers } from "lucide-react";

interface MermaidViewerProps {
  chart: string;
}

export const MermaidViewer: React.FC<MermaidViewerProps> = ({ chart }) => {
  const [svg, setSvg] = useState<string>("");
  const [hasError, setHasError] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [zoom, setZoom] = useState<number>(1);

  const rawId = useId();
  const diagramId = `mermaid_${rawId.replace(/[^a-zA-Z0-9_-]/g, "_")}`;

  useEffect(() => {
    let isMounted = true;

    async function renderChart() {
      if (!chart.trim()) return;

      try {
        setIsLoading(true);
        const mermaid = (await import("mermaid")).default;

        mermaid.initialize({
          startOnLoad: false,
          theme: "dark",
          securityLevel: "loose",
          fontFamily: "inherit",
          flowchart: {
            useMaxWidth: false,
            htmlLabels: true,
            curve: "basis",
            padding: 10,
            nodeSpacing: 20,
            rankSpacing: 25,
          },
          themeVariables: {
            fontSize: "12px",
            darkMode: true,
            background: "transparent",
            primaryColor: "#242424",
            primaryTextColor: "#ececec",
            primaryBorderColor: "#444444",
            lineColor: "#777777",
            secondaryColor: "#1e1e1e",
            tertiaryColor: "#2a2a2a",
          },
        });

        const { svg: renderedSvg } = await mermaid.render(diagramId, chart.trim());
        if (isMounted) {
          setSvg(renderedSvg);
          setHasError(false);
          setIsLoading(false);
        }
      } catch (err) {
        if (isMounted) {
          setHasError(true);
          setIsLoading(false);
        }
      }
    }

    renderChart();

    return () => {
      isMounted = false;
    };
  }, [chart, diagramId]);

  // Handle ESC key to dismiss modal
  useEffect(() => {
    if (!isModalOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setIsModalOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isModalOpen]);

  if (hasError) {
    return (
      <div className="gpt-diagram-card gpt-diagram-fallback">
        <div className="gpt-diagram-header">
          <span className="gpt-diagram-badge">Diagram (rendering...)</span>
        </div>
        <pre className="gpt-code-pre">
          <code>{chart}</code>
        </pre>
      </div>
    );
  }

  return (
    <>
      <div className="gpt-diagram-card">
        <div className="gpt-diagram-header">
          <span className="gpt-diagram-badge">
            <Layers size={13} />
            Architecture Diagram
          </span>
          {svg && (
            <button
              type="button"
              className="gpt-diagram-expand-btn"
              onClick={() => {
                setZoom(1);
                setIsModalOpen(true);
              }}
              title="Click to view full size overlay"
            >
              <Maximize2 size={12} />
              <span>Full View</span>
            </button>
          )}
        </div>

        <div
          className={`gpt-diagram-body ${svg ? "gpt-diagram-body-clickable" : ""}`}
          onClick={() => {
            if (svg) {
              setZoom(1);
              setIsModalOpen(true);
            }
          }}
          title={svg ? "Click to open fullscreen view" : undefined}
        >
          {isLoading && !svg ? (
            <div className="gpt-diagram-loading">
              <span className="gpt-pulse-dot" />
              <span className="gpt-pulse-dot" />
              <span className="gpt-pulse-dot" />
            </div>
          ) : (
            <div
              className="gpt-diagram-svg-container"
              dangerouslySetInnerHTML={{ __html: svg }}
            />
          )}
        </div>
      </div>

      {/* Fullscreen Overlay Lightbox Modal */}
      {isModalOpen && (
        <div
          className="gpt-diagram-modal-backdrop"
          onClick={() => setIsModalOpen(false)}
        >
          <div
            className="gpt-diagram-modal-content"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="gpt-diagram-modal-header">
              <div className="gpt-diagram-modal-title">
                <Layers size={15} color="#10a37f" />
                <span>Architecture Diagram (Full View)</span>
              </div>
              <div className="gpt-diagram-modal-actions">
                <button
                  type="button"
                  className="gpt-diagram-modal-btn"
                  onClick={() => setZoom((prev) => Math.min(prev + 0.25, 2.5))}
                  title="Zoom In"
                >
                  <ZoomIn size={15} />
                </button>
                <button
                  type="button"
                  className="gpt-diagram-modal-btn"
                  onClick={() => setZoom((prev) => Math.max(prev - 0.25, 0.5))}
                  title="Zoom Out"
                >
                  <ZoomOut size={15} />
                </button>
                <button
                  type="button"
                  className="gpt-diagram-modal-btn"
                  onClick={() => setZoom(1)}
                  title="Reset Zoom"
                >
                  <RotateCcw size={14} />
                </button>
                <button
                  type="button"
                  className="gpt-diagram-modal-btn gpt-diagram-modal-close"
                  onClick={() => setIsModalOpen(false)}
                  title="Close (Esc)"
                >
                  <X size={16} />
                </button>
              </div>
            </div>

            <div className="gpt-diagram-modal-body">
              <div
                className="gpt-diagram-modal-svg-wrap"
                style={{
                  transform: `scale(${zoom})`,
                  transformOrigin: "center center",
                  transition: "transform 0.2s cubic-bezier(0.16, 1, 0.3, 1)",
                }}
                dangerouslySetInnerHTML={{ __html: svg }}
              />
            </div>

            <div className="gpt-diagram-modal-footer">
              <span>Zoom: {Math.round(zoom * 100)}% • Press Esc or click outside to dismiss</span>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
