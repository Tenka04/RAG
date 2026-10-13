"use client";

import React from "react";
import { ModelOption } from "@/types";

interface HeaderProps {
  sidebarOpen: boolean;
  onOpenSidebar: () => void;
  models: ModelOption[];
  activeModel: string;
  onSelectModel: (id: string) => void;
  modelDropdownOpen: boolean;
  setModelDropdownOpen: (open: boolean) => void;
  onUploadClick: () => void;
  isDevMode: boolean;
  onOpenDevConsole: () => void;
  onOpenDevAuth: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  sidebarOpen,
  onOpenSidebar,
  models,
  activeModel,
  onSelectModel,
  modelDropdownOpen,
  setModelDropdownOpen,
  onUploadClick,
  isDevMode,
  onOpenDevConsole,
  onOpenDevAuth,
}) => {
  const selectedModelObj = models.find((m) => m.id === activeModel) || models[0];

  return (
    <header className="gpt-header">
      <div className="header-left">
        {!sidebarOpen && (
          <button
            className="sidebar-icon-btn"
            onClick={onOpenSidebar}
            title="Open sidebar"
            aria-label="Open sidebar"
            type="button"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <line x1="9" y1="3" x2="9" y2="21" />
            </svg>
          </button>
        )}

        {/* ChatGPT Model Selector */}
        <div className="model-picker-container">
          <button
            className="model-picker-btn"
            onClick={() => setModelDropdownOpen(!modelDropdownOpen)}
            type="button"
          >
            <span className="model-title-text">Assist Me</span>
            <span className="model-badge-sub">{selectedModelObj.id.replace("deepseek/", "").replace(":cloud", "").replace("llama", "")}</span>
            {selectedModelObj.is_cloud && (
              <span className="model-cloud-badge" style={{ fontSize: "10px", padding: "1px 6px", borderRadius: "10px", background: "rgba(16, 163, 127, 0.15)", color: "#10a37f", fontWeight: 600, border: "1px solid rgba(16, 163, 127, 0.3)" }}>
                ☁️ Cloud
              </span>
            )}
            {isDevMode && <span className="dev-header-badge">DEV</span>}
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>

          {modelDropdownOpen && (
            <>
              <div className="dropdown-backdrop" onClick={() => setModelDropdownOpen(false)} />
              <div className="model-dropdown-menu">
                <div className="model-menu-heading">Model Selection</div>
                {models.map((m) => (
                  <button
                    key={m.id}
                    className={`model-menu-item ${m.id === activeModel ? "selected" : ""}`}
                    onClick={() => {
                      onSelectModel(m.id);
                      setModelDropdownOpen(false);
                    }}
                    type="button"
                  >
                    <div className="model-item-text">
                      <div className="model-item-title" style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                        {m.name}
                        {m.is_cloud && (
                          <span style={{ fontSize: "9px", padding: "1px 5px", borderRadius: "4px", background: "rgba(16, 163, 127, 0.2)", color: "#10a37f", fontWeight: 700 }}>
                            CLOUD
                          </span>
                        )}
                      </div>
                      <div className="model-item-desc">{m.desc}</div>
                    </div>
                    {m.id === activeModel && (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                    )}
                  </button>
                ))}

                <div className="model-dropdown-divider" />
                <button
                  className="model-menu-item dev-opt-item"
                  onClick={() => {
                    setModelDropdownOpen(false);
                    if (isDevMode) {
                      onOpenDevConsole();
                    } else {
                      onOpenDevAuth();
                    }
                  }}
                  type="button"
                >
                  <div className="model-item-text">
                    <div className="model-item-title">
                      🛠️ {isDevMode ? "Open Developer Console" : "Enable Developer Mode"}
                    </div>
                    <div className="model-item-desc">
                      {isDevMode ? "Inspect bugs, embeddings, and telemetry" : "Protected access for developers only"}
                    </div>
                  </div>
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="header-right">
        {/* Developer Console Button in Header */}
        {isDevMode ? (
          <button
            className="dev-header-btn"
            onClick={onOpenDevConsole}
            title="Open Developer Console (Ctrl+Shift+D)"
            type="button"
          >
            <span>🛠️ Dev Console</span>
          </button>
        ) : (
          <button
            className="dev-header-btn-subtle"
            onClick={onOpenDevAuth}
            title="Developer Mode"
            type="button"
          >
            <span>🛠️ Dev Mode</span>
          </button>
        )}

        <button
          className="header-action-btn"
          onClick={onUploadClick}
          title="Upload knowledge files"
          type="button"
        >
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <span>Add Files</span>
        </button>
      </div>
    </header>
  );
};
