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

        {/* Model Picker */}
        <div style={{ position: "relative" }}>
          <div
            className="model-picker-btn"
            onClick={() => setModelDropdownOpen(!modelDropdownOpen)}
          >
            <span>Assist Me</span>
            <span className="model-picker-tag mono-text">[{selectedModelObj.id}]</span>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>

          {modelDropdownOpen && (
            <div className="model-dropdown-menu">
              {models.map((m) => (
                <div
                  key={m.id}
                  className={`model-menu-item ${m.id === activeModel ? "selected" : ""}`}
                  onClick={() => {
                    onSelectModel(m.id);
                    setModelDropdownOpen(false);
                  }}
                >
                  <div>
                    <div className="model-item-title mono-text">{m.name}</div>
                    <div className="model-item-desc">{m.desc}</div>
                  </div>
                  {m.id === activeModel && (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="header-right">
        <button
          className="sidebar-icon-btn"
          onClick={onUploadClick}
          title="Add documents"
          type="button"
        >
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </button>
      </div>
    </header>
  );
};
