"use client";

import React from "react";

export interface ComparisonItem {
  title: string;
  badge?: string;
  percentage: number;
  status?: "warning" | "success" | "danger" | "info" | string;
  caption?: string;
}

interface ComparisonWidgetProps {
  items: ComparisonItem[];
}

export const ComparisonWidget: React.FC<ComparisonWidgetProps> = ({ items }) => {
  if (!items || !Array.isArray(items) || items.length === 0) {
    return null;
  }

  const getBarColor = (status?: string) => {
    switch (status) {
      case "success":
        return "linear-gradient(90deg, #10b981 0%, #059669 100%)";
      case "warning":
        return "linear-gradient(90deg, #f97316 0%, #ea580c 100%)";
      case "danger":
        return "linear-gradient(90deg, #ef4444 0%, #dc2626 100%)";
      case "info":
      default:
        return "linear-gradient(90deg, #3b82f6 0%, #2563eb 100%)";
    }
  };

  const getBadgeStyle = (status?: string) => {
    switch (status) {
      case "success":
        return { color: "#10b981", background: "rgba(16, 185, 129, 0.12)" };
      case "warning":
        return { color: "#fb923c", background: "rgba(251, 146, 60, 0.12)" };
      case "danger":
        return { color: "#f87171", background: "rgba(248, 113, 113, 0.12)" };
      default:
        return { color: "#9ca3af", background: "rgba(156, 163, 175, 0.12)" };
    }
  };

  const rawPercentages = items.map((i) => (typeof i.percentage === "number" ? i.percentage : 50));
  const maxVal = Math.max(...rawPercentages, 1);
  const isScaleOver100 = maxVal > 100;

  return (
    <div className="gpt-comparison-card">
      {items.map((item, index) => {
        const computedPercent = isScaleOver100
          ? ((item.percentage ?? 0) / maxVal) * 100
          : (item.percentage ?? 50);
        const clampedPercent = Math.min(Math.max(computedPercent, 4), 100);
        return (
          <div key={index} className="gpt-comparison-item">
            <div className="gpt-comparison-row">
              <span className="gpt-comparison-title">{item.title}</span>
              {item.badge && (
                <span
                  className="gpt-comparison-badge"
                  style={getBadgeStyle(item.status)}
                >
                  {item.badge}
                </span>
              )}
            </div>

            <div className="gpt-comparison-track">
              <div
                className="gpt-comparison-fill"
                style={{
                  width: `${clampedPercent}%`,
                  background: getBarColor(item.status),
                }}
              />
            </div>

            {item.caption && (
              <div className="gpt-comparison-caption">{item.caption}</div>
            )}
          </div>
        );
      })}
    </div>
  );
};
