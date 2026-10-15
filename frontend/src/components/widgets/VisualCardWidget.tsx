"use client";

import React from "react";
import {
  Server,
  Zap,
  Cpu,
  Layers,
  Database,
  Network,
  Clock,
  Code,
  Shield,
  Activity,
  Box,
  SkipForward,
  FastForward,
  ArrowRight,
  Search,
  Sparkles,
  Workflow,
  FileText,
  CheckCircle,
  LucideIcon,
} from "lucide-react";

export interface VisualCardItem {
  title: string;
  description: string;
  icon?: string;
  imageUrl?: string;
}

interface VisualCardWidgetProps {
  items: VisualCardItem[];
}

const iconMap: Record<string, LucideIcon> = {
  server: Server,
  zap: Zap,
  streaming: Zap,
  cpu: Cpu,
  layers: Layers,
  database: Database,
  network: Network,
  clock: Clock,
  speed: Clock,
  code: Code,
  shield: Shield,
  activity: Activity,
  "skip-forward": SkipForward,
  skipforward: SkipForward,
  fastforward: FastForward,
  routing: SkipForward,
  arrow: ArrowRight,
  search: Search,
  sparkles: Sparkles,
  workflow: Workflow,
  document: FileText,
  check: CheckCircle,
};

export const VisualCardWidget: React.FC<VisualCardWidgetProps> = ({ items }) => {
  if (!items || !Array.isArray(items) || items.length === 0) {
    return null;
  }

  const getIcon = (iconName?: string) => {
    if (!iconName) return Box;
    const key = iconName.toLowerCase().trim();
    return iconMap[key] || Box;
  };

  return (
    <div className="gpt-visual-cards-container">
      {items.map((item, index) => {
        const IconComponent = getIcon(item.icon);

        return (
          <div key={index} className="gpt-visual-card">
            <div className="gpt-visual-card-media">
              {item.imageUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={item.imageUrl}
                  alt={item.title}
                  className="gpt-visual-card-img"
                />
              ) : (
                <div className="gpt-visual-card-icon-box">
                  <IconComponent className="gpt-visual-card-icon" />
                </div>
              )}
            </div>
            <div className="gpt-visual-card-content">
              <h4 className="gpt-visual-card-title">{item.title}</h4>
              <p className="gpt-visual-card-desc">{item.description}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
};
