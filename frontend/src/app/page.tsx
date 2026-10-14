"use client";

import React, { useState, useEffect, useRef } from "react";
import { Message, ChatSession, KBStats, ModelOption } from "@/types";
import { Sidebar } from "@/components/Sidebar";
import { Header } from "@/components/Header";
import { EmptyHero } from "@/components/EmptyHero";
import { MessageItem } from "@/components/MessageItem";
import { ChatInput } from "@/components/ChatInput";
import { UploadModal } from "@/components/UploadModal";
import { DevModeModal } from "@/components/DevModeModal";
import { DevConsoleDrawer } from "@/components/DevConsoleDrawer";

const MODELS: ModelOption[] = [
  { id: "deepseek/deepseek-chat", name: "DeepSeek V3 (deepseek-chat)", desc: "OpenRouter • SOTA Reasoning & Speed", is_cloud: true },
  { id: "deepseek/deepseek-r1", name: "DeepSeek R1 (deepseek-r1)", desc: "OpenRouter • Deep Chain-of-Thought", is_cloud: true },
  { id: "gpt-oss:20b", name: "Assist Me Cloud 20B (gpt-oss:20b)", desc: "Free Ollama Cloud • Blazing fast reasoning", is_cloud: true },
  { id: "gemma4:31b", name: "Gemma 4 31B (gemma4:31b)", desc: "Free Ollama Cloud • Deep reasoning", is_cloud: true },
  { id: "gpt-oss:120b", name: "GPT-OSS 120B (gpt-oss:120b)", desc: "Free Ollama Cloud • Flagship intelligence", is_cloud: true },
  { id: "nemotron-3-nano:30b", name: "Nemotron 30B (nemotron-3-nano:30b)", desc: "Free Ollama Cloud • Fast datacenter", is_cloud: true },
  { id: "llama3.2:3b", name: "Local Ollama 3B (llama3.2:3b)", desc: "On-device GPU/CPU execution", is_cloud: false },
  { id: "direct_synthesis", name: "Direct Grounded Mode", desc: "No LLM required • Instant local synthesis", is_cloud: false },
];

export default function Home() {
  const [sessions, setSessions] = useState<ChatSession[]>([
    { id: "default_chat", title: "New chat", messages: [], createdAt: Date.now() },
  ]);
  const [activeSessionId, setActiveSessionId] = useState<string>("default_chat");

  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeModel, setActiveModel] = useState("deepseek/deepseek-chat");
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);

  // Developer Mode States
  const [isDevMode, setIsDevMode] = useState(false);
  const [devAuthModalOpen, setDevAuthModalOpen] = useState(false);
  const [devConsoleOpen, setDevConsoleOpen] = useState(false);

  // System & KB status
  const [stats, setStats] = useState<KBStats>({
    total_chunks: 0,
    unique_documents: [],
    ollama_online: false,
  });
  const [isIngestingSamples, setIsIngestingSamples] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const currentSession = sessions.find((s) => s.id === activeSessionId) || sessions[0];
  const messages = currentSession?.messages || [];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isStreaming]);

  // Restore Developer Mode from localStorage
  useEffect(() => {
    if (typeof window !== "undefined") {
      const savedDevMode = localStorage.getItem("assistme_dev_mode");
      if (savedDevMode === "true") {
        setIsDevMode(true);
      }
    }
  }, []);

  // Global Keyboard Shortcuts (Ctrl+Shift+D or Cmd+Shift+D to toggle Dev Console)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === "D" || e.key === "d")) {
        e.preventDefault();
        if (isDevMode) {
          setDevConsoleOpen((prev) => !prev);
        } else {
          setDevAuthModalOpen(true);
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isDevMode]);

  // Fetch KB stats
  const fetchStats = async () => {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch {
      // Backend offline or starting
    }
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 6000);
    return () => clearInterval(interval);
  }, []);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3500);
  };

  // Developer Mode Handlers
  const handleDevUnlockSuccess = () => {
    setIsDevMode(true);
    if (typeof window !== "undefined") {
      localStorage.setItem("assistme_dev_mode", "true");
    }
    showToast("🛠️ Developer Mode activated! Dev Console unlocked.");
  };

  const handleLockDevMode = () => {
    setIsDevMode(false);
    setDevConsoleOpen(false);
    if (typeof window !== "undefined") {
      localStorage.removeItem("assistme_dev_mode");
    }
    showToast("🔒 Developer Mode locked.");
  };

  // Chat session management
  const handleNewChat = () => {
    const newId = `chat_${Date.now()}`;
    const newSession: ChatSession = {
      id: newId,
      title: "New chat",
      messages: [],
      createdAt: Date.now(),
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newId);
    setInput("");
  };

  const handleDeleteSession = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (sessions.length === 1) {
      setSessions([{ id: `chat_${Date.now()}`, title: "New chat", messages: [], createdAt: Date.now() }]);
      return;
    }
    const updated = sessions.filter((s) => s.id !== id);
    setSessions(updated);
    if (activeSessionId === id) {
      setActiveSessionId(updated[0].id);
    }
  };

  // Sample Documents Ingestion
  const handleIngestSamples = async () => {
    setIsIngestingSamples(true);
    try {
      const res = await fetch("/api/ingest-sample", { method: "POST" });
      const data = await res.json();
      if (res.ok) {
        showToast(`Indexed ${data.total_chunks_added} chunks from sample documents!`);
        await fetchStats();
      } else {
        showToast(data.detail || "Error ingesting samples");
      }
    } catch {
      showToast("Unable to reach backend server.");
    } finally {
      setIsIngestingSamples(false);
    }
  };

  // Clear Database
  const handleClearKB = async () => {
    if (!confirm("Clear all indexed documents from ChromaDB?")) return;
    try {
      const res = await fetch("/api/clear", { method: "POST" });
      if (res.ok) {
        showToast("Knowledge base cleared!");
        await fetchStats();
      }
    } catch {
      showToast("Error clearing index.");
    }
  };

  // Upload Documents
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        showToast(`Indexed ${data.total_chunks_added} chunks from ${files.length} file(s)!`);
        await fetchStats();
        setUploadModalOpen(false);
      } else {
        showToast(data.detail || "Upload error");
      }
    } catch {
      showToast("Error uploading documents to server.");
    } finally {
      setIsUploading(false);
    }
  };

  // Stop streaming
  const handleStopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  };

  // Send message and stream response
  const handleSendMessage = async (textToSend?: string) => {
    const query = (textToSend || input).trim();
    if (!query || isStreaming) return;

    setInput("");

    const userMessage: Message = {
      id: `usr_${Date.now()}`,
      role: "user",
      content: query,
    };

    const assistantId = `ast_${Date.now()}`;
    const assistantMessage: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      sources: [],
    };

    setSessions((prev) =>
      prev.map((s) => {
        if (s.id === activeSessionId) {
          const isFirst = s.messages.length === 0;
          return {
            ...s,
            title: isFirst ? query.slice(0, 32) + (query.length > 32 ? "..." : "") : s.title,
            messages: [...s.messages, userMessage, assistantMessage],
          };
        }
        return s;
      })
    );

    setIsStreaming(true);
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    const t0 = performance.now();
    console.group(`⏱️ [Assist Me RAG] Latency Trace: "${query.slice(0, 45)}${query.length > 45 ? "..." : ""}"`);
    console.log(`[T0 +0.0ms] 🚀 Dispatching query to POST /api/chat`, {
      query,
      model: activeModel,
      top_k: 3,
    });

    let firstTokenTime: number | null = null;
    let tokenCount = 0;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: abortController.signal,
        body: JSON.stringify({
          query,
          model: activeModel,
          top_k: 3,
        }),
      });

      const ttfb = Math.round(performance.now() - t0);
      console.log(`[TTFB +${ttfb}ms] ⚡ HTTP response headers received (Status: ${response.status})`);

      if (!response.ok || !response.body) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data: ")) {
            const rawJson = trimmed.slice(6);
            try {
              const chunk = JSON.parse(rawJson);
              if (chunk.type === "sources") {
                const tSources = Math.round(performance.now() - t0);
                console.log(`[T+${tSources}ms] 📄 Sources received: ${chunk.sources?.length || 0} document chunk(s)`);
                setSessions((prev) =>
                  prev.map((s) =>
                    s.id === activeSessionId
                      ? {
                          ...s,
                          messages: s.messages.map((m) =>
                            m.id === assistantId ? { ...m, sources: chunk.sources } : m
                          ),
                        }
                      : s
                  )
                );
              } else if (chunk.type === "token") {
                if (firstTokenTime === null) {
                  firstTokenTime = Math.round(performance.now() - t0);
                  console.log(`[TTFT +${firstTokenTime}ms] 💬 Time-To-First-Token! First word rendered on screen.`);
                }
                tokenCount++;
                setSessions((prev) =>
                  prev.map((s) =>
                    s.id === activeSessionId
                      ? {
                          ...s,
                          messages: s.messages.map((m) =>
                            m.id === assistantId ? { ...m, content: m.content + chunk.token } : m
                          ),
                        }
                      : s
                  )
                );
              } else if (chunk.type === "dev_telemetry") {
                // Developer Mode Telemetry chunk
                const tTelemetry = Math.round(performance.now() - t0);
                console.log(`[T+${tTelemetry}ms] 📊 Server Developer Telemetry received:`);
                console.table({
                  "Client Total Roundtrip": `${Math.round(performance.now() - t0)} ms`,
                  "Client TTFT (Time to 1st Token)": `${firstTokenTime ?? 0} ms`,
                  "Server Total Time": `${chunk.telemetry?.total_time_ms ?? 0} ms`,
                  "Server Embedding Latency": `${chunk.telemetry?.embedding_time_ms ?? 0} ms`,
                  "Server Retrieval Latency": `${chunk.telemetry?.retrieval_time_ms ?? 0} ms`,
                  "Server LLM Latency": `${chunk.telemetry?.llm_time_ms ?? 0} ms`,
                  "Server LLM TTFT": `${chunk.telemetry?.ttft_ms ?? 0} ms`,
                  "Model Used": chunk.telemetry?.model_used ?? activeModel,
                  "Used Fallback": chunk.telemetry?.used_fallback ? "Yes ⚠️" : "No",
                });
                if (chunk.telemetry?.bottlenecks && chunk.telemetry.bottlenecks.length > 0) {
                  console.warn("🚨 Latency Bottleneck Root Cause:", chunk.telemetry.bottlenecks);
                }
                setSessions((prev) =>
                  prev.map((s) =>
                    s.id === activeSessionId
                      ? {
                          ...s,
                          messages: s.messages.map((m) =>
                            m.id === assistantId ? { ...m, dev_telemetry: chunk.telemetry } : m
                          ),
                        }
                      : s
                  )
                );
              } else if (chunk.type === "done") {
                const totalDuration = Math.round(performance.now() - t0);
                const streamDuration = firstTokenTime ? (totalDuration - firstTokenTime) : totalDuration;
                const tokensPerSec = streamDuration > 0 ? ((tokenCount / (streamDuration / 1000)).toFixed(1)) : "0";
                console.log(`[DONE +${totalDuration}ms] 🏁 Stream complete! ${tokenCount} tokens rendered in ${streamDuration}ms (${tokensPerSec} tokens/sec).`);
                if (totalDuration > 3000) {
                  console.warn(`⚠️ High Latency Detected (${totalDuration}ms / ${(totalDuration / 1000).toFixed(2)}s). Check the Server Telemetry table above for the slowest step.`);
                }
                console.groupEnd();
                setIsStreaming(false);
              }
            } catch {
              // Parse error
            }
          }
        }
      }
    } catch (err: unknown) {
      const errTime = Math.round(performance.now() - t0);
      console.error(`[ERROR +${errTime}ms] Generation failed:`, err);
      console.groupEnd();
      if (err instanceof Error && err.name === "AbortError") {
        // User stopped generation
      } else {
        setSessions((prev) =>
          prev.map((s) =>
            s.id === activeSessionId
              ? {
                  ...s,
                  messages: s.messages.map((m) =>
                    m.id === assistantId
                      ? {
                          ...m,
                          content:
                            m.content ||
                            "⚠️ Could not reach backend server. Make sure `python server.py` is running on port 8000.",
                        }
                      : m
                  ),
                }
              : s
          )
        );
      }
    } finally {
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  };

  // Regenerate last assistant response
  const handleRegenerate = () => {
    if (messages.length === 0 || isStreaming) return;
    const lastUserMessage = [...messages].reverse().find((m) => m.role === "user");
    if (lastUserMessage) {
      handleSendMessage(lastUserMessage.content);
    }
  };

  return (
    <div className="gpt-app">
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={handleNewChat}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={setActiveSessionId}
        onDeleteSession={handleDeleteSession}
        stats={stats}
        onIngestSamples={handleIngestSamples}
        isIngestingSamples={isIngestingSamples}
        onOpenUpload={() => setUploadModalOpen(true)}
        onClearKB={handleClearKB}
        isDevMode={isDevMode}
        onOpenDevConsole={() => setDevConsoleOpen(true)}
        onOpenDevAuth={() => setDevAuthModalOpen(true)}
      />

      <main className="gpt-main">
        <Header
          sidebarOpen={sidebarOpen}
          onOpenSidebar={() => setSidebarOpen(true)}
          models={MODELS}
          activeModel={activeModel}
          onSelectModel={setActiveModel}
          modelDropdownOpen={modelDropdownOpen}
          setModelDropdownOpen={setModelDropdownOpen}
          onUploadClick={() => setUploadModalOpen(true)}
          isDevMode={isDevMode}
          onOpenDevConsole={() => setDevConsoleOpen(true)}
          onOpenDevAuth={() => setDevAuthModalOpen(true)}
        />

        {toastMessage && (
          <div className="gpt-toast-alert">
            {toastMessage}
          </div>
        )}

        <div className="chat-viewport">
          {messages.length === 0 ? (
            <EmptyHero
              input={input}
              setInput={setInput}
              onSend={handleSendMessage}
              isStreaming={isStreaming}
              onAttachClick={() => setUploadModalOpen(true)}
            />
          ) : (
            <div className="chat-scroll-content">
              {messages.map((msg, index) => (
                <MessageItem
                  key={msg.id}
                  message={msg}
                  isStreaming={isStreaming && msg.role === "assistant" && index === messages.length - 1}
                  onRegenerate={msg.role === "assistant" && index === messages.length - 1 ? handleRegenerate : undefined}
                  isDevMode={isDevMode}
                />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {messages.length > 0 && (
          <ChatInput
            input={input}
            setInput={setInput}
            onSend={() => handleSendMessage()}
            onStop={handleStopStreaming}
            isStreaming={isStreaming}
            onAttachClick={() => setUploadModalOpen(true)}
          />
        )}
      </main>

      <UploadModal
        isOpen={uploadModalOpen}
        onClose={() => setUploadModalOpen(false)}
        onUpload={handleFileUpload}
        isUploading={isUploading}
      />

      <DevModeModal
        isOpen={devAuthModalOpen}
        onClose={() => setDevAuthModalOpen(false)}
        onSuccess={handleDevUnlockSuccess}
      />

      <DevConsoleDrawer
        isOpen={devConsoleOpen}
        onClose={() => setDevConsoleOpen(false)}
        onLockDevMode={handleLockDevMode}
      />
    </div>
  );
}
