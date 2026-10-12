"use client";

import React, { useState, useEffect, useRef } from "react";
import { Message, ChatSession, KBStats, ModelOption } from "@/types";
import { Sidebar } from "@/components/Sidebar";
import { Header } from "@/components/Header";
import { EmptyHero } from "@/components/EmptyHero";
import { MessageItem } from "@/components/MessageItem";
import { ChatInput } from "@/components/ChatInput";
import { UploadModal } from "@/components/UploadModal";

const MODELS: ModelOption[] = [
  { id: "llama3.2:3b", name: "Assist Me 3B (llama3.2:3b)", desc: "Flagship reasoning • Fits 4GB VRAM GPU" },
  { id: "llama3.2:1b", name: "Assist Me 1B (llama3.2:1b)", desc: "Ultra-fast lightweight model" },
  { id: "direct_synthesis", name: "Direct Grounded Mode", desc: "No Ollama required • Instant local synthesis" },
  { id: "gemma4:cloud", name: "Cloud Engine (gemma4:cloud)", desc: "Ollama cloud datacenter tier" },
];

export default function Home() {
  const [sessions, setSessions] = useState<ChatSession[]>([
    { id: "default_chat", title: "New chat", messages: [], createdAt: Date.now() },
  ]);
  const [activeSessionId, setActiveSessionId] = useState<string>("default_chat");

  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeModel, setActiveModel] = useState("llama3.2:3b");
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);

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
              } else if (chunk.type === "done") {
                setIsStreaming(false);
              }
            } catch {
              // Parse error
            }
          }
        }
      }
    } catch (err: unknown) {
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
    </div>
  );
}
