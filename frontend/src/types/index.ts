export interface SourceItem {
  content: string;
  metadata?: {
    source?: string;
    chunk_id?: number | string;
    [key: string]: any;
  };
  score?: number | string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceItem[];
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
}

export interface KBStats {
  total_chunks: number;
  unique_documents: string[];
  ollama_online: boolean;
  default_model?: string;
}

export interface ModelOption {
  id: string;
  name: string;
  desc: string;
}
