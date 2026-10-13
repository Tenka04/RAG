export interface SourceItem {
  content: string;
  metadata?: {
    source?: string;
    chunk_id?: number | string;
    [key: string]: any;
  };
  score?: number | string;
  raw_distance?: number;
}

export interface DevTelemetry {
  query?: string;
  total_time_ms?: number;
  retrieval_time_ms?: number;
  embedding_time_ms?: number;
  llm_time_ms?: number;
  model_used?: string;
  used_fallback?: boolean;
  fallback_reason?: string;
  prompt_preview?: string;
  query_embedding?: {
    model?: string;
    dimensions?: number;
    norm?: number;
    sample_values?: number[];
  };
  chunks_retrieved_count?: number;
  raw_chunks?: Array<{
    id: string | number;
    source: string;
    score?: number | string;
    raw_distance?: number;
    content_preview?: string;
    char_count?: number;
    metadata?: Record<string, any>;
  }>;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceItem[];
  dev_telemetry?: DevTelemetry;
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
  ollama_host?: string;
  is_cloud?: boolean;
  default_model?: string;
  embedding_model?: string;
  collection_name?: string;
}

export interface ModelOption {
  id: string;
  name: string;
  desc: string;
  is_cloud?: boolean;
}

export interface DevLogEntry {
  id: string;
  timestamp: string;
  level: "INFO" | "WARN" | "ERROR" | "DEBUG";
  message: string;
  endpoint?: string;
  duration_ms?: number;
  traceback?: string;
  details?: any;
}

export interface EmbeddingTestResult {
  model: string;
  dimensions: number;
  norm: number;
  sample_vector: number[];
  text_length: number;
  latency_ms: number;
}

export interface SystemDiagnostics {
  status: string;
  python_version: string;
  chromadb_version: string;
  persist_dir: string;
  total_chunks: number;
  collection_name: string;
  ollama_online: boolean;
  ollama_host?: string;
  is_cloud?: boolean;
  api_key_configured?: boolean;
  api_key_masked?: string;
  available_ollama_models: string[];
  default_llm_model: string;
  embedding_model: string;
  server_time: string;
  error_count: number;
}
