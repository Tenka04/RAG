"""
RAG Engine: Vector Store (ChromaDB), Embedding, Retrieval, and Generation (Ollama / Ollama Cloud)
Equipped with Developer Telemetry, Embedding Inspection, and Diagnostics.
"""

import os
import sys
import time
import uuid
import math
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional
import chromadb
from chromadb.config import Settings
import ollama
import json
import requests


# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    import dotenv
    dotenv.load_dotenv(dotenv_path=backend_dir / ".env")
except ImportError:
    pass

from document_loader import load_and_chunk_file


# ==============================================================================
# Console Timing Formatting & ANSI Styling Helpers
# ==============================================================================

ANSI_CYAN = "\033[1;36m"
ANSI_BLUE = "\033[1;34m"
ANSI_GREEN = "\033[0;32m"
ANSI_YELLOW = "\033[0;33m"
ANSI_RED = "\033[1;31m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RESET = "\033[0m"


def format_timing_report(
    query: str,
    model: str,
    total_ms: float,
    steps: List[Dict[str, Any]],
    bottlenecks: List[str]
) -> str:
    """Format a detailed ASCII breakdown table with latency status, percentage shares, and root-cause analysis."""
    w = 92
    lines = [
        "",
        f"{ANSI_CYAN}╔" + "═" * (w - 2) + f"╗{ANSI_RESET}",
        f"{ANSI_CYAN}║ ⏱️  RAG PIPELINE EXECUTION & LATENCY BREAKDOWN" + " " * (w - 50) + f"║{ANSI_RESET}",
        f"{ANSI_CYAN}╚" + "═" * (w - 2) + f"╝{ANSI_RESET}",
        f"  {ANSI_BOLD}Query:{ANSI_RESET} \"{query[:55]}{'...' if len(query) > 55 else ''}\"",
        f"  {ANSI_BOLD}Model:{ANSI_RESET} {model} | {ANSI_BOLD}Total Response Time:{ANSI_RESET} {ANSI_BOLD}{total_ms:,.2f} ms{ANSI_RESET} ({total_ms/1000:.2f}s)",
        "─" * w,
        f"{ANSI_DIM}┌───┬──────────────────────────────────────────────┬──────────────┬────────────┬───────────┐{ANSI_RESET}",
        f"{ANSI_DIM}│{ANSI_RESET} # │ Step Description                             │  Duration ms │  Share (%) │ Status    {ANSI_DIM}│{ANSI_RESET}",
        f"{ANSI_DIM}├───┼──────────────────────────────────────────────┼──────────────┼────────────┼───────────┤{ANSI_RESET}",
    ]

    for i, s in enumerate(steps, 1):
        d_ms = s.get("duration_ms", 0.0)
        pct = (d_ms / total_ms * 100) if total_ms > 0 else 0
        if d_ms < 150:
            status = f"{ANSI_GREEN}🟢 Fast   {ANSI_RESET}"
        elif d_ms < 1000:
            status = f"{ANSI_YELLOW}🟡 Moderate{ANSI_RESET}"
        else:
            status = f"{ANSI_RED}🔴 SLOW    {ANSI_RESET}"
        name = s.get("name", "")[:44].ljust(44)
        lines.append(f"{ANSI_DIM}│{ANSI_RESET} {i} │ {name} │ {d_ms:>10.2f} ms │ {pct:>8.1f} % │ {status} {ANSI_DIM}│{ANSI_RESET}")

    lines.append(f"{ANSI_DIM}└───┴──────────────────────────────────────────────┴──────────────┴────────────┴───────────┘{ANSI_RESET}")

    if bottlenecks:
        lines.append(f"\n{ANSI_RED}{ANSI_BOLD}🚨 LATENCY ROOT CAUSE ANALYSIS & DIAGNOSIS:{ANSI_RESET}")
        for b in bottlenecks:
            lines.append(f"  {ANSI_YELLOW}•{ANSI_RESET} {b}")

    lines.append("═" * w + "\n")
    return "\n".join(lines)


def format_ingest_report(
    filename: str,
    chunks_count: int,
    total_ms: float,
    steps: List[Dict[str, Any]]
) -> str:
    """Format document ingestion performance report."""
    w = 80
    lines = [
        "",
        f"{ANSI_BLUE}┌" + "─" * (w - 2) + f"┐{ANSI_RESET}",
        f"{ANSI_BLUE}│ 📄 [INGESTION TIMING] Indexed '{filename}'" + " " * max(1, w - len(filename) - 34) + f"│{ANSI_RESET}",
        f"{ANSI_BLUE}├" + "─" * (w - 2) + f"┤{ANSI_RESET}",
    ]
    for s in steps:
        lines.append(f"{ANSI_BLUE}│{ANSI_RESET}  ├─ {s['name'][:35].ljust(35)}: {s['duration_ms']:>8.2f} ms  {s.get('details', '')}")
    chunks_per_sec = (chunks_count / (total_ms / 1000)) if total_ms > 0 else 0
    lines.append(f"{ANSI_BLUE}│{ANSI_RESET}  └─ Total: {total_ms:,.2f} ms ({chunks_count} chunks @ {chunks_per_sec:.1f} chunks/sec)")
    lines.append(f"{ANSI_BLUE}└" + "─" * (w - 2) + f"┘{ANSI_RESET}\n")
    return "\n".join(lines)


class RAGEngine:
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "rag_knowledge_base",
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        ollama_host: Optional[str] = None,
        ollama_api_key: Optional[str] = None,
    ):
        if persist_dir is None:
            backend_chroma = Path(__file__).resolve().parent / "chroma_db"
            root_chroma = Path(__file__).resolve().parent.parent / "chroma_db"
            if backend_chroma.exists():
                persist_dir = str(backend_chroma)
            elif root_chroma.exists():
                persist_dir = str(root_chroma)
            else:
                persist_dir = str(backend_chroma)

        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Host & API Key configuration (Ollama Cloud or Local)
        self.ollama_host = ollama_host or os.environ.get("OLLAMA_HOST", "https://ollama.com")
        self.ollama_api_key = ollama_api_key or os.environ.get("OLLAMA_API_KEY", "0f64427080054799997426594bb41a76.Ga6nHBfHUGxPBYFMGcyygSEO")

        # Models configuration
        self.llm_model = llm_model or os.environ.get("OLLAMA_DEFAULT_MODEL", "gpt-oss:20b")
        self.embedding_model = embedding_model or os.environ.get("OLLAMA_EMBED_MODEL", "all-minilm")
        self.init_time = time.time()

        # OpenRouter Configuration (Supports OPENROUTER_API_KEY, OPEN_ROUTER_KEY, OPENROUTER_KEY)
        self.openrouter_api_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPEN_ROUTER_KEY")
            or os.environ.get("OPENROUTER_KEY")
        )

        # Initialize Ollama Client with Auth Bearer token if provided
        headers = {}
        if self.ollama_api_key:
            headers["Authorization"] = f"Bearer {self.ollama_api_key}"
        self.client = ollama.Client(host=self.ollama_host, headers=headers if headers else None)

        # Initialize persistent ChromaDB client
        os.makedirs(self.persist_dir, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document embeddings collection"}
        )

        # Default embedding fallback helper & status caching
        self._fallback_embedder = None
        self._ollama_avail_cache = None  # Tuple[bool, float]
        self._ollama_embed_unauthorized = False

    def _check_ollama_available(self) -> bool:
        """Check if Ollama service (cloud or local) is accessible with 15-second TTL cache."""
        now = time.time()
        if self._ollama_avail_cache is not None and (now - self._ollama_avail_cache[1] < 15.0):
            return self._ollama_avail_cache[0]
        try:
            self.client.list()
            self._ollama_avail_cache = (True, now)
            return True
        except Exception:
            self._ollama_avail_cache = (False, now)
            return False

    def _is_openrouter_model(self, model: str) -> bool:
        """Check if model targets OpenRouter (e.g. DeepSeek models)."""
        if not model:
            return False
        m = model.strip().lower()
        if m in ("deepseek", "deepseek-chat", "deepseek-r1", "deepseek/deepseek-chat", "deepseek/deepseek-r1"):
            return True
        if m.startswith("deepseek/") or m.startswith("openrouter/"):
            return True
        if "/" in m and self.openrouter_api_key:
            return True
        return False

    def _resolve_openrouter_model(self, model: str) -> str:
        """Resolve model name alias to canonical OpenRouter model slug."""
        m = model.strip()
        if m.startswith("openrouter/"):
            m = m[len("openrouter/"):]
        if m in ("deepseek", "deepseek-chat"):
            return "deepseek/deepseek-chat"
        if m == "deepseek-r1":
            return "deepseek/deepseek-r1"
        return m

    def _generate_openrouter_stream(self, model: str, prompt: str) -> Generator[str, None, None]:
        """Stream response tokens from OpenRouter API with provider fallback and reasoning support."""
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key is not configured. Add OPEN_ROUTER_KEY in backend/.env")

        target_model = self._resolve_openrouter_model(model)
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Assist Me RAG"
        }
        fallback_models = [target_model]
        if target_model != "deepseek/deepseek-chat":
            fallback_models.append("deepseek/deepseek-chat")
        if target_model != "deepseek/deepseek-r1":
            fallback_models.append("deepseek/deepseek-r1")

        payload = {
            "model": target_model,
            "models": fallback_models,
            "provider": {"allow_fallbacks": True},
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "stream": True
        }

        print(f"    ├─ 🌐 OpenRouter: Connecting to https://openrouter.ai/api/v1/chat/completions (model: '{target_model}')...")
        t_req_start = time.perf_counter()
        response = None
        for attempt in range(3):
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=45
            )
            if response.status_code == 429 and attempt < 2:
                wait_sec = 1.5 * (attempt + 1)
                print(f"    ├─ ⚠️ OpenRouter Rate Limited (HTTP 429). Retrying in {wait_sec}s (attempt {attempt+1}/3)...")
                time.sleep(wait_sec)
                continue
            break

        http_latency_ms = round((time.perf_counter() - t_req_start) * 1000, 2)
        if response is None or response.status_code != 200:
            err_msg = response.text if response is not None else "No response"
            print(f"    ├─ ❌ OpenRouter returned HTTP {response.status_code if response else 'ERR'} in {http_latency_ms} ms: {err_msg[:120]}")
            raise RuntimeError(f"OpenRouter returned HTTP {response.status_code if response else 'ERR'}: {err_msg}")

        print(f"    ├─ ⚡ OpenRouter: HTTP {response.status_code} OK received in {http_latency_ms} ms (TTFB / Connection Latency)")

        in_reasoning = False
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                # Handle DeepSeek R1 reasoning chain-of-thought
                reasoning = delta.get("reasoning")
                if reasoning:
                    if not in_reasoning:
                        in_reasoning = True
                        yield "> **Thinking Process:**\n> "
                    yield reasoning.replace("\n", "\n> ")

                content = delta.get("content")
                if content:
                    if in_reasoning:
                        in_reasoning = False
                        yield "\n\n"
                    yield content
            except Exception:
                continue

    def _generate_openrouter_sync(self, model: str, prompt: str) -> str:
        """Synchronously generate response from OpenRouter API."""
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key is not configured. Add OPEN_ROUTER_KEY in backend/.env")

        target_model = self._resolve_openrouter_model(model)
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Assist Me RAG"
        }
        fallback_models = [target_model]
        if target_model != "deepseek/deepseek-chat":
            fallback_models.append("deepseek/deepseek-chat")
        if target_model != "deepseek/deepseek-r1":
            fallback_models.append("deepseek/deepseek-r1")

        payload = {
            "model": target_model,
            "models": fallback_models,
            "provider": {"allow_fallbacks": True},
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "stream": False
        }

        print(f"    ├─ 🌐 OpenRouter Sync: Connecting to API (model: '{target_model}')...")
        t_req_start = time.perf_counter()
        response = None
        for attempt in range(3):
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=45
            )
            if response.status_code == 429 and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            break

        http_latency_ms = round((time.perf_counter() - t_req_start) * 1000, 2)
        if response is None or response.status_code != 200:
            err_msg = response.text if response is not None else "No response"
            print(f"    ├─ ❌ OpenRouter Sync failed in {http_latency_ms} ms: {err_msg[:120]}")
            raise RuntimeError(f"OpenRouter returned HTTP {response.status_code if response else 'ERR'}: {err_msg}")

        print(f"    ├─ ⚡ OpenRouter Sync: HTTP {response.status_code} OK received in {http_latency_ms} ms")
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return ""

        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        reasoning = msg.get("reasoning")

        if reasoning and not content:
            return f"> **Thinking Process:**\n{reasoning}"
        elif reasoning and content:
            return f"> **Thinking Process:**\n{reasoning}\n\n{content}"
        return content


    def _get_fallback_embedder(self):
        """Lazy load chromadb default embedding function (ONNX all-MiniLM-L6-v2) when needed."""
        if self._fallback_embedder is None:
            try:
                from chromadb.utils import embedding_functions
                self._fallback_embedder = embedding_functions.DefaultEmbeddingFunction()
            except Exception:
                pass
        return self._fallback_embedder

    def get_embedding(
        self,
        text: str,
        verbose: bool = False,
        return_diagnostics: bool = False
    ) -> Any:
        """
        Generate embedding vector for a given text.
        Attempts Ollama embed first; if unauthorized or not supported by endpoint,
        seamlessly uses local Chroma ONNX DefaultEmbeddingFunction (all-MiniLM-L6-v2).
        Records granular sub-step timings and returns diagnostics.
        """
        t0 = time.perf_counter()
        diag = {
            "ollama_checked": False,
            "ollama_available": False,
            "ollama_check_ms": 0.0,
            "ollama_embed_ms": 0.0,
            "ollama_embed_error": None,
            "ollama_legacy_ms": 0.0,
            "ollama_legacy_error": None,
            "fallback_ms": 0.0,
            "fallback_used": False,
            "total_ms": 0.0,
            "dimensions": 0,
            "provider": None,
        }

        # Check Ollama unless known unauthorized
        if not self._ollama_embed_unauthorized:
            t_chk = time.perf_counter()
            is_avail = self._check_ollama_available()
            diag["ollama_check_ms"] = round((time.perf_counter() - t_chk) * 1000, 2)
            diag["ollama_checked"] = True
            diag["ollama_available"] = is_avail

            if verbose:
                print(f"      ├─ [EMBED 1/3] Ollama check ({self.ollama_host}): {diag['ollama_check_ms']} ms ({'Online' if is_avail else 'Offline'})")

            if is_avail:
                t_embed = time.perf_counter()
                try:
                    res = self.client.embed(model=self.embedding_model, input=text)
                    diag["ollama_embed_ms"] = round((time.perf_counter() - t_embed) * 1000, 2)
                    emb = None
                    if isinstance(res, dict) and "embeddings" in res and res["embeddings"]:
                        emb = [float(x) for x in res["embeddings"][0]]
                    elif hasattr(res, "embeddings") and res.embeddings:
                        emb = [float(x) for x in res.embeddings[0]]

                    if emb:
                        diag["dimensions"] = len(emb)
                        diag["provider"] = f"ollama/{self.embedding_model}"
                        diag["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                        if verbose:
                            print(f"      └─ [EMBED 2/3] ✅ Ollama client.embed succeeded in {diag['ollama_embed_ms']} ms ({len(emb)} dims)")
                        return (emb, diag) if return_diagnostics else emb
                except Exception as e:
                    diag["ollama_embed_ms"] = round((time.perf_counter() - t_embed) * 1000, 2)
                    diag["ollama_embed_error"] = str(e)
                    if "401" in str(e) or "unauthorized" in str(e).lower():
                        self._ollama_embed_unauthorized = True
                    if verbose:
                        print(f"      ├─ [EMBED 2/3] ⚠️ Ollama client.embed failed in {diag['ollama_embed_ms']} ms: {str(e)[:90]}")

                # Try legacy embeddings API
                t_legacy = time.perf_counter()
                try:
                    response = self.client.embeddings(model=self.embedding_model, prompt=text)
                    diag["ollama_legacy_ms"] = round((time.perf_counter() - t_legacy) * 1000, 2)
                    emb = response.get("embedding")
                    if emb:
                        vec = [float(x) for x in emb]
                        diag["dimensions"] = len(vec)
                        diag["provider"] = f"ollama/{self.embedding_model}"
                        diag["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                        if verbose:
                            print(f"      └─ [EMBED 2/3] ✅ Ollama legacy embeddings succeeded in {diag['ollama_legacy_ms']} ms ({len(vec)} dims)")
                        return (vec, diag) if return_diagnostics else vec
                except Exception as e:
                    diag["ollama_legacy_ms"] = round((time.perf_counter() - t_legacy) * 1000, 2)
                    diag["ollama_legacy_error"] = str(e)
                    if verbose:
                        print(f"      ├─ [EMBED 2/3] ⚠️ Ollama legacy embeddings failed in {diag['ollama_legacy_ms']} ms: {str(e)[:90]}")
        else:
            if verbose:
                print(f"      ├─ [EMBED 1/3] Ollama remote embed skipped (remote host {self.ollama_host} does not permit embeddings / 401)")

        # Fallback to local Chroma ONNX DefaultEmbeddingFunction (offline, instant)
        t_fallback = time.perf_counter()
        try:
            embedder = self._get_fallback_embedder()
            if embedder:
                results = embedder([text])
                diag["fallback_ms"] = round((time.perf_counter() - t_fallback) * 1000, 2)
                if results and len(results) > 0:
                    vec = [float(x) for x in results[0]]
                    diag["fallback_used"] = True
                    diag["dimensions"] = len(vec)
                    diag["provider"] = "chroma-onnx-all-MiniLM-L6-v2 (offline local)"
                    diag["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                    if verbose:
                        print(f"      ├─ [EMBED 3/3] 🔄 Chroma ONNX local embedder: {diag['fallback_ms']} ms [SUCCESS: {len(vec)} dims]")
                        if diag["total_ms"] > 1000:
                            print(f"      └─ ⚠️ High Latency Alert: Total embedding took {diag['total_ms']} ms! (Network delay to {self.ollama_host})")
                        else:
                            print(f"      └─ Total embedding latency: {diag['total_ms']} ms")
                    return (vec, diag) if return_diagnostics else vec
        except Exception as e:
            diag["fallback_ms"] = round((time.perf_counter() - t_fallback) * 1000, 2)
            if verbose:
                print(f"      └─ [EMBED 3/3] ❌ Chroma fallback embedder failed in {diag['fallback_ms']} ms: {e}")

        diag["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return (None, diag) if return_diagnostics else None

    def test_embedding(self, text: str) -> Dict[str, Any]:
        """
        Developer Mode inspection: computes embedding for arbitrary text,
        measuring latency, dimensions, norm, and sample float values.
        """
        embedding, diag = self.get_embedding(text, verbose=True, return_diagnostics=True)
        latency_ms = diag.get("total_ms", 0.0)

        if embedding:
            norm = math.sqrt(sum(x * x for x in embedding))
            provider = diag.get("provider") or ("chroma-onnx-all-MiniLM-L6-v2 (offline local)" if len(embedding) == 384 else f"ollama/{self.embedding_model}")
            return {
                "success": True,
                "model": provider,
                "dimensions": len(embedding),
                "norm": round(norm, 4),
                "latency_ms": latency_ms,
                "diagnostics": diag,
                "sample_vector": [round(x, 5) for x in embedding[:24]],
                "text_length": len(text)
            }
        else:
            return {
                "success": False,
                "error": f"Could not generate vector embedding. Both Ollama ({self.ollama_host}) and local ONNX failed.",
                "model": self.embedding_model,
                "dimensions": 0,
                "norm": 0.0,
                "latency_ms": latency_ms,
                "diagnostics": diag,
                "sample_vector": [],
                "text_length": len(text)
            }

    def ingest_file(self, file_path: str, chunk_size: int = 600, chunk_overlap: int = 100) -> int:
        """
        Load, chunk, and index a file into ChromaDB with timing instrumentation.
        Returns the number of chunks indexed.
        """
        t_ingest_start = time.perf_counter()
        filename = Path(file_path).name
        print(f"\n📂 [INGEST] Starting ingestion for file: '{filename}'")

        # Step 1: Text extraction
        t_extract = time.perf_counter()
        from document_loader import extract_text_from_file, chunk_text
        text = extract_text_from_file(file_path)
        extract_ms = round((time.perf_counter() - t_extract) * 1000, 2)
        print(f"   ├─ [1/4] Text extracted: {len(text):,} chars in {extract_ms} ms")

        # Step 2: Chunking
        t_chunk = time.perf_counter()
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, metadata={"source": filename})
        chunk_ms = round((time.perf_counter() - t_chunk) * 1000, 2)
        if not chunks:
            print(f"   └─ ⚠️ No chunks generated from file.")
            return 0
        print(f"   ├─ [2/4] Text chunked: {len(chunks)} chunks created in {chunk_ms} ms")

        documents = []
        metadatas = []
        ids = []
        embeddings = []

        # Step 3: Embeddings
        t_embeds = time.perf_counter()
        print(f"   ├─ [3/4] Generating vector embeddings for {len(chunks)} chunk(s)...")
        for chunk in chunks:
            chunk_content = chunk["content"]
            chunk_meta = chunk["metadata"]
            
            clean_meta = {k: str(v) if not isinstance(v, (int, float, bool, str)) else v for k, v in chunk_meta.items()}
            doc_id = f"{filename}#chunk_{clean_meta.get('chunk_id', uuid.uuid4().hex[:8])}"
            
            documents.append(chunk_content)
            metadatas.append(clean_meta)
            ids.append(doc_id)

            emb = self.get_embedding(chunk_content, verbose=False)
            if emb:
                embeddings.append(emb)

        embeds_ms = round((time.perf_counter() - t_embeds) * 1000, 2)
        avg_per_chunk = round(embeds_ms / len(chunks), 2) if chunks else 0
        print(f"   ├─ [3/4] Embeddings complete: {len(embeddings)}/{len(documents)} vectors in {embeds_ms} ms ({avg_per_chunk} ms/chunk)")

        # Step 4: Upsert into ChromaDB
        t_upsert = time.perf_counter()
        if embeddings and len(embeddings) == len(documents):
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
        else:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        upsert_ms = round((time.perf_counter() - t_upsert) * 1000, 2)
        print(f"   ├─ [4/4] ChromaDB upsert: {len(documents)} chunks stored in {upsert_ms} ms")

        total_ingest_ms = round((time.perf_counter() - t_ingest_start) * 1000, 2)
        steps_summary = [
            {"name": "1. Text Extraction", "duration_ms": extract_ms, "details": f"{len(text):,} chars"},
            {"name": "2. Text Chunking", "duration_ms": chunk_ms, "details": f"{len(chunks)} chunks"},
            {"name": "3. Vector Embeddings", "duration_ms": embeds_ms, "details": f"avg {avg_per_chunk} ms/chunk"},
            {"name": "4. ChromaDB Upsert", "duration_ms": upsert_ms, "details": f"{len(documents)} records stored"},
        ]
        report = format_ingest_report(filename, len(documents), total_ingest_ms, steps_summary)
        print(report)

        return len(documents)

    def retrieve(self, query: str, top_k: int = 3, verbose: bool = True) -> Dict[str, Any]:
        """
        Retrieve relevant context chunks with performance timing and raw distances.
        Records granular sub-step latency for ChromaDB count, embedding, search, and filtering.
        """
        t_start = time.perf_counter()
        if verbose:
            print(f"\n [RETRIEVAL] 🔍 Searching ChromaDB knowledge base (top_k={top_k})...")

        # Sub-step 1: KB Count check
        t_count = time.perf_counter()
        doc_count = self.collection.count()
        count_ms = round((time.perf_counter() - t_count) * 1000, 2)

        if doc_count == 0:
            if verbose:
                print(f"    └─ ⚠️ Collection is empty (0 chunks). Skipping vector search ({count_ms} ms).")
            return {
                "chunks": [],
                "retrieval_ms": round((time.perf_counter() - t_start) * 1000, 2),
                "embedding_ms": 0.0,
                "vector_search_ms": 0.0,
                "filter_ms": 0.0,
                "count_check_ms": count_ms,
                "query_embedding": None,
                "embed_diagnostics": {}
            }

        if verbose:
            print(f"    ├─ ChromaDB count check: {count_ms} ms ({doc_count} chunks indexed in DB)")

        # Sub-step 2: Query Embedding
        query_embedding, embed_diag = self.get_embedding(query, verbose=verbose, return_diagnostics=True)
        embedding_ms = embed_diag.get("total_ms", 0.0)

        # Sub-step 3: Vector search
        t_search = time.perf_counter()
        n_res = min(top_k, doc_count)
        if query_embedding:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_res
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_res
            )
        vector_search_ms = round((time.perf_counter() - t_search) * 1000, 2)

        # Sub-step 4: Distance ranking & score calculation
        t_filter = time.perf_counter()
        retrieved_docs = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(docs)
            ids = results.get("ids", [[]])[0] if "ids" in results else []

            for i in range(len(docs)):
                raw_dist = distances[i] if i < len(distances) else 0.0
                score = round(1.0 - raw_dist, 3) if raw_dist <= 1.0 else round(1.0 / (1.0 + raw_dist), 3)
                retrieved_docs.append({
                    "content": docs[i],
                    "metadata": metas[i] if i < len(metas) else {},
                    "score": score,
                    "raw_distance": round(raw_dist, 4),
                    "chunk_id": ids[i] if i < len(ids) else f"chunk_{i}"
                })
        filter_ms = round((time.perf_counter() - t_filter) * 1000, 2)
        total_retrieval_ms = round((time.perf_counter() - t_start) * 1000, 2)

        if verbose:
            print(f"    ├─ ChromaDB vector query: {vector_search_ms} ms (searched {n_res} candidates)")
            print(f"    ├─ Score calculation & metadata mapping: {filter_ms} ms")
            if retrieved_docs:
                chunks_summary = ", ".join([f"'{d.get('metadata', {}).get('source', 'doc')}' (dist: {d.get('raw_distance')})" for d in retrieved_docs[:3]])
                print(f"    └─ ✅ Retrieval complete in {total_retrieval_ms} ms: {len(retrieved_docs)} chunk(s) [{chunks_summary}]")
            else:
                print(f"    └─ ⚠️ Retrieval complete in {total_retrieval_ms} ms (0 matches)")

        return {
            "chunks": retrieved_docs,
            "retrieval_ms": total_retrieval_ms,
            "embedding_ms": embedding_ms,
            "vector_search_ms": vector_search_ms,
            "filter_ms": filter_ms,
            "count_check_ms": count_ms,
            "query_embedding": query_embedding,
            "embed_diagnostics": embed_diag
        }

    @staticmethod
    def _is_casual_message(query: str) -> bool:
        """Check if user query is a simple greeting or casual conversational message."""
        clean = query.strip().lower().rstrip("!.,?").strip()
        casual_phrases = [
            "hi", "hello", "hey", "hola", "greetings", "good morning",
            "good afternoon", "good evening", "howdy", "sup", "what's up",
            "yo", "how are you", "who are you", "what can you do", "thanks",
            "thank you", "bye", "goodbye"
        ]
        if clean in casual_phrases:
            return True
        for p in casual_phrases:
            if clean.startswith(f"{p} ") or clean.endswith(f" {p}"):
                return True
        return False

    def build_prompt(self, query: str, context_chunks: Optional[List[Dict[str, Any]]] = None) -> str:
        """Construct a friendly, flexible grounded prompt or general chat prompt."""
        if context_chunks and len(context_chunks) > 0 and not self._is_casual_message(query):
            context_blocks = []
            for i, chunk in enumerate(context_chunks):
                source = chunk.get("metadata", {}).get("source", f"Document {i+1}")
                page_info = chunk.get("metadata", {}).get("chunk_id", i)
                context_blocks.append(f"--- SOURCE: {source} (Chunk {page_info}) ---\n{chunk['content']}")

            context_text = "\n\n".join(context_blocks)

            system_instruction = (
                "You are Assist Me, a friendly, intelligent, and versatile AI assistant equipped with document knowledge.\n"
                "Instructions:\n"
                "1. GREETINGS & CASUAL CHAT: If the user says 'hi', 'hello', asks how you are, or engages in casual conversation, reply naturally, warmly, and politely without citing documents.\n"
                "2. DOCUMENT QUESTIONS: When the user asks about topics described in the provided CONTEXT SOURCES, provide an accurate, grounded answer and cite the source name(s) (e.g., [document.txt]).\n"
                "3. GENERAL QUESTIONS: When the user asks general questions not covered in the documents (e.g. coding, math, general knowledge, explanations, writing), answer them directly, helpfully, and clearly using your general intelligence.\n"
                "4. Tone: Helpful, clear, concise, and conversational. Do NOT say 'I cannot answer based on the provided documents' for casual conversation or general knowledge."
            )

            full_prompt = (
                f"{system_instruction}\n\n"
                f"=== CONTEXT SOURCES ===\n{context_text}\n========================\n\n"
                f"User Question: {query}\n\n"
                f"Answer:"
            )
            return full_prompt
        else:
            system_instruction = (
                "You are Assist Me, a friendly, intelligent, and highly capable AI assistant.\n"
                "Answer the user's questions clearly, accurately, and politely. Be conversational, helpful, and concise."
            )
            return f"{system_instruction}\n\nUser Question: {query}\n\nAnswer:"

    def _generate_fallback_synthesis(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """Fallback synthesis when Ollama is offline or in Direct Grounded Mode."""
        if sources:
            lines = [
                "### 📄 Grounded Document Findings\n",
                f"Here are the most relevant excerpts retrieved for: **\"{query}\"**\n\n"
            ]
            for i, src in enumerate(sources):
                src_name = src.get("metadata", {}).get("source", f"Source {i+1}")
                score = src.get("score", "N/A")
                lines.append(f"**From `{src_name}` (Relevance: {score}):**\n")
                lines.append(f"> {src['content'].strip()}\n\n")
            
            lines.append("\n---\n*💡 **Tip**: Direct Grounded Mode active. Select `gpt-oss:20b` for full conversational reasoning.*")
            return "".join(lines)
        else:
            return (
                f"Hello! I am **Assist Me**.\n\n"
                f"You asked: *\"{query}\"*\n\n"
                f"No relevant documents were found in the database. In Direct Grounded Mode, responses are synthesized directly from indexed documents. "
                f"Upload documents in the sidebar or switch to **`gpt-oss:20b`** for conversational AI."
            )

    def generate_response_stream(
        self,
        query: str,
        top_k: int = 3,
        model_override: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream the generated answer while yielding retrieved sources and developer telemetry.
        Prints real-time sub-step timing breakdown and root-cause bottleneck analysis to console.
        Yields:
          - {"type": "sources", "sources": [...]}
          - {"type": "token", "token": "..."}
          - {"type": "dev_telemetry", "telemetry": {...}}
          - {"type": "done"}
        """
        t_overall_start = time.perf_counter()
        active_model = model_override or self.llm_model
        steps: List[Dict[str, Any]] = []
        bottlenecks: List[str] = []

        print(f"\n{ANSI_CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗{ANSI_RESET}")
        print(f"{ANSI_CYAN}║ 🚀 [RAG PIPELINE] Starting response generation stream                                ║{ANSI_RESET}")
        print(f"{ANSI_CYAN}╚══════════════════════════════════════════════════════════════════════════════════════╝{ANSI_RESET}")
        print(f"   Query: \"{query}\"")
        print(f"   Model: {active_model} | Top-K: {top_k}")
        print("─" * 88)

        # Step 1: Casual check & Collection count
        t_step1 = time.perf_counter()
        is_casual = self._is_casual_message(query)
        doc_count = self.collection.count()
        step1_ms = round((time.perf_counter() - t_step1) * 1000, 2)
        steps.append({
            "name": "1. Intent & KB Check",
            "duration_ms": step1_ms,
            "details": f"casual={is_casual}, chunks_in_db={doc_count}"
        })
        print(f" [Step 1/5] Intent & Knowledge Base Check: {step1_ms} ms")
        print(f"    └─ Casual conversational query: {is_casual} | Total chunks in database: {doc_count}")

        # Step 2 & 3: Context Retrieval (Embedding + Vector Search)
        if not is_casual and doc_count > 0:
            print(f"\n [Step 2/5] Vector Embedding & ChromaDB Retrieval")
            retrieval_data = self.retrieve(query, top_k=top_k, verbose=True)
            raw_sources = retrieval_data["chunks"]
            # Filter sources to only include chunks within meaningful semantic distance
            sources = [s for s in raw_sources if s.get("raw_distance", 2.0) <= 1.45]
            retrieval_ms = retrieval_data["retrieval_ms"]
            embedding_ms = retrieval_data["embedding_ms"]
            vector_search_ms = retrieval_data.get("vector_search_ms", 0.0)
            query_embedding = retrieval_data["query_embedding"]
            embed_diag = retrieval_data.get("embed_diagnostics", {})

            # Record embedding step
            steps.append({
                "name": "2. Query Vector Embedding",
                "duration_ms": embedding_ms,
                "details": f"provider={embed_diag.get('provider')}, dims={embed_diag.get('dimensions')}"
            })
            if embedding_ms > 1000:
                bottlenecks.append(
                    f"Step 2 (Query Vector Embedding) took {embedding_ms:,.2f} ms: "
                    f"Remote Ollama host ({self.ollama_host}) took multiple seconds or timed out before falling back to local ONNX. "
                    f"Using local Ollama or setting embeddings to Chroma local ONNX eliminates this latency."
                )

            # Record Chroma vector search step
            steps.append({
                "name": "3. ChromaDB Vector Retrieval",
                "duration_ms": vector_search_ms,
                "details": f"matched={len(sources)}/{len(raw_sources)} chunks"
            })
        else:
            sources = []
            retrieval_ms = 0.0
            embedding_ms = 0.0
            vector_search_ms = 0.0
            query_embedding = None
            embed_diag = {}
            steps.append({"name": "2. Query Vector Embedding", "duration_ms": 0.0, "details": "skipped"})
            steps.append({"name": "3. ChromaDB Vector Retrieval", "duration_ms": 0.0, "details": "skipped"})
            print(f"\n [Step 2/5] Context Retrieval: Skipped (casual={is_casual}, doc_count={doc_count})")

        # Yield sources right away for UI display
        t_yield_sources = time.perf_counter()
        yield {"type": "sources", "sources": sources}
        yield_sources_ms = round((time.perf_counter() - t_yield_sources) * 1000, 2)

        # Step 4: Prompt Construction
        t_prompt = time.perf_counter()
        prompt = self.build_prompt(query, sources if sources else None)
        prompt_ms = round((time.perf_counter() - t_prompt) * 1000, 2)
        prompt_len = len(prompt)
        approx_tokens = prompt_len // 4
        steps.append({
            "name": "4. Context & Prompt Assembly",
            "duration_ms": prompt_ms,
            "details": f"chars={prompt_len}, ~{approx_tokens} tokens, sources={len(sources)}"
        })
        print(f"\n [Step 3/5] Prompt Construction: {prompt_ms} ms")
        print(f"    └─ Prompt length: {prompt_len:,} characters (~{approx_tokens} tokens) | Sources cited: {len(sources)}")

        # Step 5: LLM Generation & Token Streaming
        print(f"\n [Step 4/5] LLM Generation & Token Streaming (Model: '{active_model}')")
        t_llm_start = time.perf_counter()
        first_token_time = None
        ttft_ms = 0.0
        token_count = 0
        used_fallback = False
        fallback_reason = None

        def token_stream_generator():
            nonlocal used_fallback, fallback_reason
            if active_model == "direct_synthesis":
                used_fallback = True
                fallback_reason = "User requested Direct Grounded Mode."
                fallback_text = self._generate_fallback_synthesis(query, sources)
                for w in fallback_text.split(" "):
                    yield w + " "
            elif self._is_openrouter_model(active_model):
                try:
                    for tok in self._generate_openrouter_stream(active_model, prompt):
                        yield tok
                except Exception as e:
                    used_fallback = True
                    fallback_reason = str(e)
                    print(f"    ├─ ⚠️ OpenRouter streaming error: {e}")
                    print(f"    ├─ 🔄 Falling back to Direct Grounded Mode synthesis...")
                    fallback_text = self._generate_fallback_synthesis(query, sources)
                    for w in fallback_text.split(" "):
                        yield w + " "
            else:
                try:
                    print(f"    ├─ Calling Ollama client.generate(model='{active_model}', host='{self.ollama_host}', stream=True)...")
                    stream = self.client.generate(model=active_model, prompt=prompt, stream=True)
                    for chunk in stream:
                        tok = chunk.get("response", "")
                        if tok:
                            yield tok
                except Exception as e:
                    used_fallback = True
                    fallback_reason = str(e)
                    print(f"    ├─ ⚠️ Ollama streaming error: {e}")
                    print(f"    ├─ 🔄 Falling back to Direct Grounded Mode synthesis...")
                    fallback_text = self._generate_fallback_synthesis(query, sources)
                    for w in fallback_text.split(" "):
                        yield w + " "

        for token in token_stream_generator():
            if first_token_time is None:
                first_token_time = time.perf_counter()
                ttft_ms = round((first_token_time - t_llm_start) * 1000, 2)
                print(f"    ├─ 💬 Time-To-First-Token (TTFT): {ttft_ms} ms (First token yielded to user)")
            token_count += 1
            yield {"type": "token", "token": token}

        t_llm_end = time.perf_counter()
        total_llm_ms = round((t_llm_end - t_llm_start) * 1000, 2)
        stream_ms = round((t_llm_end - first_token_time) * 1000, 2) if first_token_time else 0.0
        tok_per_sec = round((token_count / (stream_ms / 1000)), 1) if stream_ms > 0 else 0.0

        if ttft_ms == 0.0 and first_token_time is not None:
            ttft_ms = round((first_token_time - t_llm_start) * 1000, 2)

        steps.append({
            "name": "5. LLM Time-To-First-Token (TTFT)",
            "duration_ms": ttft_ms,
            "details": f"model={active_model}, first_token_latency"
        })
        steps.append({
            "name": f"6. LLM Token Streaming ({token_count} tokens)",
            "duration_ms": stream_ms,
            "details": f"{tok_per_sec} tokens/sec"
        })

        if ttft_ms > 1500:
            bottlenecks.append(
                f"Step 5 (LLM Time-To-First-Token) took {ttft_ms:,.2f} ms: "
                f"Network round-trip to remote provider ({active_model}) and initial prompt processing. "
                f"Remote API queue times and model inference cold-start are the primary drivers."
            )

        if used_fallback:
            print(f"    ├─ ⚠️ Fallback Mode Used: {fallback_reason}")
            bottlenecks.append(f"Model generation fell back to Direct Grounded Mode: {fallback_reason}")

        print(f"    └─ 🏁 LLM Streaming Complete: {token_count} tokens in {stream_ms} ms ({tok_per_sec} tok/s) | Total LLM Time: {total_llm_ms} ms")

        # Step 6: Summary & Latency Analysis
        t_overall_end = time.perf_counter()
        total_time_ms = round((t_overall_end - t_overall_start) * 1000, 2)

        # Print full summary report to console
        summary_table = format_timing_report(
            query=query,
            model=active_model,
            total_ms=total_time_ms,
            steps=steps,
            bottlenecks=bottlenecks
        )
        print(summary_table)

        # Build comprehensive developer telemetry
        dev_telemetry = {
            "query": query,
            "total_time_ms": total_time_ms,
            "retrieval_time_ms": retrieval_ms,
            "embedding_time_ms": embedding_ms,
            "llm_time_ms": total_llm_ms,
            "ttft_ms": ttft_ms,
            "tokens_count": token_count,
            "tokens_per_second": tok_per_sec,
            "model_used": active_model,
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
            "prompt_preview": prompt,
            "timing_steps": steps,
            "bottlenecks": bottlenecks,
            "query_embedding": {
                "model": embed_diag.get("provider", (self.embedding_model if self._check_ollama_available() else "chroma-onnx-default")),
                "dimensions": len(query_embedding) if query_embedding else 0,
                "norm": round(math.sqrt(sum(x * x for x in query_embedding)), 4) if query_embedding else 0.0,
                "sample_values": [round(x, 4) for x in query_embedding[:16]] if query_embedding else [],
            },
            "chunks_retrieved_count": len(sources),
            "raw_chunks": [
                {
                    "id": s.get("chunk_id", i),
                    "source": s.get("metadata", {}).get("source", "unknown"),
                    "score": s.get("score"),
                    "raw_distance": s.get("raw_distance"),
                    "content_preview": s["content"][:240] + ("..." if len(s["content"]) > 240 else ""),
                    "char_count": len(s["content"]),
                    "metadata": s.get("metadata", {})
                }
                for i, s in enumerate(sources)
            ]
        }

        yield {"type": "dev_telemetry", "telemetry": dev_telemetry}
        yield {"type": "done"}

    def query(self, query: str, top_k: int = 3, model_override: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous query execution with full step-by-step console timing."""
        t_overall_start = time.perf_counter()
        active_model = model_override or self.llm_model
        steps: List[Dict[str, Any]] = []
        bottlenecks: List[str] = []

        print(f"\n{ANSI_CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗{ANSI_RESET}")
        print(f"{ANSI_CYAN}║ 🚀 [SYNC RAG QUERY] Starting execution                                               ║{ANSI_RESET}")
        print(f"{ANSI_CYAN}╚══════════════════════════════════════════════════════════════════════════════════════╝{ANSI_RESET}")
        print(f"   Query: \"{query}\"")
        print(f"   Model: {active_model} | Top-K: {top_k}")
        print("─" * 88)

        t_step1 = time.perf_counter()
        is_casual = self._is_casual_message(query)
        doc_count = self.collection.count()
        step1_ms = round((time.perf_counter() - t_step1) * 1000, 2)
        steps.append({"name": "1. Intent & KB Check", "duration_ms": step1_ms, "details": f"casual={is_casual}"})

        if not is_casual and doc_count > 0:
            retrieval_data = self.retrieve(query, top_k=top_k, verbose=True)
            raw_sources = retrieval_data["chunks"]
            sources = [s for s in raw_sources if s.get("raw_distance", 2.0) <= 1.45]
            embedding_ms = retrieval_data["embedding_ms"]
            vector_search_ms = retrieval_data.get("vector_search_ms", 0.0)
            steps.append({"name": "2. Query Vector Embedding", "duration_ms": embedding_ms})
            steps.append({"name": "3. ChromaDB Vector Retrieval", "duration_ms": vector_search_ms})
            if embedding_ms > 1000:
                bottlenecks.append(f"Step 2 (Embedding) took {embedding_ms} ms due to remote Ollama network delay.")
        else:
            sources = []
            steps.append({"name": "2. Query Vector Embedding", "duration_ms": 0.0})
            steps.append({"name": "3. ChromaDB Vector Retrieval", "duration_ms": 0.0})

        t_prompt = time.perf_counter()
        prompt = self.build_prompt(query, sources if sources else None)
        prompt_ms = round((time.perf_counter() - t_prompt) * 1000, 2)
        steps.append({"name": "4. Context & Prompt Assembly", "duration_ms": prompt_ms})

        t_llm = time.perf_counter()
        if active_model == "direct_synthesis":
            ans = self._generate_fallback_synthesis(query, sources)
        elif self._is_openrouter_model(active_model):
            try:
                ans = self._generate_openrouter_sync(active_model, prompt)
            except Exception as e:
                print(f"   ⚠️ OpenRouter error: {e}. Falling back to synthesis.")
                ans = self._generate_fallback_synthesis(query, sources)
        else:
            try:
                print(f"   ├─ Calling Ollama client.generate(model='{active_model}')...")
                res = self.client.generate(model=active_model, prompt=prompt, stream=False)
                ans = res.get("response", "")
            except Exception as e:
                print(f"   ⚠️ Ollama error: {e}. Falling back to synthesis.")
                ans = self._generate_fallback_synthesis(query, sources)

        llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)
        steps.append({"name": "5. LLM Response Generation", "duration_ms": llm_ms})
        if llm_ms > 1500:
            bottlenecks.append(f"Step 5 (LLM Generation) took {llm_ms} ms waiting for remote API completion.")

        total_ms = round((time.perf_counter() - t_overall_start) * 1000, 2)
        print(format_timing_report(query, active_model, total_ms, steps, bottlenecks))

        return {"answer": ans, "sources": sources, "timing_ms": total_ms}

    def get_all_chunks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch indexed chunks from ChromaDB for developer inspection."""
        count = self.collection.count()
        if count == 0:
            return []

        data = self.collection.get(
            limit=limit,
            include=["documents", "metadatas"]
        )

        chunks_list = []
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        ids = data.get("ids", [])

        for i in range(len(docs)):
            chunks_list.append({
                "id": ids[i] if i < len(ids) else f"chunk_{i}",
                "content": docs[i],
                "char_count": len(docs[i]),
                "metadata": metas[i] if i < len(metas) else {}
            })

        return chunks_list

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics on currently indexed documents."""
        count = self.collection.count()
        sources = set()
        if count > 0:
            metas = self.collection.get(include=["metadatas"])["metadatas"]
            for m in metas:
                if m and "source" in m:
                    sources.add(m["source"])

        return {
            "total_chunks": count,
            "unique_documents": list(sources),
            "collection_name": self.collection_name,
            "ollama_online": self._check_ollama_available(),
            "ollama_host": self.ollama_host,
            "is_cloud": "ollama.com" in self.ollama_host,
            "default_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "openrouter_configured": bool(self.openrouter_api_key),
            "persist_dir": self.persist_dir
        }

    def clear_index(self):
        """Wipe the current collection and reset."""
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document embeddings collection"}
        )
